import torch as tc
from .layer import Layer
from .loss_functions import loss_functions
import matplotlib.pyplot as plt
import numpy as np
import random
import math

class FFNN:
    def __init__(self, layer_sizes, activations_list,
                 loss_function='mse',
                 weight_inits='random_uniform',
                 init_params_list=None,
                 regularization='none', req_lambda=0.01, use_rmsnorm=False):
        if loss_function not in loss_functions:
            raise NotImplementedError(f"Loss function '{loss_function}' tidak dikenali.")
        self.loss_func, self.loss_grad_func = loss_functions[loss_function]
        self.regularization = regularization
        self.reg_lambda = req_lambda

        self.loss_name = loss_function
        self.layers: list[Layer] = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(
                input_dim=layer_sizes[i],
                output_dim=layer_sizes[i + 1],
                activation_name=activations_list[i],
                weight_init=weight_inits[i] if weight_inits else 'random_uniform',
                init_params=init_params_list[i] if init_params_list else None,
                use_rmsnorm=use_rmsnorm
            )
            self.layers.append(layer)

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def compute_loss_with_regularization(self, y_true, y_pred):
        loss = self.loss_func(y_true, y_pred)
        if self.regularization == 'L1':
            reg_loss = 0.0
            for layer in self.layers:
                reg_loss += tc.sum(tc.abs(layer.weights)) / layer.input.shape[0]
            loss += self.reg_lambda * reg_loss
        elif self.regularization == 'L2':
            reg_loss = 0.0
            for layer in self.layers:
                reg_loss += tc.sum(layer.weights ** 2) / layer.input.shape[0]
            loss += (self.reg_lambda / 2) * reg_loss
        return loss

    def backward(self, y_true, y_pred):
        dO = self.loss_grad_func(y_true, y_pred)
        for layer in reversed(self.layers):
            dO = layer.backward(dO)
            m = layer.input.shape[0]
            if self.regularization == 'L1':
                layer.grad_weights += self.reg_lambda * tc.sign(layer.weights) / m
            elif self.regularization == 'L2':
                layer.grad_weights += 2*(self.reg_lambda * layer.weights / m)

    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, learning_rate=0.01, verbose=1, tol=1e-4, patience=10, stop_in_convergence=False):
        history = {'train_loss': [], 'val_loss': []}
        num_samples = X_train.shape[0]
        
        best_loss = float('inf')
        no_improve_count = 0

        for epoch in range(epochs):
            # indices =  tc.random.permutation(num_samples)
            indices =  tc.randperm(num_samples)
            
            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0.0

            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                y_pred = self.forward(X_batch)
                loss = self.compute_loss_with_regularization(y_batch, y_pred)
                epoch_loss += loss.item() * X_batch.shape[0]

                self.backward(y_batch, y_pred)
                self.update_weights(learning_rate)

            epoch_loss /= num_samples
            history['train_loss'].append(epoch_loss)

            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self.compute_loss_with_regularization(y_val, y_val_pred).item()
                history['val_loss'].append(val_loss)

                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} "
                          f"- Train Loss: {epoch_loss:.8f} - Val Loss: {val_loss:.8f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.8f}")
            
            # if stop_in_convergence:    
            #     if abs(epoch_loss - best_loss) < tol:
            #         no_improve_count += 1
            #     else:
            #         no_improve_count = 0
            #         best_loss = epoch_loss

            #     if no_improve_count >= patience:
            #         if verbose:
            #             print(f"Early stopping triggered at epoch {epoch+1} - loss did not improve more than {tol} for {patience} consecutive epochs.")
            #         break

        return history
    def save(self, file_path):
        layer_sizes = []
        if self.layers:
            layer_sizes.append(self.layers[0].weights.shape[0])
            for layer in self.layers:
                layer_sizes.append(layer.weights.shape[1])
        
        state = {
            'layer_sizes': layer_sizes,
            'activations': [layer.activation_name for layer in self.layers],
            'loss_name': self.loss_name,
            'weight_init': self.layers[0].weight_init if self.layers else 'random_uniform',
            'weights': [layer.weights.detach() for layer in self.layers],
            'biases': [layer.biases.detach() for layer in self.layers],
        }
        tc.save(state, file_path)
        print(f"Model saved to {file_path}")

    @classmethod
    def load(cls, file_path):
        checkpoint = tc.load(file_path, weights_only=False)
        
        model = cls(
            layer_sizes=checkpoint['layer_sizes'],
            activations_list=checkpoint['activations'],
            loss_function=checkpoint['loss_name'],
            weight_init=checkpoint.get('weight_init', 'random_uniform')
        )
        
        for i, layer in enumerate(model.layers):
            layer.weights = checkpoint['weights'][i]
            layer.biases = checkpoint['biases'][i]
        
        print(f"Model loaded from {file_path}")
        return model
        
    def summary(self):
        print("Model Summary:")
        for idx, layer in enumerate(self.layers):
            print(f"Layer {idx+1}:")
            layer.summary()
    def plot_gradients_distribution(self, layers_to_plot,bins=100, normalized=True):
        if not layers_to_plot:
            print("Tidak ada layer yang dipilih untuk plot distribusi bobot.")
            return
        plt.figure(figsize=(12, 8))
        for layer_index in layers_to_plot:
            if layer_index < 0 or layer_index >= len(self.layers):
                print(f"Layer index {layer_index} berada di luar jangkauan. Abaikan.")
                continue
            weights = self.layers[layer_index].grad_weights
            if isinstance(weights, tc.Tensor):
                weights = weights.detach().cpu().numpy()
            else:
                weights = np.array(weights)
            weights_flat = weights.flatten()
            if normalized:
                norm_weights = np.ones_like(weights_flat) / len(weights_flat)
                plt.hist(weights_flat, bins=bins, weights=norm_weights, alpha=0.5, label=f"Layer {layer_index}")
            else:
                plt.hist(weights_flat, bins=bins, alpha=0.5, label=f"Layer {layer_index}")

        plt.title("Distribusi Bobot Gradien Tiap Layer")
        plt.xlabel("Nilai Bobot")
        plt.ylabel("Frekuensi (Persentase)" if normalized else "Jumlah Instance")
        plt.legend()
        plt.grid(True)
        plt.show()
    def plot_weight_distribution(self, layers_to_plot,bins=100, normalized=True):
        if not layers_to_plot:
            print("Tidak ada layer yang dipilih untuk plot distribusi bobot.")
            return
        plt.figure(figsize=(12, 8))
        for layer_index in layers_to_plot:
            if layer_index < 0 or layer_index >= len(self.layers):
                print(f"Layer index {layer_index} berada di luar jangkauan. Abaikan.")
                continue
            weights = self.layers[layer_index].weights
            if isinstance(weights, tc.Tensor):
                weights = weights.detach().cpu().numpy()
            else:
                weights = np.array(weights)
            weights_flat = weights.flatten()
            if normalized:
                norm_weights = np.ones_like(weights_flat) / len(weights_flat)
                plt.hist(weights_flat, bins=bins, weights=norm_weights, alpha=0.5, label=f"Layer {layer_index}")
            else:
                plt.hist(weights_flat, bins=bins, alpha=0.5, label=f"Layer {layer_index}")
        plt.title("Distribusi Bobot Tiap Layer")
        plt.xlabel("Nilai Bobot")
        plt.ylabel("Frekuensi (Persentase)" if normalized else "Jumlah Instance")
        plt.legend()
        plt.grid(True)
        plt.show()
    def plot_network_structure(self, limited_size):
        if not self.layers:
            print("Model tidak memiliki layer.")
            return
        
        plt.figure(figsize=(20, 14))
        light_colors = [
            '#FFD3B6',  
            '#DCEDC1',  
            '#A8E6CE',  
            '#FFAAA5',  
            '#D5E5F2',  
            '#E8E1EF',  
            '#FFF1BD', 
            '#C7CEEA'   
        ]
        
        layer_weights = []
        layer_grad_weights = []
        layer_biases = []
        layer_grad_biases = []
        layer_sizes = []
        
        for layer in self.layers:
            if isinstance(layer.weights, tc.Tensor):
                weights = layer.weights.detach().cpu().numpy()
            else:
                weights = np.array(layer.weights)
            layer_weights.append(weights)
            if isinstance(layer.grad_weights, tc.Tensor):
                grad_weights = layer.grad_weights.detach().cpu().numpy()
            else:
                grad_weights = np.array(layer.grad_weights)
            layer_grad_weights.append(grad_weights)
            if isinstance(layer.biases, tc.Tensor):
                biases = layer.biases.detach().cpu().numpy().flatten()
            else:
                biases = np.array(layer.biases).flatten()
            layer_biases.append(biases)
            if isinstance(layer.grad_biases, tc.Tensor):
                grad_biases = layer.grad_biases.detach().cpu().numpy().flatten()
            else:
                grad_biases = np.array(layer.grad_biases).flatten()
            layer_grad_biases.append(grad_biases)
            if len(layer_sizes) == 0:
                layer_sizes.append(weights.shape[0])
                layer_sizes.append(weights.shape[1])
            else:
                layer_sizes.append(weights.shape[1])
        
        limited_nodes = []
        for size in layer_sizes:
            if size <= limited_size:
                limited_nodes.append(list(range(size)))
            else:
                limited_nodes.append(random.sample(range(size), limited_size))
        pos = {}
        layer_spacing = 1.5  
        
        for layer_idx, nodes in enumerate(limited_nodes):
            for i, node_idx in enumerate(nodes):
                x = layer_idx * layer_spacing
                y = (i - len(nodes)/2) * 0.7 
                pos[(layer_idx, node_idx)] = (x, y)
        
        for layer_idx in range(len(limited_nodes) - 1):  
            lowest_y = min([pos[(layer_idx, node_idx)][1] for node_idx in limited_nodes[layer_idx]])
            bias_y = lowest_y - 0.8 
            pos[(layer_idx, 'bias')] = (layer_idx * layer_spacing, bias_y)
        
        for layer_idx, nodes in enumerate(limited_nodes):
            for node_idx in nodes:
                plt.scatter(*pos[(layer_idx, node_idx)], s=2000, c='skyblue', zorder=2, edgecolor='black')
                plt.text(*pos[(layer_idx, node_idx)], f"{layer_idx}-{node_idx}", 
                        ha='center', va='center', fontsize=10, fontweight='bold')
        
        for layer_idx in range(len(limited_nodes) - 1):
            bias_pos = pos[(layer_idx, 'bias')]
            plt.scatter(*bias_pos, s=2000, c='lightgreen', zorder=2, edgecolor='black')
            plt.text(*bias_pos, f"Bias {layer_idx}", ha='center', va='center', fontsize=10, fontweight='bold')
        for layer_idx in range(len(limited_nodes) - 1):
            source_nodes = limited_nodes[layer_idx]
            target_nodes = limited_nodes[layer_idx + 1]
            
            for s_idx in source_nodes:
                for t_idx in target_nodes:
                    weight_value = layer_weights[layer_idx][s_idx, t_idx]
                    grad_weight_value = layer_grad_weights[layer_idx][s_idx, t_idx]
                    
                    edge_color = random.choice(light_colors)
                    start_pos = pos[(layer_idx, s_idx)]
                    end_pos = pos[(layer_idx + 1, t_idx)]
                    
                    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                            color=edge_color, alpha=0.8, zorder=1, linewidth=2)
                    
                    mid_x = (start_pos[0] + end_pos[0]) / 2
                    mid_y = (start_pos[1] + end_pos[1]) / 2
                    offset_x = random.uniform(-0.2, 0.2)
                    offset_y = random.uniform(-0.05, 0.05)
                    
                    angle = math.degrees(math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0]))
                    if angle > 90:
                        angle -= 180
                    elif angle < -90:
                        angle += 180
                        
                    weight_text = f"w = {weight_value:.4f} g = {grad_weight_value:.4f}"
                    plt.text(mid_x + offset_x, mid_y + offset_y, weight_text, 
                            fontsize=7, ha='center', va='center', color='black',
                            bbox=dict(facecolor=edge_color, alpha=0.9, pad=1, boxstyle='round'))
        for layer_idx in range(len(limited_nodes) - 1):
            bias_pos = pos[(layer_idx, 'bias')]
            target_nodes = limited_nodes[layer_idx + 1]
            
            for t_idx_pos, t_idx in enumerate(target_nodes):
                bias_value = layer_biases[layer_idx][t_idx]  
                grad_bias_value = layer_grad_biases[layer_idx][t_idx]  
                
                edge_color = random.choice(light_colors)
                end_pos = pos[(layer_idx + 1, t_idx)]
                
                plt.plot([bias_pos[0], end_pos[0]], [bias_pos[1], end_pos[1]], 
                        color=edge_color, alpha=0.8, zorder=1, linewidth=2, linestyle='--')
                
                mid_x = (bias_pos[0] + end_pos[0]) / 2
                mid_y = (bias_pos[1] + end_pos[1]) / 2
                offset_x = random.uniform(-0.1, 0.1)
                offset_y = random.uniform(-0.1, 0.1)
                
                bias_text = f"b = {bias_value:.4f} g = {grad_bias_value:.4f}"
                plt.text(mid_x + offset_x, mid_y + offset_y, bias_text, 
                        fontsize=7, ha='center', va='center', color='black',
                        bbox=dict(facecolor=edge_color, alpha=0.9, pad=1, boxstyle='round4'))

        plt.title('Neural Network Structure with Biases')
        plt.axis('off')
        plt.tight_layout()
        plt.show()