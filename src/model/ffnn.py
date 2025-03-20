import torch as tc
from .layer import Layer
from .loss_functions import loss_functions
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import random
import math
class FFNN:
    def __init__(self, layer_sizes, activations_list,
                 loss_function='mse', weight_init='random_uniform', init_params=None):
        if loss_function not in loss_functions:
            raise NotImplementedError(f"Loss function '{loss_function}' tidak dikenali.")
        self.loss_func, self.loss_grad_func = loss_functions[loss_function]

        self.loss_name = loss_function
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(
                layer_sizes[i], 
                layer_sizes[i+1],
                activations_list[i],
                weight_init=weight_init,
                init_params=init_params
            )
            self.layers.append(layer)

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output


    def backward(self, y_true, y_pred):
        dO = self.loss_grad_func(y_true, y_pred)
        for layer in reversed(self.layers):
            dO = layer.backward(dO)

    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, learning_rate=0.01, verbose=1, tol=1e-4, patience = 10):
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
                loss = self.loss_func(y_batch, y_pred)
                epoch_loss += loss.item() * X_batch.shape[0]

                self.backward(y_batch, y_pred)
                self.update_weights(learning_rate)

            epoch_loss /= num_samples
            history['train_loss'].append(epoch_loss)

            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_func(y_val, y_val_pred).item()
                history['val_loss'].append(val_loss)

                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} "
                          f"- Train Loss: {epoch_loss:.8f} - Val Loss: {val_loss:.8f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.8f}")
                    
            if abs(epoch_loss - best_loss) < tol:
                no_improve_count += 1
            else:
                no_improve_count = 0
                best_loss = epoch_loss
            
            if no_improve_count >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1} - loss did not improve more than {tol} for {patience} consecutive epochs.")
                break

        return history
    
    def summary(self):
        print("Model Summary:")
        for idx, layer in enumerate(self.layers):
            print(f"Layer {idx+1}:")
            layer.summary()

    def plot_weights(self, layer_indices):
        for idx in layer_indices:
            if idx < len(self.layers):
                self.layers[idx].plot_weight_distribution()
            else:
                print(f"Layer index {idx} tidak ada.")
    def plot_weight_distribution(self):
        plt.hist(self.weights.detach().cpu().numpy().flatten(), bins=30)
        plt.title("Distribusi Bobot")
        plt.xlabel("Nilai Bobot")
        plt.ylabel("Frekuensi")
        plt.show()
    def plot_network_structure(self,limited_size):
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
        for layer_idx, nodes in enumerate(limited_nodes):
            for node_idx in nodes:
                plt.scatter(*pos[(layer_idx, node_idx)], s=2000, c='skyblue', zorder=2, edgecolor='black')
                plt.text(*pos[(layer_idx, node_idx)], f"{layer_idx}-{node_idx}", 
                        ha='center', va='center', fontsize=10, fontweight='bold')
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

        plt.title('Neural Network Structure')
        plt.axis('off')
        plt.tight_layout()
        plt.show()