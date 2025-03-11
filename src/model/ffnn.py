import torch as tc
from .layer import Layer
from .loss_functions import loss_functions

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
              epochs=100, batch_size=32, learning_rate=0.01, verbose=1, tol=1e-6, patience = 10):
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
                          f"- Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}")
                    
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

    def plot_gradients(self, layer_indices):
        for idx in layer_indices:
            if idx < len(self.layers):
                self.layers[idx].plot_gradient_distribution()
            else:
                print(f"Layer index {idx} tidak ada.")
