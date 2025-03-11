import torch as tc
import matplotlib.pyplot as plt
from .activations import activation_functions
from .initializers import WeightInitializer

class Layer:
    def __init__(self, input_dim, output_dim, activation_name,
                 weight_init='random_uniform', init_params=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation_name
        self.weight_init = weight_init
        self.init_params = init_params

        self.weights = WeightInitializer.initialize(input_dim, output_dim,
                                                    method=weight_init, init_params=init_params)
        self.biases = tc.zeros((1, output_dim))
        self.activation, self.d_activation = activation_functions[activation_name]

        self.input = None
        self.net = None  
        self.out= None  

        self.grad_weights = tc.zeros_like(self.weights)
        self.grad_biases = tc.zeros_like(self.biases)

    def forward(self, X):
        self.input = X
        self.net = X @ self.weights + self.biases  
        self.out= self.activation(self.net)
        return self.out

    def backward(self, dO):
        m = self.input.shape[0]
        if self.activation_name == 'softmax':
            error_term = dO
        else:
            error_term = dO * self.d_activation(self.net)
        # self.grad_weights = (self.input.T @ error_term) / m
        self.grad_weights = (self.input.t() @ error_term) / m
        # self.grad_biases = tc.sum(error_term, axis=0, keepdims=True) / m
        self.grad_biases = tc.sum(error_term, dim=0, keepdims=True) / m
        # dO_prev = error_term @ self.weights.T
        dO_prev = error_term @ self.weights.t()
        return dO_prev
    
    def update(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases

    def summary(self):
        print(f"Layer: {self.input_dim} -> {self.output_dim} | Activation: {self.activation_name}")
        print("Weights:")
        print(self.weights)
        print("Biases:")
        print(self.biases)
        print("Gradients (Weights):")
        print(self.grad_weights)
        print("Gradients (Biases):")
        print(self.grad_biases)
        print("-" * 50)

    def plot_weight_distribution(self):
        plt.hist(self.weights.flatten().cpu().numpy(), bins=30)
        plt.title("Distribusi Bobot")
        plt.xlabel("Nilai Bobot")
        plt.ylabel("Frekuensi")
        plt.show()

    def plot_gradient_distribution(self):
        plt.hist(self.grad_weights.flatten().cpu().numpy(), bins=30)
        plt.title("Distribusi Gradien Bobot")
        plt.xlabel("Nilai Gradien")
        plt.ylabel("Frekuensi")
        plt.show()
