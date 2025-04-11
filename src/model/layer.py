import torch as tc
import matplotlib.pyplot as plt
from .activations import activation_functions
from .initializers import WeightInitializer
from .rms_norm import RMSNorm

class Layer:
    def __init__(self, input_dim, output_dim, activation_name,
                 weight_init='random_uniform', init_params=None, use_rmsnorm=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation_name
        self.weight_init = weight_init
        self.init_params = init_params
        
        self.use_rmsnorm = use_rmsnorm  
        
        self.weights = WeightInitializer.initialize(input_dim, output_dim,
                                                    method=weight_init, init_params=init_params, activation=activation_name)
        self.biases = WeightInitializer.initialize(1, output_dim,
                                                    method=weight_init, init_params=init_params, activation=activation_name)
        self.activation, self.d_activation, self.d_activation_times_vector = activation_functions[activation_name]
        
        if self.use_rmsnorm:
            self.rmsnorm = RMSNorm(output_dim)  
        
        self.input = None
        self.net = None  
        self.out = None

    def forward(self, X):
        self.input = X
        self.net = X @ self.weights + self.biases  
        if self.use_rmsnorm:
            self.normalized_net = self.rmsnorm.forward(self.net)
            self.out = self.activation(self.normalized_net)
        else:
            self.out = self.activation(self.net)
        return self.out

    
    def backward(self, dO):
        m = self.input.shape[0]
        if self.activation_name == 'softmax':
            error_term = self.d_activation_times_vector(self.out, dO)
        else:
            error_term = dO * self.d_activation(self.net)
        if self.use_rmsnorm:
            d_net = self.rmsnorm.backward(error_term)
        else:
            d_net = error_term
        self.grad_weights = (self.input.t() @ d_net) / m
        self.grad_biases = tc.sum(d_net, dim=0, keepdims=True) / m
        dO_prev = d_net @ self.weights.t()
        return dO_prev

    
    def update(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases
        if self.use_rmsnorm:
            self.rmsnorm.update(learning_rate)

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