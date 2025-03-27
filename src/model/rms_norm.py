import torch as tc
class RMSNorm:
    def __init__(self, num_features, eps=1e-8):
        self.eps = eps
        self.gamma = tc.ones(num_features)  
        self.beta = tc.zeros(num_features)  

    def forward(self, x):
        self.rms = tc.sqrt(tc.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / self.rms) + self.beta
    
    def backward(self, x, error_term, m):
        normalized = x / self.rms
        self.grad_gamma = (error_term * normalized).sum(dim=0) / m
        self.grad_beta = error_term.sum(dim=0)
        dnormalized = error_term * self.gamma
        n = x.shape[-1]
        new_error_term = dnormalized / self.rms - x * ((dnormalized * x).sum(dim=-1, keepdim=True)) / (n * self.rms**3)
        return new_error_term
    
    def update(self, learning_rate):
        self.gamma -= learning_rate * self.grad_gamma
        self.beta -= learning_rate * self.grad_beta