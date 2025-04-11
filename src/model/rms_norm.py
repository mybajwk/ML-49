import torch as tc
class RMSNorm:
    def __init__(self, num_features, eps=1e-5):
        self.eps = eps
        self.gamma = tc.ones(num_features)  
        self.beta = tc.zeros(num_features)  

    def forward(self, x):
        self.input = x
        self.rms = tc.sqrt(tc.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / self.rms) + self.beta
    
    def backward(self, grad_output):
        normalized = self.input / self.rms
        self.grad_gamma = (grad_output * normalized).sum(dim=0)
        self.grad_beta = grad_output.sum(dim=0)
        dnormalized = grad_output * self.gamma
        n = self.input.shape[-1]
        grad_input = dnormalized / self.rms - self.input * ((dnormalized * self.input).sum(dim=-1, keepdim=True)) / (n * self.rms**3)
        return grad_input
    
    def update(self, learning_rate):
        self.gamma -= learning_rate * self.grad_gamma
        self.beta -= learning_rate * self.grad_beta
        return