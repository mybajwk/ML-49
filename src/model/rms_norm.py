import torch as tc
class RMSNorm:
    def __init__(self, num_features, eps=1e-5):
        self.eps = eps
        self.gamma = tc.ones(num_features)  
        self.beta = tc.zeros(num_features)  

    def forward(self, x):
        rms = tc.sqrt(tc.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms) + self.beta