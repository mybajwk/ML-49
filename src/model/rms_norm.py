import torch as tc
class RMSNorm:
    def __init__(self, num_features, eps=1e-5):
        self.eps = eps
        self.gamma = tc.ones(num_features)
        self.beta = tc.zeros(num_features)
        self.grad_gamma = tc.zeros_like(self.gamma)
        self.grad_beta = tc.zeros_like(self.beta)
    
    def forward(self, x):
        # x: tensor dengan shape (batch, features)
        self.x = x  # simpan untuk backward
        self.mean_square = tc.mean(x**2, dim=-1, keepdim=True)
        self.rms = tc.sqrt(self.mean_square + self.eps)
        # gamma dan beta akan di-broadcast agar sesuai dengan shape x
        return self.gamma * (x / self.rms) + self.beta

    def backward(self, x, dout):
        # x: input ke RMSNorm (nilai sebelum normalisasi)
        # dout: gradien dari output, dengan shape (batch, features)
        normalized = x / self.rms
        self.grad_gamma = (dout * normalized).sum(dim=0)
        self.grad_beta = dout.sum(dim=0)
        dnormalized = dout * self.gamma
        n = x.shape[-1]
        dx = dnormalized / self.rms - x * ((dnormalized * x).sum(dim=-1, keepdim=True)) / (n * self.rms**3)
        return dx
    def update(self, learning_rate):
        self.gamma -= learning_rate * self.grad_gamma
        self.beta -= learning_rate * self.grad_beta