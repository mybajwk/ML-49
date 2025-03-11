import torch as tc


class Activations:
    
    def linear(x):
        return x

    
    def d_linear(x):
        return tc.ones_like(x)

    
    def relu(x):
        # return tc.maximum(x, tc.zeros_like(x))
        return tc.max(x, tc.zeros_like(x))

    
    def d_relu(x):
        # return (x > 0).astype(tc.float32)
        return (x > 0).float()

    
    def sigmoid(x):
        return 1 / (1 + tc.exp(-x))

    
    def d_sigmoid(x):
        s = Activations.sigmoid(x)
        return s * (1 - s)

    
    def tanh(x):
        return tc.tanh(x)

    
    def d_tanh(x):
        return 1 - tc.tanh(x) ** 2

    
    def softmax(x):
        # ex = tc.exp(x - tc.max(x, axis=1, keepdim=True)[0])
        # return ex / tc.sum(ex, axis=1, keepdim=True)
        ex = tc.exp(x - tc.max(x, dim=1, keepdim=True)[0])
        return ex / tc.sum(ex, dim=1, keepdim=True)

activation_functions = {
    'linear': (Activations.linear, Activations.d_linear),
    'relu': (Activations.relu, Activations.d_relu),
    'sigmoid': (Activations.sigmoid, Activations.d_sigmoid),
    'tanh': (Activations.tanh, Activations.d_tanh),
    'softmax': (Activations.softmax, None)  
}