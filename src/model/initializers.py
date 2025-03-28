import torch as tc

class WeightInitializer:
    def initialize(input_dim, output_dim, method=None, init_params=None, activation=None):
        init_params = init_params or {}
        
        seed = init_params.get('seed', None)
        if seed is not None:
            tc.manual_seed(seed)
        
        if method == "he_xavier":
            if activation in ['relu', 'leaky_relu']:
                method = 'he'  
            elif activation in ['sigmoid', 'tanh', 'softmax']:
                method = 'xavier'  
            else:
                method = 'random_uniform'
        if method == 'zero':
            return tc.zeros((input_dim, output_dim))
        elif method == 'random_uniform':
            lower = init_params.get('lower', -0.5)
            upper = init_params.get('upper', 0.5)
            # return tc.random.uniform(lower, upper, (input_dim, output_dim))
            return tc.empty((input_dim, output_dim)).uniform_(lower, upper)
        elif method == 'random_normal':
            mean = init_params.get('mean', 0.0)
            variance = init_params.get('variance', 1.0)
            std = variance ** 0.5
            # return tc.random.normal(mean, std, (input_dim, output_dim))
            return tc.randn((input_dim, output_dim)) * std + mean
        elif method in ['xavier', 'he']:  
            if activation in ['relu', 'leaky_relu']:
                std = (2 / input_dim) ** 0.5  
                return tc.randn((input_dim, output_dim)) * std
            elif activation in ['sigmoid', 'tanh', 'softmax']:
                limit = (6 / (input_dim + output_dim)) ** 0.5  
                return tc.empty((input_dim, output_dim)).uniform_(-limit, limit)
            else:
                raise ValueError(f"Inisialisasi '{method}' tidak cocok untuk aktivasi '{activation}'")
        else:
            raise ValueError("Metode inisialisasi bobot tidak dikenali")
