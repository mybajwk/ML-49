import torch as tc

class WeightInitializer:
    def initialize(input_dim, output_dim, method='random_uniform', init_params=None):
        init_params = init_params or {}
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
        else:
            raise ValueError("Metode inisialisasi bobot tidak dikenali")
