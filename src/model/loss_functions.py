import torch as tc

class LossFunctions:

    def mse(y_true, y_pred):
  
        return 0.5 * tc.mean((y_true - y_pred) ** 2)

    def d_mse(y_true, y_pred):
        
        return -(y_true - y_pred)  

    def binary_cross_entropy(y_true, y_pred, eps=1e-9):
        
        return -tc.mean( 
            y_true * tc.log(y_pred + eps) + 
            (1 - y_true) * tc.log(1 - y_pred + eps)
        )

    def d_binary_cross_entropy(y_true, y_pred, eps=1e-9):
        
        return (y_pred - y_true) / ( (y_pred*(1 - y_pred)) + eps )

    def categorical_cross_entropy(y_true, y_pred, eps=1e-9):
       
        return -tc.mean(tc.sum(y_true * tc.log(y_pred + eps), dim=1))

    def d_categorical_cross_entropy(y_true, y_pred):
        
        return (y_pred - y_true)


loss_functions = {
    'mse': (
        LossFunctions.mse,
        LossFunctions.d_mse
    ),
    'bce': (
        LossFunctions.binary_cross_entropy,
        LossFunctions.d_binary_cross_entropy
    ),
    'cce': (
        LossFunctions.categorical_cross_entropy,
        LossFunctions.d_categorical_cross_entropy
    ),
}
