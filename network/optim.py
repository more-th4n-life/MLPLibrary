import numpy as np
from layer import Linear

class SGD:
    """
    Stochastic Gradient Descent: Also implements weight decay (can be removed by setting to zero)

    To add momentum etc. just adds another terms
    """
    def __init__(self, learning_rate=0.04, weight_decay = 0, momentum = 0.5):
        self.lr, self.wd, self.momentum = learning_rate, weight_decay, momentum
        
    def step(self, network):
        """
        Update weights and biases in each layer wrt learned dW and db AND learning rate (+ weight decay) term
        """
        for layer in [l for l in network.layers if isinstance(l, Linear)]:
    
            layer.diffW *= self.momentum
            layer.diffb *= self.momentum

            layer.diffW = self.lr * (-layer.dW - self.wd * layer.W)
            layer.diffb = -self.lr * np.mean(layer.db, axis=0)

            layer.W += layer.diffW
            layer.b += layer.diffb

            