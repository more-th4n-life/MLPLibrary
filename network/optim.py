import numpy as np
from layer import Layer

class SGD:
    def __init__(self, learning_rate, weight_decay):
        self.lr, self.wd = learning_rate, weight_decay
        
    def step(self, network):
        layers = [l for l in network.layers if isinstance(l, Layer)]

        for layer in layers:

            layer.W += self.lr * (-layer.dW - self.wd * layer.W) 
            layer.b += -self.lr * layer.db 