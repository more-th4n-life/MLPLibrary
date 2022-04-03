import numpy as np
from layer import Linear


class SGD:
    """
    Stochastic Gradient Descent: Also implements weight decay (can be removed by setting to zero)

    To add momentum etc. just adds another terms
    """
    def __init__(self, learning_rate=0.04, weight_decay = 0, momentum = 0.5):
        self.lr, self.wd, self.momentum = learning_rate, weight_decay, momentum

    def __repr__(self):
        if self.wd and self.momentum:
            return "SGD(wd+mm)"
        elif self.wd:
            return "SGD(wd)"
        elif self.momentum:
            return "SGD(mm)"
        return "SGD"
        
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


class Adam:

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-09):

        self.lr, self.beta1, self.beta2 = learning_rate, beta1, beta2
        
        self.v, self.v_corr = {}, {}
        self.s, self.s_corr = {}, {}

        self.epsilon = epsilon

    def __repr__(self):
        return "Adam"

    def add_to_dict(self, indim, outdim, i):

        self.v["dW" + str(i)] = np.zeros((indim, outdim))
        self.v["db" + str(i)] = np.zeros(outdim,)

        self.s["dW" + str(i)] = np.zeros((indim, outdim))
        self.s["db" + str(i)] = np.zeros(outdim,)
    
    def step(self, network, t):

        for k, layer in enumerate([L for L in network.layers if isinstance(L, Linear)]):
            
            i = k + 1

            # calculate moving avg of gradients 
            self.v["dW" + str(i)] = self.beta1 * self.v["dW" + str(i)] + (1 - self.beta1) * layer.dW
            self.v["db" + str(i)] = self.beta1 * self.v["db" + str(i)] + (1 - self.beta1) * layer.db

            # correct bias with first moment estimates
            self.v_corr["dW" + str(i)] = self.v["dW" + str(i)] / (1 - self.beta1 ** t)
            self.v_corr["db" + str(i)] = self.v["db" + str(i)] / (1 - self.beta1 ** t)

            # calculate moving avg of squared gradients
            self.s["dW" + str(i)] = self.beta2 * self.s["dW" + str(i)] + (1 - self.beta2) * (np.square(layer.dW))
            self.s["db" + str(i)] = self.beta2 * self.s["db" + str(i)] + (1 - self.beta2) * (np.square(layer.db))

            # correct bias with second moment estimates
            self.s_corr["dW" + str(i)] = self.s["dW" + str(i)] / (1 - self.beta2 ** t)
            self.s_corr["db" + str(i)] = self.s["db" + str(i)] / (1 - self.beta2 ** t)

            # update params
            layer.W += -self.lr * self.v_corr["dW" + str(i)] / (np.sqrt(self.s_corr["dW" + str(i)]) + self.epsilon)
            layer.b += -self.lr * self.v_corr["db" + str(i)] / (np.sqrt(self.s_corr["db" + str(i)]) + self.epsilon)

