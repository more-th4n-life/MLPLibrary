from itertools import dropwhile
from matplotlib.pyplot import axis
import numpy as np

def xavier(size, gain=1):
    """
    Helper function for Xavier initialisation
    """
    i = gain * np.sqrt(6. / np.sum(size))
    return np.random.uniform(low = -i, high = i, size = size)


class Layer():
    """
    Abstract class for layer: used to differentiate b/w activations
    i.e. For updating weights and bias 
    """
    def __init__(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass
    def update(self):
        pass

class Linear(Layer):

    def __init__(self, indim, outdim, dropout=0):
        """
        Layer: Linear y = xw + b

        Args
            indim: incoming n features
            outdim: outgoing m features
        """
        self.indim, self.outdim = indim, outdim
        self.batch_norm, self.alpha = False, None     # set in network as parameter
        self.dropout = dropout
        self.epsilon = 1e-10
        
        # initialize weights and bias (Xavier init)
        self.W = xavier((indim, outdim))
        self.b = xavier((1, outdim))

        self.dW = np.zeros((indim, outdim))
        self.db = np.zeros(outdim,)

        # momentum terms
        self.diffW = np.zeros((indim, outdim))
        self.diffb = np.zeros(outdim,)

        # batch norm
        self.gamma = np.ones(int(np.prod(outdim)),)
        self.beta = np.zeros(int(np.prod(outdim)),)
        self.run_var = np.ones(outdim,)
        self.run_mean = np.zeros(outdim,)

    def batch_norm_forward(self, x, predict=False):
        """
        Batch norm forward pass using linear layer output
        If predict (validation / test): 
            Use running mean / var from training to compute norm
        """
        if predict:
            self.norm = (x - self.run_mean) / np.sqrt(self.run_var + self.epsilon)

        else:
            self.batch_sd = np.sqrt(np.var(x, axis=0) + self.epsilon)
            self.norm = (x - np.mean(x, axis=0)) / self.batch_sd

            self.run_var *= self.alpha
            self.run_mean *= self.alpha

            self.run_var += np.var(x, axis=0) * (1 - self.alpha) 
            self.run_mean +=  np.mean(x, axis=0) * (1 - self.alpha)
        
        return self.gamma * self.norm + self.beta

    def batch_norm_backward(self, dy):
        """
        Batch norm derivative backward pass to linear layer
        """
        dbeta = np.sum(dy, axis = 0, keepdims=True)
        dgamma = np.sum(self.norm * dy, axis=0, keepdims=True)

        return (self.gamma * (dy.shape[0] * dy - self.norm * dgamma - dbeta)) / (self.batch_sd * dy.shape[0])

    def dropout_forward(self, x, predict):
        """
        Disable dropout during test / validation, only used during training 
        """
        if predict:  return x
        
        self.mask = np.random.binomial(1, (1 - self.dropout), size = x.shape)
        return x * self.mask / (1 - self.dropout)

    def dropout_backward(self, dy):
        """
        Dropout correction for backward pass
        """
        return dy * self.mask / (1 - self.dropout)

    def forward(self, x, predict=False):
        """
        Apply wx + b linear transformations
        """
        self.x = x
        out = x @ self.W + self.b

        if self.batch_norm:
            out = self.batch_norm_forward(out, predict)

        if self.dropout:
            out = self.dropout_forward(out, predict)

        return out

    def backward(self, dy, reg_term=0):
        """
        Calculate gradient wrt dy for update step
        """
        if self.batch_norm:
            dy = self.batch_norm_backward(dy)

        self.dW = self.x.T @ dy
        self.db = np.sum(dy, axis=0, keepdims=True)

        if self.dropout:
            dy = self.dropout_backward(dy)
        
        # L2 Regularization not equivalent to weight decay (however is understood to be for SGD) 
        if reg_term:
            self.dW = self.x.T @ dy - reg_term * self.W / self.x.shape[0]

        return dy @ self.W.T

    def reset_gradients(self):
        """
        Zero out gradients after each sample
        """
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.diffW = np.zeros_like(self.W)
        self.diffb = np.zeros_like(self.b)


if __name__ == "__main__":

    # Linear Tests

    x = np.random.uniform(size=(10, 128))
    linear = Linear(128, 32)
    y = linear.forward(x)
    
    assert (y.shape == np.array([10, 32])).all()
    
    dy = np.random.uniform(size=(10,32))
    dx = linear.backward(dy)

    assert dx.shape == x.shape
    assert linear.dW.shape == linear.W.shape
    assert linear.db.shape == linear.b.shape

    assert isinstance(linear, Layer)