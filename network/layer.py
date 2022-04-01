import numpy as np
from pyparsing import line

def xavier(size, gain):
    i = -gain * np.sqrt(6. / np.sum(size))
    return np.random.uniform(low=i, high=-i, size=size)

class Layer():
    def __init__(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass
    def update(self):
        pass

class Linear(Layer):

    def __init__(self, indim, outdim):
        """
        Layer: Linear y = xw + b

        Args
            indim: incoming n features
            outdim: outgoing m features
        """
        self.indim, self.outdim = indim, outdim
        
        # initialize weights and bias
        self.W = xavier(gain=1, size=(indim, outdim)) 
        self.b = xavier(gain=1, size=(1, outdim)) 

        self.dW = np.zeros((indim, outdim))
        self.db = np.zeros((1, outdim))

        #self.diffW = np.zeros((indim, outdim))
        #self.diffb = np.zeros((1, outdim))

    def forward(self, x):
        """
        Apply xw + b linear transform
        """
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dy):
        """
        Calculate gradient wrt dy
        """
        self.dW = np.dot(self.x.T, dy)
        self.db = np.sum(dy, axis=0, keepdims=True)
        return np.dot(dy, self.W.T)

    def update(self, lr):
        """
        ..using optimizer method instead 
        """
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def reset_gradients(self):
        self.dW = np.zeros_like(self.dW)
        self.db = np.zeros_like(self.db)

class Activation():
    def __init__(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass

class ReLU(Activation):
    def __init__(self):
        """
        Rectified Linear Unit max(x, 0)
        """
        self.x = None

    def forward(self, x):
        self.x = x
        return x * (x > 0)

    def backward(self, dy):
        """
        If < 0 ret 0
        """
        return dy * (self.x > 0)

class LeakyReLU(Activation):
    def __init__(self):
        self.leak = 0.03

    def forward(self, x):
        self.x = x
        return x * (x > 0) + (x <= 0) * x * self.leak

    def backward(self, dy):
        """
        If < 0 ret leak
        """
        return dy * (self.x > 0) + (self.x <= 0) * dy * self.leak

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

    # ReLU Tests

    x = np.array([[3,2], [1,-4]])
    reLU = ReLU()
    y = reLU.forward(x)

    assert (y == np.array([[3,2], [1,0]])).all()

    x = np.array([[1,1], [1,1]])
    y = reLU.backward(x)

    assert (y == np.array([[1,1], [1,0]])).all()

    assert isinstance(linear, Layer)
    assert isinstance(reLU, Activation)