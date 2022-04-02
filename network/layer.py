import numpy as np
from pyparsing import line

def xavier(size, gain):
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

    def __init__(self, indim, outdim):
        """
        Layer: Linear y = xw + b

        Args
            indim: incoming n features
            outdim: outgoing m features
        """
        self.indim, self.outdim = indim, outdim
        
        # initialize weights and bias (currently using Xavier)

        self.W = xavier((indim, outdim), gain=1)
        self.b = xavier((1, outdim), gain=1)

        self.dW = np.zeros((indim, outdim))
        self.db = np.zeros((1, outdim))

    def forward(self, x):
        """
        Apply wx + b linear transformations
        """
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        """
        Calculate gradient wrt dy for update step
        """
        #self.dcost = dy
        #self.dz = self.h

        #self.dcostw = np.dot((self.dz.T, self.dcost))
        #self.dcostb = self.dcost

        self.dW = self.x.T @ dy
        self.db = np.sum(dy, axis=0, keepdims=True)
        return dy @ self.W.T

    def update(self, lr):
        """
        ..using optimizer method instead 
        Update weight and bias wrt gradient from loss
        """
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def reset_gradients(self):
        """
        Zero out gradients after each sample
        """
        self.dW = np.zeros_like(self.dW)
        self.db = np.zeros_like(self.db)

class Activation:
    """
    Abstract class for activations 
    """
    def __init__(self):
        pass
    def forward(self, x):
        return x
    def backward(self, dy):
        return dy

class ReLU(Activation):
    def __init__(self):
        """
        Rectified Linear Unit max(x, 0)
        """
        self.x = None

    def forward(self, x):
        """
        Returns x if x > 0
        """
        self.x = x
        return x * (x > 0)

    def backward(self, dy):
        """
        If x <= 0 ret 0 else 1
        """
        return dy * (self.x > 0)

class LeakyReLU(Activation):
    def __init__(self):
        """
        Leaky ReLU
        """
        self.leak = 0.03

    def forward(self, x):
        """
        If > 0 return x else return leaky amount * x
        """
        self.x = x
        return x * (x > 0) + (x <= 0) * self.leak * x

    def backward(self, dy):
        """
        If > 0 ret 1 else return leaky amount
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


    # LeakyReLU Tests

    x = np.array([[3,2], [1,-4]])
    lr = LeakyReLU()
    y = lr.forward(x)

    assert np.allclose(y, np.array([[3,2], [1,-0.12]]))

    x = np.array([[1,1], [1,1]])
    y = lr.backward(x)

    assert np.allclose(y, np.array([[1,1], [1,0.03]]))

    assert isinstance(linear, Layer)
    assert isinstance(lr, Activation)