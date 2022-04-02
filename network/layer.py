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

    def __init__(self, indim, outdim):
        """
        Layer: Linear y = xw + b

        Args
            indim: incoming n features
            outdim: outgoing m features
        """
        self.indim, self.outdim = indim, outdim
        
        # initialize weights and bias (currently using Xavier)

        self.W = xavier((indim, outdim))
        self.b = xavier((1, outdim))
        #xavier((1, outdim))

        self.dW = np.zeros((indim, outdim))
        self.db = np.zeros(outdim,)

        # momentum terms

        self.diffW = np.zeros((indim, outdim))
        self.diffb = np.zeros(outdim,)

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