import numpy as np

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

    def __repr__(self):
        return "ReLU"

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

    def __repr__(self):
        return "LeakyReLU"

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

    # ReLU Tests

    x = np.array([[3,2], [1,-4]])
    reLU = ReLU()
    y = reLU.forward(x)

    assert (y == np.array([[3,2], [1,0]])).all()

    x = np.array([[1,1], [1,1]])
    y = reLU.backward(x)

    assert (y == np.array([[1,1], [1,0]])).all()

    assert isinstance(reLU, Activation)


    # LeakyReLU Tests

    x = np.array([[3,2], [1,-4]])
    lr = LeakyReLU()
    y = lr.forward(x)

    assert np.allclose(y, np.array([[3,2], [1,-0.12]]))

    x = np.array([[1,1], [1,1]])
    y = lr.backward(x)

    assert np.allclose(y, np.array([[1,1], [1,0.03]]))

    assert isinstance(lr, Activation)