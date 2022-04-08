import numpy as np

class Activation:
    """
    Parent class for Non-Linear Activation Functions

    This class is inherited by currently supported child classes: ReLU and LeakyReLU.
    """
    def __init__(self):
        pass
    def forward(self, x):
        return x
    def backward(self, dy):
        return dy

class ReLU(Activation):
    """
    Class for Activation function: Rectified Linear Unit (ReLU)

    Output is then scaled by multiplication with dy in the backward pass.
    """
    def __init__(self):
        self.x = None

    def __repr__(self):
        """
        Repr method for ReLU class, used in Net class when generating model names.

        Returns:
            str: returns the string representation of the class
        """
        return "ReLU"

    def forward(self, x):
        """
        Calculates ReLU-fied output of passed in mini-batch data (x) in forward pass.
        This function is activated during both training and prediction. This is then
        fed into the next layer (most likely a Linear layer).

        Args:
            x (np.ndarray): mini-batch data passed to network in forward pass

        Returns:
            np.ndarray: returns ReLU-fied output of the mini-batch data
        """
        self.x = x
        return x * (x > 0)

    def backward(self, dy):
        """
        Passes the gradient of the loss (cost / error) function w.r.t weights in 
        previous layers and applies the derivative of the ReLU function in processing
        input for the backward pass of backpropagation to update weights and biases.

        Args:
            dy (np.ndarray): gradient of the loss function in backward pass

        Returns:
            np.ndarray: returns derivative of ReLU applied to dy and cached x in forward pass.
        """
        return dy * (self.x > 0)

class LeakyReLU(Activation):
    """
    Class for Activation function: Leaky Rectified Linear Unit (LeakyReLU)

    Output is then scaled by multiplication with dy in the backward pass.

    Args:
        leak (float): the leaky amount (slope) in Leaky ReLU function (usually very small). \
            Defaults to 0.03 (arbitrarily chosen amount).
    """
    def __init__(self, leak = 0.03):
        self.leak = leak

    def __repr__(self):
        """
        Repr method for ReLU class, used in Net class when generating model names.
        """
        return "LeakyReLU"

    def forward(self, x):
        """
        Calculate Leaky ReLU-fied output of passed in mini-batch data (x) in forward pass.
        This function is activated during both training and prediction. This is then
        fed into the next layer (most likely a Linear layer).

        Args:
            x (np.ndarray): mini-batch data passed to network in forward pass

        Returns:
            np.ndarray: returns Leaky ReLU-fied output of the mini-batch data
        """
        self.x = x
        return x * (x > 0) + (x <= 0) * self.leak * x

    def backward(self, dy):
        """
        Passes the gradient of the loss (cost / error) function w.r.t weights in 
        previous layers and applies the derivative of the ReLU function in processing
        input for the backward pass of backpropagation to update weights and biases.

        Args:
            dy (np.ndarray): gradient of the loss function in backward pass

        Returns:
            np.ndarray: returns derivative of ReLU applied to dy and cached x in forward pass.
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