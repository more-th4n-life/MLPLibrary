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
