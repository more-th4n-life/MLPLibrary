import numpy as np

class Loss():
    """
    Abstract class for Loss 
    """
    def __init__(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass

class CrossEntropyLoss(Loss):
    """
    Calc softmax then negative log-likelihood loss
    """
    def __init__(self):
        self.cache = None
        self.epsilon = 1e-09

    def softmax(self, x, n):
        """
        Returns probabilities for each class
        """
        e_out = np.exp(x - np.amax(x, axis=1).reshape(n, 1))
        return (e_out) / np.sum(e_out, axis=1, keepdims=True)

    def forward(self, x, label):
        """
        Uses softmax to calculate Cross entropy loss
        """
        prob = self.softmax(x, x.shape[0])
        self.cache = label, prob
        return -np.mean(np.log(prob + self.epsilon) * label)  # prevent log(0)

    def backward(self):
        """
        Returns difference between softmax probs and ground truth for update
        """
        label, prob = self.cache
        return (prob - label) / prob.shape[0]

    def __call__(self, input, target):
        return self.forward(input, target)

if __name__ == "__main__":
    from loader.process import one_hot

    ce = CrossEntropyLoss()
    x = np.array([[-0.8,  0.5,  0.4, -1.1,  -1.6,  0.2, 1.2, 3.1, 2.0, -0.1]])

    label = one_hot(7, 10)
    print(x)
    print(label)

    s = ce.softmax(x, x.shape[0])
    print(s)

    print(ce.forward(x, label))
    print(ce.backward())