import numpy as np

class Loss():
    def __init__(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass

class CrossEntropyLoss(Loss):
    """
    calc softmax then negative log-likelihood loss
    """
    def __init__(self):
        self.cache = None
        self.epsilon = 1e-09

    def softmax(self, x, n):
        e_out = np.exp(x - np.amax(x, axis=1).reshape(n, 1))
        return (e_out) / np.sum(e_out, axis=1, keepdims=True)

    def forward(self, x, label):
        prob = self.softmax(x, x.shape[0])
        self.cache = label, prob
        return -np.sum(np.multiply(label, np.log(prob + self.epsilon))) / x.shape[0] # prevent log(0)

    def backward(self):
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

    #print(ce.forward(x, label))
    #print(ce.backward())