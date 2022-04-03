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
        self.prob, self.label, self.logit = None, None, None
        self.epsilon = 1e-09

    def __repr__(self):
        return "CELoss"

    def softmax(self, x):
        """
        Returns probabilities for each class
        """
        clas = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        return clas / np.sum(clas, axis=1, keepdims=True)

    def forward(self, x, label):
        """
        Uses softmax to calculate Cross entropy loss
        """
        self.prob, self.label = self.softmax(x), label
        self.logit = -np.log(self.prob + self.epsilon)
        
        return np.sum(self.logit * label) / self.prob.shape[0], self.prob  # prevent log(0)
    
    def backward(self):
        """
        Returns difference between softmax probs and ground truth for update
        """
        return (self.prob - self.label) / self.prob.shape[0]
       

    def __call__(self, x, label):
        return self.forward(x, label)


if __name__ == "__main__":
    from loader.process import one_hot

    ce = CrossEntropyLoss()
    x = np.array([[-0.8,  0.5,  0.4, -1.1,  -1.6,  0.2, 1.2, 3.1, 2.0, -0.1]])

    label = one_hot(7, 10)
    #print(x)
    #print(label)

    s = ce.softmax(x)
    #print(s)

    print(ce.forward(x, label)[0])
    #print(ce.backward())

    x = ce.forward(x, label)[0]
    #print(np.sum(x) / x.shape[1])

    print(x.shape[1])

    #print(np.mean(x))

    y = ce.backward()
    print(y)
