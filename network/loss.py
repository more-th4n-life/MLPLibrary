import numpy as np

class Loss:
    """
    Parent class for Loss Criterion

    This class is inherited by currently only supported child class: CrossEntropyLoss
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
    Class for Cross-Entropy Loss. By default, this class applies Softmax activation on input mini-batch x
    to calculate probabilities for each class. These quantized probabilities for each class are then used 
    in the calculation of Cross-Entropy Loss for backpropagation (training) of the network, or directly in
    making a prediction by taking the class with the maximum Softmax calculated probability.
    """
    def __init__(self):
        self.prob, self.label, self.logit = None, None, None
        self.epsilon = 1e-09

    def __repr__(self):
        """
        Repr method for CrossEntropyLoss class, used in Net class when generating model names.
        """
        return "CELoss"

    def softmax(self, x):
        """
        Used for calculating the Softmax probabilities of each class. Acts as the default last activation in
        the layer prior to calculation of Cross Entropy Loss.
        
        Args:
            x (np.ndarray): mini-batch x of values passed through in the forward pass of training / prediction

        Returns:
            np.ndarray: returns the probabilities for each class, the class with the highest probability is our predicted label
        """
        clas = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        return clas / np.sum(clas, axis=1, keepdims=True)

    def forward(self, x, label):
        """
        The forward step in Cross-Entropy Loss is the calculation of the Cross-Entropy Loss value, 
        determined by calculating Softmax probabilities for each class and taking the log-likelihood of
        probabilities which determines the negative average of log corrected predicted probabilities for
        the mini-batch x (ie. against true labels). The probabilities calculated with Softmax are cached
        along with ground-truth labels, to then be used for calculating the derivative of Cross-Entropy 
        for the backward pass for updating weights/biases in each layer. Softmax probabilities are also
        returned for use in evaluating the accuracy of the model, or making predictions on unlabelled data.
        Epsilon is used to stabilize calculation of log Softmax where unwanted overflow/underflow side-effects
        could occur due to floating-point representation of small probability values. 
        
        Args:
            x (np.ndarray): mini-batch x of values passed through in the forward pass of training / prediction

        Returns:
            float: returns the calculated cross-entropy loss over mini-batch x;
            np.ndarray: also returns Softmax probabilities for each class for prediction / accuracy evaluation
        """
        self.prob, self.label = self.softmax(x), label
        self.logit = -np.log(self.prob + self.epsilon)
        
        return np.sum(self.logit * label) / self.prob.shape[0], self.prob  # prevent log(0)
    
    def backward(self):
        """
        The backward pass of Cross-Entropy Loss calculates the derivative of Cross-Entropy Loss (loss function)
        using Softmax activation. Both Softmax probabilities and ground-truth labels are cached for the current 
        mini-batch in the forward pass and used to calculate the derivative in the backward pass. This is simply 
        the difference of probabilities with one-hot encoded ground-truth labels averaged over all the samples.

        Returns:
            np.ndarray: derivative of Cross-Entropy Loss with Softmax activation averaged number of samples in mini-batch x.
        """
        return (self.prob - self.label) / self.prob.shape[0]
       

    def __call__(self, x, label=None):
        """
        Since softmax is treated by default as the last activation layer for the network when calculating the CrossEntropyLoss
        criterion, calling CrossEntropyLoss when making predictions should only return the Softmax probailities calculated for 
        each class. During prediction, it is implied that we do not know the ground-truth labels and so should account for this.
        If we do have ground-truth labels, we can calculate Cross-Entropy Loss wrt labels and return both loss and probabilities.
    
        """
        return self.forward(x, label) if isinstance(label, np.ndarray) else self.softmax(x)


if __name__ == "__main__":
    from network.loader.process import one_hot

    ce = CrossEntropyLoss()
    x = np.array([[-0.8,  0.5,  0.4, -1.1,  -1.6,  0.2, 1.2, 3.1, 2.0, -0.1]])
    label = one_hot(7, 10)
    s = ce.softmax(x)
    print(ce.forward(x, label)[0])
    x = ce.forward(x, label)[0]
    print(x.shape[1])
    y = ce.backward()
    print(y)
