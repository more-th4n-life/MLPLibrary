import numpy as np
from network.layer import Linear

class Optimizer:
    """
    Parent Optimizer class. Attributes: epochs, epoch, iters and time_step are all set upon in the net function,
    if using Adam, time_step is used. If SGD, these values are used to calculate using a selected decay method
    upon initializaton, the learning rate for the next iteration in training.
    """
    def __init__(self):
        self.epochs = None
        self.epoch = None
        self.iters = None 
        self.time_step = None
        
class SGD(Optimizer):
    """
    Class Stochastic Gradient Descent.

    Args:
        learning_rate (float): learning rate to control step size in traversing the loss function canyon :)
        weight_decay (float): a term that is used to penalize weight values as a factor of learning rate
        momentum (float): extra term to accelerate derivates towards minima points (speeds up convergence)
        lr_decay (string): learning rate schedular, can choose from "default", "step", "exp" & "time" decay
        step_terms (tuple): terms step i.e. num of epochs, before each drop (a factor it learning rate is reduced by)
    
    """
    def __init__(self, learning_rate=0.04, weight_decay = 0, momentum = 0.5, lr_decay = "default", step_terms = (10, 0.5)):
        super(Optimizer, self).__init__()

        self.lr, self.wd, self.momentum = learning_rate, weight_decay, momentum

        # Learning rate schedulers
        sched = {
            'time': self.time_decay,      
            'step': self.step_decay,     
            'exp': self.exp_decay,
            'default': self.get_lr
        }
        # randomly init weights based on chosen init method
        self.lr_sceduler = sched[lr_decay]
        self.step_terms = step_terms  # only used in step decay

    def __repr__(self):
        """
        Repr method for SGD class, used in Net class when generating model names.

        Returns:
            str: returns the string representation of the class
        """
        if self.wd and self.momentum:
            return "SGD(wd+mm)"
        elif self.wd:
            return "SGD(wd)"
        elif self.momentum:
            return "SGD(mm)"
        return "SGD"

    def get_lr(self):
        """
        Getter method for learning rate

        Returns:
            float: learning rate
        """
        return self.lr

    def time_decay(self):
        """
        Return decreased lr by time_step factor (epoch * iter)

        Returns:
            float: next lr value
        """
        k = self.lr / self.epochs
        return self.lr * (1 / (1 + k * self.time_step))

    def step_decay(self):
        """
        Return decreased lr by factor of "drop" (default: half) every "step" (default 10) epochs

        Returns:
            float: next lr value
        """
        step = self.step_terms[0]
        drop = self.step_terms[1]

        return self.lr * (drop ** ((1 + self.epoch) // step))

    def exp_decay(self):
        """
        Return decayed lr by an exponential factor of time and chosen constant k (exp term)
        I have chosen k to factor in total training iters st lr magnitude pans over all planned epochs

        Returns:
            float: next lr values
        """
        exp_term = 1 / self.iters
        return self.lr * np.exp(-exp_term * self.time_step)
        
    def step(self, network):
        """
        Update weights and biases in each layer wrt learned dW and db AND learning rate (+ weight decay) term
        """
        lr = self.lr_sceduler()

        for layer in [l for l in network.layers if isinstance(l, Linear)]:
    
            layer.diffW *= self.momentum
            layer.diffb *= self.momentum

            # if weight decay, large W is penalized more than small W
            layer.diffW = lr * (-layer.dW - self.wd * layer.W)
            layer.diffb = lr * np.mean(layer.db, axis=0)

            layer.W += layer.diffW
            layer.b += layer.diffb


class Adam(Optimizer):
    """
    Class Adam optimizer

    Args:
        learning_rate (float): default 0.001
        beta1 (float): default 0.9
        beta2 (float): default 0.999
        epsilon (float): default 1e-09, used for stability
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-09):
        super(Optimizer, self).__init__()

        self.lr, self.beta1, self.beta2 = learning_rate, beta1, beta2
        
        self.v, self.v_corr = {}, {}
        self.s, self.s_corr = {}, {}

        self.epsilon = epsilon
        self.time_step = None

    def __repr__(self):
        """
        Repr method for SGD class, used in Net class when generating model names.

        Returns:
            str: returns the string representation of the class
        """
        return "Adam"

    def add_to_dict(self, indim, outdim, i):
        """
        Helper class to initialize each codebook 
        """

        self.v["dW" + str(i)] = np.zeros((indim, outdim))
        self.v["db" + str(i)] = np.zeros(outdim,)

        self.s["dW" + str(i)] = np.zeros((indim, outdim))
        self.s["db" + str(i)] = np.zeros(outdim,)
    
    def step(self, network):
        """
        Update step in Adam: 
        (1) Calculates moving average of gradients
        (2) Corrects bias with first moment estimates
        (3) Calculates moving average of squared gradients
        (4) Corrects bias with second moment estimates
        
        Lastly, updates all parameters
        """

        for k, layer in enumerate([L for L in network.layers if isinstance(L, Linear)]):
            
            i = k + 1

            # calculate moving avg of gradients 
            self.v["dW" + str(i)] = self.beta1 * self.v["dW" + str(i)] + (1 - self.beta1) * layer.dW
            self.v["db" + str(i)] = self.beta1 * self.v["db" + str(i)] + (1 - self.beta1) * layer.db

            # correct bias with first moment estimates
            self.v_corr["dW" + str(i)] = self.v["dW" + str(i)] / (1 - self.beta1 ** self.time_step)
            self.v_corr["db" + str(i)] = self.v["db" + str(i)] / (1 - self.beta1 ** self.time_step)

            # calculate moving avg of squared gradients
            self.s["dW" + str(i)] = self.beta2 * self.s["dW" + str(i)] + (1 - self.beta2) * (np.square(layer.dW))
            self.s["db" + str(i)] = self.beta2 * self.s["db" + str(i)] + (1 - self.beta2) * (np.square(layer.db))

            # correct bias with second moment estimates
            self.s_corr["dW" + str(i)] = self.s["dW" + str(i)] / (1 - self.beta2 ** self.time_step)
            self.s_corr["db" + str(i)] = self.s["db" + str(i)] / (1 - self.beta2 ** self.time_step)

            # update params
            layer.W += -self.lr * self.v_corr["dW" + str(i)] / (np.sqrt(self.s_corr["dW" + str(i)]) + self.epsilon)
            layer.b += -self.lr * self.v_corr["db" + str(i)] / (np.sqrt(self.s_corr["db" + str(i)]) + self.epsilon)

