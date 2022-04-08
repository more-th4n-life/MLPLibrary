import numpy as np

class Layer:
    """
    Parent class for Hidden Layers

    This class is inherited by currently only supported child class: Linear
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
    """
    Class for Linear Layer

    Args:
        indim (int): input dimensions of mini-batch data or number of dimensions of input feature map 
        outdim (int): output dimensions or number of hidden neurons within linear layer to process input
        dropout (float): value between 0 and 1 to indicate ratio of neurons to randomly deactivate in layer
        weights (str): weights init method with typical selection from "xavier" and "kaiming" (defaults to "xavier")'
        bias (str): bias init method with selection from above and either "zero" or "const" (defaults to "zero")
    """

    def __init__(self, indim, outdim, dropout=0, weights="xavier", bias="zero"):
        self.indim, self.outdim = indim, outdim
        self.batch_norm, self.alpha, self.L2_reg_term = False, None, None    # set within network as parameters
        self.dropout = dropout
        self.epsilon = 1e-10
        
        const = lambda size: np.full(size, 0.01)

        # Initialization methods (weights init randomly and bias either small or zero)
        init = {
            # Uniform Initialization for Weights
            'xavier': self.xavier,      # more suitable for weights with tanh activation
            'kaiming': self.kaiming,     # more suitable for weight with relu activation

            # Constant Initialization for Bias (user also select Xavier or Kaiming)
            'const': const,            # sometimes 0.01 preferred over 0 with relu activation
        }

        # randomly init weights based on chosen init method
        self.W = init[weights]((indim, outdim))
        self.b = np.zeros((1, outdim)) if bias=="zero" else init[bias]((1, outdim))
        
        # gradient of W and b terms
        self.dW = np.zeros((indim, outdim))
        self.db = np.zeros(outdim,)

        # momentum terms
        self.diffW = np.zeros((indim, outdim))
        self.diffb = np.zeros(outdim,)

        # batch norm
        self.gamma = np.ones(int(np.prod(outdim)),)
        self.beta = np.zeros(int(np.prod(outdim)),)
        self.run_var = np.ones(outdim,)
        self.run_mean = np.zeros(outdim,)

        # predict toggled for validation / testing
        self.predict = False

    def __repr__(self):
        """
        Repr method for Linear class, used in Net class when generating model names.
        """
        return f"[{self.outdim}]"

    def xavier(self, size, gain=1):
        """
        Function for Xavier Uniform Initialization, primarily used for randomly initializing
        weight values in a layer, however, can be used for biases as well.

        Args:
            size (tuple): dimensions of generated np.ndarray i.e. (#row, #col)
            gain (int): scalar multiplication of generated weight/bias values 

        Returns:
            np.ndarray: returns weights/bias of given size (and gain) using Xavier initialization
        """
        i = gain * np.sqrt(6. / np.sum(size))
        return np.random.uniform(low = -i, high = i, size = size)

    def kaiming(self, size, gain=1):
        """
        Function for Kaiming-He Uniform Initialization, primarily used for randomly initializing
        weight values in a layer, however, can also be used for biases as well.

        Args:
            size (tuple): dimensions of generated np.ndarray i.e. (#row, #col)
            gain (int): scalar multiplication of generated weight/bias values 

        Returns:
            np.ndarray: returns weights/bias of given size (and gain) using Kaiming-He initialization
        """
        i = gain * np.sqrt(6. / size[0])
        return np.random.uniform(low = -i, high = i, size = size)

    def batch_norm_forward(self, x):
        """
        Helper function for a forward pass using batch normalization on mini-batch x.
        If making a prediction (i.e. not undergoing backpropagation and only making a forward pass),
        then the running means and variance calculated in this layer from training are then used 
        instead to normalise the output of this batch for the next layer. The predict function is
        flagged True in the Net class upon calling the Net.predict() method on the network, this
        sets all layer.predict attributes to True to determine to use previously calculated means
        and variance for normalizing x, or if False (during training) to calculate this mean and variance.

        Gamma and beta are initialized during initialization of Linear layer to scale amd shift our
        normalized value of x, alpha is a scalar term set upon Net initialization to scale the running mean and variance.

        Args:
            x (np.ndarray): mini-batch x during forward pass of training or prediction.

        Returns:
            np.ndarray: returns normalized mini-batch x, that is scaled with gamma and shifted with beta
        
        """
        if self.predict: self.norm = (x - self.run_mean) / np.sqrt(self.run_var + self.epsilon)

        else:
            self.batch_sd = np.sqrt(np.var(x, axis=0) + self.epsilon)
            self.norm = (x - np.mean(x, axis=0)) / self.batch_sd

            self.run_var *= self.alpha
            self.run_mean *= self.alpha

            self.run_var += np.var(x, axis=0) * (1 - self.alpha) 
            self.run_mean +=  np.mean(x, axis=0) * (1 - self.alpha)
        
        return self.gamma * self.norm + self.beta

    def batch_norm_backward(self, dy):
        """
        Helper function for the backward pass of batch normalization using the derivative of the loss
        function. Partial derivatives for shift values of beta are calculated as the sum of all values over the 
        input derivative of the loss function and derivatives for for gamma using the chain rule as the sum of 
        all normalized values pre-cached values in forward pass multiplied against the derivative of loss. 
        This is then used to calculate the derivate for the mini-batch x w.r.t. the loss function and returns
        it's batch normalized value.

        Args:
            dy (np.ndarray): derivative of the loss function passed in via back-propagation.

        Returns:
            np.ndarray: returns batch-normalized value for the derivative of the loss function in this layer
        
        """
        dbeta = np.sum(dy, axis = 0, keepdims=True)
        dgamma = np.sum(self.norm * dy, axis=0, keepdims=True)

        return (self.gamma * (dy.shape[0] * dy - self.norm * dgamma - dbeta)) / (self.batch_sd * dy.shape[0])

    def dropout_forward(self, x):
        """
        Helper function for applying dropout to the forward pass in either training or prediction. If making a
        prediction, dropout is deactivated. During training, dropout randomly chooses neurons to drop in the 
        linear transformation of x in the forward pass w.r.t. each neuron's weight and bias. Dropout parameter
        is initialized upon Layer initialization, and used to drop a percentage of the neuron's in this Layer.

        Args:
            x (np.ndarray): mini-batch x during forward pass of training or prediction.

        Returns:
            np.ndarray: returns new x values whereby proportion (dropout) is randomly masked as zeros, turned off.
        """
        if self.predict: return x
        
        self.mask = np.random.binomial(1, (1 - self.dropout), size = x.shape)
        return x * self.mask / (1 - self.dropout)
        

    def dropout_backward(self, dy):
        """
        Helper function for applying dropout to the backward pass on the derivative for the loss function. Using
        the cached mask from forward pass in this layer, applies this mask similarly to the derivative of loss.

        Args:
            dy (np.ndarray): derivative of the loss function passed in via back-propagation.

        Returns:
            np.ndarray: returns dx values whereby proportion (dropout) is randomly masked as zeros, turned off.
        """
        return dy * self.mask / (1 - self.dropout)


    def forward(self, x):
        """
        Main method for Layer in making a forward pass in training / prediction. Using linear transformation
        on mini-batch x to calculate an output w.r.t. learned weights and biases from backpropagation. This 
        transformation: y = wx + b; takes mini-batch x as input, scales it by learned weight values, and shifts
        it by learned bias values to calculate output y to be used in the next layer. Ordering of applying both
        batch-norm and dropout is calculated in the order stated, when both are set for use within the Layer.

        Args:
            x (np.ndarray): mini-batch x during forward pass of training or prediction.

        Returns:
            np.ndarray: returns linearly transformed x values as input to next layer (i.e. for activation)
        """
        self.x = x
        out = x @ self.W + self.b

        # batch_norm before dropout for better performance https://arxiv.org/abs/1801.05134 
        if self.batch_norm:
            out = self.batch_norm_forward(out)

        if self.dropout:
            out = self.dropout_forward(out)

        return out

    def backward(self, dy):
        """
        Main method for Layer in making a backward pass for backpropagation. Order of applying the gradients
        of dropout and batch norm are applied in the order stated and is a mirror application of the forward pass.
        The derivative of the weight values are calculated as the dot product of loss derivative with cached input
        x (that is matrix rotated / transposed) and for bias, the sum of the gradient values of the loss function.
        Lastly, derivative of the mini-batch x is returned as the dot product between rotated gradient of weights
        multiplied against the the derivative of the loss function values. If L2 regularization is applied (with 
        value determined upon the initialization of Net), then the derivative of weights reduced by a factor of this
        term, the value of weights and the size of mini-batch x. This is understood to penalize larger weight values.
        
        Args:
            dy (np.ndarray): derivative of the loss function passed in via back-propagation.

        Returns:
            np.ndarray: derivative of the loss function w.r.t. the weights of this layer.
        """
        if self.dropout:
            dy = self.dropout_backward(dy)

        if self.batch_norm:
            dy = self.batch_norm_backward(dy)
        
        dW = self.x.T @ dy
        self.db = np.sum(dy, axis=0, keepdims=True)

        # L2 Regularization:
        self.dW = dW - self.L2_reg_term * self.W / self.x.shape[0] if self.L2_reg_term else dW
        # not equivalent to weight decay (however is understood to be for SGD) 

        return dy @ self.W.T

    def reset_gradients(self):
        """
        Method to reset the gradient of the weight and bias values for each Layer, called upon
        after each iteration of batch training.
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