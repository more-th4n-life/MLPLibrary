from network.loss import CrossEntropyLoss
from network.layer import Linear, Layer
from network.activ import ReLU
from network.optim import SGD, Adam

import numpy as np
import os, pickle
from tqdm import tqdm 
from functools import partial 
from time import time

tqdm = partial(tqdm, position = 0, leave = True)
show_time = lambda t: f"{int(t//60)} min {np.round(np.mod(t,60), 1)} s" if t >= 60 else f"{np.round(t,1)} sec(s)"

class Net:
    """
    Class for Neural Networks (Multi-Layer Perceptron models).
    Currently supports: Hidden Layers (Linear) and Activation functions (ReLU, LeakyReLU); Loss Criterion: 
    CrossEntropyLoss (implicitly with Softmax activation); Optimizers: SGD (with or without Momentum and/or 
    Weight decay) and Adam; Regularization: L2 Regularization over network, Dropout (specific to each layer) 
    and Batch Normalization (toggled for whole network for ease of convenience).

    Args:
        optimizer (Optimizer): Optimizer to be applied for updating weights and biases in each layer during training. 
        criterion (Loss): Loss criterion used for the evaluation and optimization of minimizing the loss function.
        batch_norm (bool): Flag to toggle application of batch normalizations of Linear layers in the network.
        alpha (float): Alpha term used in the calculation of running mean and variance in batch normalization.
        L2_reg_term (str): L2 regularization term used for penalizing larger weights in the network

    """
    def __init__(self, optimizer=SGD(), criterion=CrossEntropyLoss(), batch_norm=False, alpha=0.9, L2_reg_term=0):
        self.layers, self.size, self.linear = [], 0, 0
        self.optimizer, self.criterion = optimizer, criterion
        self.batch_norm, self.alpha = batch_norm, alpha  # alpha only used if batch norm is set
        self.L2_reg_term = L2_reg_term  # dropout percentage and L2 regularization term
        self.model_name = None  # used to identify model for saving / loading

    def __repr__(self):
        """
        Repr method for Net class, generates names from repr of selected hyperparameters within Net

        Returns:
            str: returns the string representation of the class (the entire model name)
        """
        if self.model_name:
            return self.model_name

        model = repr(self.optimizer) + repr(self.criterion)
        model += f"{'L2' if self.L2_reg_term else ''}"
        model += f"{'BN' if self.batch_norm is True else ''}"

        for layer in self.layers:
            model += repr(layer) 

        return model

    def set_name(self, name):
        """
        Setter method for model name, useful for saving / loading models for re-use in making prediction or for continuing training

        Args:
            name (str): name for the model
        """
        self.model_name = name

    def add(self, layer):
        """
        Net method to add a layer to the model, layers can be selected as either a Linear layer or as Activation function between
        layers (e.g. ReLU or LeakyReLU)

        Args:
            layer (Layer or Activation): Hidden layer to be added (either Linear or Activation function between Linear layers)
        """
        if isinstance(layer, Linear): 
            if self.batch_norm:
                layer.batch_norm, layer.alpha = True, self.alpha
            if self.L2_reg_term:
                layer.L2_reg_term = self.L2_reg_term
            if isinstance(self.optimizer, Adam):
                self.linear += 1
                self.optimizer.add_to_dict(layer.indim, layer.outdim, self.linear)
            
        self.layers += [layer]
        self.size += 1

    def forward(self, x, predict=False):
        """
        Main method for Net in making a forward pass in training / prediction. Iterates from input layer to
        output layer each class' forward function and passes forward predictions to use in calculating loss. 
        Predict is set to false by default to indicate training is occuring - this modifies the control-flow
        within the Linear class, which specifically modifies the behaviour of batch normalization and dropout.

        Args:
            x (np.ndarray): mini-batch x during forward pass of training or prediction.
            predict (bool): True if performing a forward pass in prediction, or False if model is in training.
                By default, set to false.

        Returns:
            np.ndarray: returns output values of Net used for calculating Loss using the selected Criterion function
        """
        for layer in self.layers:
            if isinstance(layer, Linear): 
                layer.predict = predict     # toggled during validation / testing    
            x = layer.forward(x)
        return x

    def backward(self, dy):
        """
        Main method for Net in making a backward pass in backpropagation. Passes the gradient of the loss
        function from the output layer to the input layer, which updates the weights and biases within the
        Linear layers with the objective of minimizing the calculation of the Loss function in subsequent
        iterations (i.e. improving model performance).

        Args:
            dy (np.ndarray): passes the gradient of the loss function generated by the Loss criterion.
        """

        for layer in self.layers[::-1]:
            dy = layer.backward(dy)

    def update(self):
        """
        Update step that uses the selected optimizer to update the weights and biases of each Linear layer, using
        the differentials of the weights and biases within each layer (and parameters unique to the Optimizer algorithm)
        """
        self = self.optimizer.step(self) 

    def reset_gradients(self):
        """
        Helper method to zero out gradients of weights and biases in each layer, called before each iteration of training.
        """
        layers = [l for l in self.layers if isinstance(l, Layer)]
        for layer in layers:
            layer.reset_gradients() 

    def __call__(self, x):
        """
        Function call to perform forward pass

        Args:
            x (np.ndarray): mini-batch x during forward pass of training or prediction
        
        Returns:
            np.ndarray: returns output values of Net done by calling the forward method of Net class
        """
        return self.forward(x)

    ##############################################################
    #                                                            #
    #  BATCH TRAINING & VALIDATION                               #
    #                                                            #
    ##############################################################

    def train_batch(self, x, label):
        """
        Method that performs a training step on mini-batch x. Makes a forward pass, calculates loss and probabilities with
        chosen loss criterion, then makes a backward pass to update the weights and biases of each Linear layer. 

        Args:
            x (np.ndarray): mini-batch x of training set during forward pass of the training phase
            label (np.ndarray): ground-truth labels of mini-batch x, used for evaluating loss and accuracy
        
        Returns:
            np.ndarray, np.ndarray: returns the calculated loss for the training iteration and proportion of correctly predicted labels 
        """

        self.reset_gradients()
        out = self.forward(x)
        loss, prob = self.criterion(out, label)        
        self.backward(self.criterion.backward())
        self.update()   # Optizer step
        pred, target = np.argmax(prob, axis=1), np.argmax(label, axis=1)
        correct = np.sum(pred==target) / x.shape[0]

        return loss, correct

    def validate_batch(self, valid_x, valid_y, batch_size=20):
        """
        Method that performs a validation step on mini-batch x. Makes a forward pass, calculates loss and probabilities with
        chosen loss criterion, then makes a backward pass to update the weights and biases of each Linear layer. 

        Args:
            x (np.ndarray): mini-batch x of the validation set during forward pass of training
            label (np.ndarray): ground-truth of labels of mini-batch x, used for evaluating loss and accuracy
        
        Returns:
            np.ndarray, np.ndarray: returns the calculated loss for the training iteration and proportion of correctly predicted labels 
        """

        N = valid_x.shape[0]

        losses = 0
        correct = 0

        for START in range(0, N, batch_size):
            END = min(START + batch_size, N)
            
            x, label = valid_x[START : END], valid_y[START : END]

            out = self.forward(x, predict=True)

            loss, prob = self.criterion(out, label)
            losses += loss * (END - START)
            pred, target = np.argmax(prob, axis=1), np.argmax(label, axis=1)

            correct += np.sum(pred==target)

        return losses/N, correct/N

    ##############################################################
    #                                                            #
    #  NETWORK TRAINING TILL CONVERGENCE OR N EPOCHS             #
    #                                                            #
    ##############################################################
    
    def train_convergence(self, train_set, valid_set, batch_size=20, threshold=0.01, report_interval=10, planned_epochs=1000, last_check=10):
        """
        Method for training model using an objective criteria that measures "convergence." Two conditions are used to measure convergence:
        (1) If training loss in a subsequent model is not below a selected N percentage threshold, then convergence is achieved; or 
        (2) If validation loss in chosen M subsequent models is not less than the last best model with minimal validation loss, then convergence is achieved.
        Using these two disjunction of these two objectives, provides an approximate estimate for a good model using both the training loss 
        to check if loss is not decreasing enough, and validation loss to suggest that no better model can be found within reasonable time.

        Args:
            train_set (np.ndarray, np.ndarray): tuple whereby first index refers to training data and second index refers to training labels
            valid_set (np.ndarray, np.ndarray): tuple whereby first index refers to validation data and second index refers to validation labels
            batch_size (int): size of each mini-batch used within each training iteration (each epoch)
            threshold (float): a chosen percentage for the difference in training loss calculated each epoch to decide when to end training
            report_interval (int): represents the number of epochs before each reporting interval. If 10 is set, a report is produced each 10 epochs in training.
            planned_epochs (int): the maximum number of epoch to train model in the case that convergence criteria is not achieved within this time.
            last_check (int): number of models to check after best model with minimum validation loss, if none have lower validation loss then convergence is achieved.

        Returns:
            int, np.ndarray, np.ndarray, np.ndarray, np.ndarray: returns best model epoch, training loss log, training accuracy log, validation loss log
            and validation accuracy log.
        """
        
        train_x, train_y = train_set
        valid_x, valid_y = valid_set

        best_model = {"ep":0,"t_loss":0,"t_acc":0,"v_loss":0,"v_acc":0}

        t_loss_graph, t_acc_graph = np.zeros(planned_epochs), np.zeros(planned_epochs)
        v_loss_graph, v_acc_graph = np.zeros(planned_epochs), np.zeros(planned_epochs)

        start_interval, train_start, prev_train_loss = time(), time(), 0
        N, no_decrease, val_loss, val_loss_min = train_x.shape[0], 0, 0, np.Inf

        self.optimizer.epochs = planned_epochs
        self.optimizer.iters = planned_epochs * (N // batch_size)

        time_step = 0
        for ep in tqdm(range(planned_epochs)):

            self.optimizer.epoch = ep
            order = np.random.permutation(N)  # shuffling indices
            train_loss, train_acc = 0, 0

            for START in range(0, N, batch_size):

                END = min(START + batch_size, N)
                i = order[START : END]   # batch indices
                
                x, label, time_step = train_x[i], train_y[i], time_step + 1
                self.optimizer.time_step = time_step 

                loss, acc = self.train_batch(x, label)
                train_loss, train_acc = train_loss + loss, train_acc + acc
                
            train_acc = train_acc / (N // batch_size)
            train_loss = train_loss / (N // batch_size)

            val_loss, val_acc = self.validate_batch(valid_x, valid_y, batch_size)

            if ep % report_interval == 0:
                # show train log for interval
                self._train_log(ep, train_loss, train_acc, val_loss, val_acc, start_interval)
                start_interval = time()  # reset timer for new report interval

            # check if train loss not decreasing enough
            train_convergence = (prev_train_loss > 0) and (1 - train_loss / prev_train_loss) < (threshold / 100)

            # check if validation loss stops decreasing re last 5 models - more likely to occur first
            valid_convergence = no_decrease >= last_check and val_loss_min < val_loss and val_loss_min > 0

            if val_loss <= val_loss_min:
                val_loss_min, no_decrease = val_loss, 0
                best_model = {
                    "ep":ep, 
                    "t_loss":train_loss,
                    "t_acc":train_acc,
                    "v_loss":val_loss,
                    "v_acc":val_acc,
                }
                self.save_model(train=True)

            else: no_decrease += 1
            prev_train_loss = train_loss

            if valid_convergence or train_convergence:
                if valid_convergence:
                    print(f"\n\nNo decrease in validation loss during last {last_check} epoch(s).")
                else:
                    print(f"\n\nMinimum percent change ({threshold}%) in training loss not exceeded.")

                print(f"\nConvergence criteria achieved.\nTraining completed @ Epoch {ep}.")
                self._train_finish(train_start, best_model)

                return best_model['ep'], t_loss_graph[:ep], t_acc_graph[:ep], v_loss_graph[:ep], v_acc_graph[:ep]

            t_loss_graph[ep], t_acc_graph[ep], v_loss_graph[ep], v_acc_graph[ep] = train_loss, train_acc, val_loss, val_acc

        print(f"\n\nMaximum planned number of epoch(s) exhausted.\n\nTraining is complete @ Epoch {ep}.")
        self._train_finish(train_start, best_model)
        self.save_model(train=True)

        return best_model['ep'], t_loss_graph, t_acc_graph, v_loss_graph, v_acc_graph

    def train_network(self, train_set, valid_set, epochs, batch_size=20, report_interval=1):
        """
        Method for training model over a selected (fixed) number of epochs. 

        Args:
            train_set (np.ndarray, np.ndarray): tuple whereby first index refers to training data and second index refers to training labels
            valid_set (np.ndarray, np.ndarray): tuple whereby first index refers to validation data and second index refers to validation labels
            batch_size (int): size of each mini-batch used within each training iteration (each epoch)
            epochs (int): number of epochs to train the network
            report_interval (int): represents the number of epochs before each reporting interval. If 10 is set, a report is produced each 10 epochs in training.

        Returns:
            int, np.ndarray, np.ndarray, np.ndarray, np.ndarray: returns best model epoch, training loss log, training accuracy log, validation loss log
            and validation accuracy log.
        """
        train_x, train_y = train_set
        valid_x, valid_y = valid_set

        best_model = {"ep":0,"t_loss":0,"t_acc":0,"v_loss":0,"v_acc":0}

        t_loss_graph, t_acc_graph = np.zeros(epochs), np.zeros(epochs)
        v_loss_graph, v_acc_graph = np.zeros(epochs), np.zeros(epochs)

        start_interval, train_start = time(), time()
        N, val_loss, val_loss_min = train_x.shape[0], 0, np.Inf

        self.optimizer.epochs = epochs
        self.optimizer.iters = epochs * (N // batch_size)

        time_step = 0
        for ep in tqdm(range(epochs)):

            self.optimizer.epoch = ep
            order = np.random.permutation(N)  # shuffling indices to prevent learning order of training
            train_loss, train_acc = 0, 0

            for START in range(0, N, batch_size):

                END = min(START + batch_size, N)
                i = order[START : END]   # mini batch indices
                
                x, label, time_step = train_x[i], train_y[i], time_step + 1
                self.optimizer.time_step = time_step 

                loss, acc = self.train_batch(x, label)
                train_loss, train_acc = train_loss + loss, train_acc + acc
                
            train_acc = train_acc / (N // batch_size)
            train_loss = train_loss / (N // batch_size)

            val_loss, val_acc = self.validate_batch(valid_x, valid_y, batch_size)

            if ep % report_interval == 0:
                # show train log for interval
                self._train_log(ep, train_loss, train_acc, val_loss, val_acc, start_interval)
                start_interval = time()  # reset start

            if val_loss <= val_loss_min:
                val_loss_min = val_loss
                best_model = {
                    "ep":ep, 
                    "t_loss":train_loss,
                    "t_acc":train_acc,
                    "v_loss":val_loss,
                    "v_acc":val_acc,
                }
                self.save_model(train=True)

            t_loss_graph[ep], t_acc_graph[ep], v_loss_graph[ep], v_acc_graph[ep] = train_loss, train_acc, val_loss, val_acc

        print(f"\n\nModel has finished training after {ep} epoch(s).\n\n")
        self._train_finish(train_start, best_model)
        self.save_model(train=True)

        return best_model['ep'], t_loss_graph, t_acc_graph, v_loss_graph, v_acc_graph

    ########################################
    #                                      #                      
    #  TESTING FUNCTIONS                   #
    #                                      #                      
    ########################################

    def _predict(self, x):
        """
        Performs a single forward pass and returns the output of applying Softmax on the mini-batch sample x

        Args:
            x (np.ndarray): a mini-batch x used in forward pass, for testing or making predictions
        Returns:
            (np.ndarray): returns the probabilities for multi-classification by applying Softmax actuvation on the last layer's output
        """
        out = self.forward(x, predict=True)
        return self.criterion(out)

    def predict(self, x, n_classes):
        """
        Performs a forward pass and builds an array of predictions made for each example in the test data the output of applying Softmax on the mini-batch sample x

        Args:
            x (np.ndarray): a mini-batch x used in forward pass, for testing or making predictions
        Returns:
            (np.ndarray): returns the probabilities for multi-classification by applying Softmax actuvation on the last layer's output
        """
        ret = np.zeros((x.shape[0], n_classes))

        for i in range(x.shape[0]):
            ret[i] = self._predict(x[i,:])

        return ret


    def test_network(self, test_set, data="test data"):
        """
        Method for testing the network, first performs a forward pass to predict the labels for mini-batch x. This is then used against
        ground-truth labels to calculate accuracy, and also includes the number of correct predictions (accuracy) for each class label 

        Args:
            x (np.ndarray): a test mini-batch x used in forward pass, for testing or making predictions
            data (str): default value test data which is used in formatting the statistics of correct predictions made on
                            the kind of data x belongs to. This can be over-ridden, and used to format output for accuracy results
                            of train, validaton and test data.
        
        """

        x, labels = test_set
        n_classes = labels.shape[1]

        out = self.predict(x, n_classes)
        pred, target = np.argmax(out, axis=1), np.argmax(labels, axis=1)

        correct = np.sum(pred==target)

        match = [0] * n_classes
        count = [0] * n_classes
        for p, t in zip(pred, target):
            if p == t:
                match[t-1] += 1
            count[t-1] += 1

        print("-------------------------------------------")
        print(f"Accuracy on {data}: {(correct / pred.shape[0]) * 100:.2f}%")
        print("Total Count: ", pred.shape[0])
        print("Total Match: ", correct)
        print("-------------------------------------------")
        for i in range(n_classes): 
            print(f'Test Accuracy of\t{i}: {match[i] / count[i] * 100:.2f}% ({np.sum(match[i])}/{np.sum(count[i])})')

    
    ############################################################
    #                                                          #
    #  TRAINING OUTPUT LOGS HELPERS                            #
    #                                                          #
    ############################################################

    def _display(self, best_model):
        return f"""
                Best model found @ Epoch {best_model['ep']}
                --------------------------------------------
                Training Loss: {best_model['t_loss']:.6f}
                Validation Loss: {best_model['v_loss']:.6f}
                --------------------------------------------
                Training Accuracy: {best_model['t_acc']:.6f}
                Validation Accuracy: {best_model['v_acc']:.6f}\n"""

    def _train_log(self, ep, train_loss, train_acc, val_loss, val_acc, start_interval):
        elapsed = time() - start_interval  # update elapsed
        print(f"Epoch: {ep}\tInterval Time: {show_time(elapsed)}\tTraining Loss: {train_loss:.6f}\t\tTraining Accuracy: {train_acc:.6f}")
        print(f"\t\t\t\t\t\tValidation Loss:{val_loss:.6f}\tValidation Accuracy: {val_acc:.6f}")

    def _train_finish(self, train_start, best_model):
        print(f"Total training time: {show_time(time() - train_start)}")
        print(self._display(best_model))
        print(f"\nBest model '{self.model_name}' saved in 'model/' directory.")


    ############################################################
    #                                                          #
    #  MODEL SAVING AND LOADING                                #
    #                                                          #
    ############################################################
    
    def save_model(self, train=False):
        """
        Instance method for saving a Net object (i.e. it's state) whenever it is called.
        Mostly used during training, whereby when validation loss is minimized by the model
        in this epoch - it's state is saved as the 'best model' representation of itself.

        Args:
            train (bool): Used to control the flow of execution when training, not to print
            output message for successful / unsuccessful save as this is done repeatedly.

        Returns:
            None: The recursive return statement is used to try saving a model again, when
            the model directly does not yet exist within the adjacent subfolder this method
            is called.
        """
        file_path = os.path.join(os.getcwd(), 'model')
        if os.path.exists(file_path):
            try:
                with open(file_path + "/" + repr(self), "wb") as file:
                    pickle.dump(self, file, protocol = pickle.HIGHEST_PROTOCOL)

                if train: return

                print("\nModel Save Successful!", end='\n\n')
                print(f"Model name: {repr(self)}")
                format_path = file_path.replace("\\","/") + '/'
                print(f"Saved in: {format_path}", end='\n\n')
                print(f"Full path: {format_path + repr(self)}")
            except Exception as e:
                print("Save unsuccessful: ", e)
        else:
            os.mkdir(file_path)
            return self.save_model(train)

    
    @staticmethod
    def load_model(file_path):
        """
        Class method used for loading a Net object. Can be used for loading any model 
        saved previously, or to reload the same network (self), but as a previous state
        that is deemed *best model* in minimizing validation loss.

        Args:
            file_path (str): File path where the model is stored. Usually file path refers
            to a `model` sub-directory.
        """
        try:
            with open(file_path, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            print("Load unsuccessful: ", e)


if __name__ == "__main__":

    model1 = Net.load_model("network/model/AdamCELoss[1024]LeakyReLU[64]LeakyReLU[32]LeakyReLU[10]")
    assert repr(model1) == "AdamCELoss[1024]LeakyReLU[64]LeakyReLU[32]LeakyReLU[10]"

    # will only work for my pc obviously
    model2 = Net.load_model("C:/Users/imgap/github/MLPLibrary/network/model/AdamCELoss[1024]LeakyReLU[64]LeakyReLU[32]LeakyReLU[10]")
    assert repr(model2) == "AdamCELoss[1024]LeakyReLU[64]LeakyReLU[32]LeakyReLU[10]"

    assert repr(model1) == repr(model2)

    model3 = Net(optimizer=SGD(weight_decay=0.01, momentum=0.5), criterion=CrossEntropyLoss())
    model3.add(Linear(128, 1024))
    model3.add(ReLU())
    model3.add(Linear(1024, 64))
    model3.add(ReLU())
    model3.add(Linear(64, 10))

    assert repr(model3) == "SGD(wd+mm)CELoss[1024]ReLU[64]ReLU[10]"

    model3.set_name("SGD_mm_1024_64_10_ReLU")

    model3.save_model()
    assert repr(model3) == repr(Net.load_model("network/model/SGD_mm_1024_64_10_ReLU"))