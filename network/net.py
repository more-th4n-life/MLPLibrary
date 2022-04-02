from loss import CrossEntropyLoss
from optim import SGD
from layer import Layer
import numpy as np

class Net:
    """
    MLP Model class 

    Currently supports: 
        Layers: Linear 
        Activations: ReLU, LeakyReLU
        Loss Criteria: CrossEntropyLoss (uses Softmax 'activation' on output layer)
        Optimizer: SGD + Weight Decay 

    """
    def __init__(self, optimizer=SGD, criterion=CrossEntropyLoss):
        self.layers, self.size = [], 0
        self.optimizer, self.criterion = optimizer, criterion

    def add(self, layer):
        """
        Add layer: e.g. Linear or Activation
        """
        self.layers += [layer]
        self.size += 1

    def forward(self, x):
        """
        Forward pass
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dy):
        """
        Backward pass
        """
        for layer in self.layers[::-1]:
            dy = layer.backward(dy)

    def update(self):
        """
        Uses optimizer to update weights and biases in each layer based on saved dW and db
        """
        self = self.optimizer.step(self)

    def reset_gradients(self):
        """
        Zeros gradients in each layer after each training iteration
        """
        layers = [l for l in self.layers if isinstance(l, Layer)]
        for layer in layers:
            layer.reset_gradients() 

    def __call__(self, x):
        return self.forward(x)

    def train_batch(self, x, label):

        self.reset_gradients()
        out = self.forward(x)
        loss, _ = self.criterion(out, label)        
        self.backward(self.criterion.backward())

        self.update() # SGD step

        return loss

    def validate_batch(self, valid_x, valid_y, batch_size=20):
        N = valid_x.shape[0]

        losses = 0
        correct = 0

        for START in range(0, N, batch_size):
            END = min(START + batch_size, N)
            
            x, label = valid_x[START : END], valid_y[START : END]
            
            out = self.forward(x)
            loss, prob = self.criterion(out, label)
            losses += loss * (END - START)
            pred, target = np.argmax(prob, axis=1), np.argmax(label, axis=1)

            correct += np.sum(pred==target)

        return losses/N, correct/N

    def train_network(self, train_set, valid_set, epochs, batch_size=20):
        """
        Mini batch training.. separated from train that uses a dataloader which can also load batches, but
        I think that it could be overkill and also doesn't shuffle / take random samples like it should

        This is not main train yet because not working well for low sizes, e.g. batch_size=1 or 2
        however, just noticed it might be related to the log function in ce loss
        """
        train_x, train_y = train_set
        valid_x, valid_y = valid_set

        N = train_x.shape[0]
        loss_graph = np.zeros(epochs)

        for ep in range(epochs):

            order = np.random.permutation(N)
            train_loss = 0

            for START in range(0, N, batch_size):

                END = min(START + batch_size, N)
                i = order[START : END] 
                
                x, label = train_x[i], train_y[i]

                train_loss += self.train_batch(x, label)
            
            train_loss = train_loss / (N // batch_size)

            val_loss, val_acc = self.validate_batch(valid_x, valid_y, batch_size)

            if ep % 10 == 0:
                print(f"Epoch: {ep} \t Training Loss:{train_loss:.6f}")
                print(f"Epoch: {ep} \t Validation Loss:{val_loss:.6f} \t Validation Accuracy: {val_acc:.6f}")

            loss_graph[ep] = train_loss

        return loss_graph
    
    def train(self, train_loader, valid_loader, epochs):
        """
        A training function, load in train / validate data loaders
        Batch size is applied within loaders themselves that can be configured in loader/test_loader.py where loaders are imported
        """
        for ep in range(epochs):
            tr_loss, tr_accu = [], []
            va_loss, va_accu = [], []

            for x, label in train_loader:
                
                self.reset_gradients()
    
                out = self.forward(x)
                loss, pred = self.criterion(out, label)

                pred, target = np.argmax(pred, axis=1), np.argmax(label, axis=1)
                
                self.backward(self.criterion.backward())
                self.update() # SGD step

                tr_loss.append(loss)
                tr_accu.append(np.sum(pred==target) / x.shape[0])

            for x, label in valid_loader:

                out = self.forward(x)

                loss, pred = self.criterion(out, label)
                pred, target = np.argmax(pred, axis=1), np.argmax(label, axis=1)

                va_loss.append(loss)
                va_accu.append(np.sum(pred==target) / x.shape[0])

            print(f"Epoch: {ep+1} \t Training Loss:{np.mean(tr_loss):.6f} \t Training Accuracy: {np.mean(tr_accu):.6f}")
            print(f"Epoch: {ep+1} \t Validate Loss:{np.mean(va_loss):.6f} \t Validate Accuracy: {np.mean(va_accu):.6f}")

    
    def test2(self, test_set):
        """
        Still being fixed, to work with train_mb 
        """
        test_x, test_y = test_set
            
        test_loss = []
        correct = [0] * 10
        size = [0] * 10
        accu = []

        for i in range(0, test_x.shape[0]):

            x = test_x[i]
            label = test_y[i]

            out = self.forward(x)

            loss = self.criterion(out, label)

            test_loss.append(loss)

            pred, target = np.argmax(out, axis=1), np.argmax(label, axis=1)
            matches = np.squeeze(pred==target)

            for i in range(target.shape[0]):
                y = target[i]
                correct[y] += matches[y].item()
                size[y] += 1

            accu.append(np.sum(pred==target) / x.shape[0])
        
        print(f"Test loss: {np.mean(test_loss)}")
        print(f"Test accu: {np.mean(accu)}")

        for i in range(10):
           
            print(f'Test Accuracy of\t{i}: {correct[i] / size[i] * 100:.2f}% ({np.sum(correct[i])}/{np.sum(size[i])})')

    
    def test(self, test_loader):
        """
        Test function, loads in test data loader 
        """
        test_loss = []
        correct = [0] * 10
        size = [0] * 10
        accu = []

        for x, label in test_loader:

            out = self.forward(x)

            loss, pred = self.criterion(out, label)

            test_loss.append(loss)

            pred, target = np.argmax(pred, axis=1), np.argmax(label, axis=1)
            matches = np.squeeze(pred==target)

            for i in range(target.shape[0]):
                y = target[i]
                correct[y] += matches[y].item()
                size[y] += 1

            accu.append(np.sum(pred==target) / x.shape[0])
        
        print(f"Test loss: {np.mean(test_loss)}")
        print(f"Test accu: {np.mean(accu)}")

        for i in range(10):
           
            print(f'Test Accuracy of\t{i}: {correct[i] / size[i] * 100:.2f}% ({np.sum(correct[i])}/{np.sum(size[i])})')
