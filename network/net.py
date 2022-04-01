from loss import CrossEntropyLoss
from optim import SGD
from layer import Layer
import numpy as np

class Net:
    def __init__(self, optimizer=SGD, criterion=CrossEntropyLoss):
        self.layers, self.size = [], 0
        self.optimizer = optimizer
        self.criterion = criterion

    def add(self, layer):
        self.layers += [layer]
        self.size += 1

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def reset_gradients(self):
        layers = [l for l in self.layers if isinstance(l, Layer)]
        for layer in layers:
            layer.reset_gradients() 

    def backward(self, dy):
        for layer in self.layers[::-1]:
            dy = layer.backward(dy)
        return dy

    def update(self):
        self = self.optimizer.step(self)

    def __call__(self, x):
        return self.forward(x)

    def train_mb(self, train_set, valid_set, epochs, batch_size=1):
        """
        Mini batch training.. separated from train that uses a dataloader which can also load batches, but
        I think that it could be overkill and also doesn't shuffle / take random samples like it should

        This is not main train yet because not working well for low sizes, e.g. batch_size=1 or 2
        however, just noticed it might be related to the log function in ce loss
        """
        mini_batch = batch_size > 1
        perm = lambda x: np.random.permutation(x)

        train_x, train_y = train_set
        valid_x, valid_y = valid_set

        for ep in range(epochs):
            tr_loss, tr_accu = [], []
            va_loss, va_accu = [], []

            for i in range(0, train_x.shape[0], batch_size):

                id = perm(train_x.shape[0])[i:i+batch_size] if mini_batch else [i]
                x, label = train_x[id], train_y[id]

                self.reset_gradients()
                out = self.forward(x)
                loss = self.criterion(out, label)
                pred, target = np.argmax(out, axis=1), np.argmax(label, axis=1)
                
                self.backward(self.criterion.backward())
                self.update() # SGD step

                tr_loss.append(loss)
                tr_accu.append(np.sum(pred==target) / x.shape[0])

            for i in range(0, valid_x.shape[0], batch_size):

                id = perm(valid_x.shape[0])[i:i+batch_size] if mini_batch else [i]                
                x, label = valid_x[id], valid_y[id]

                out = self.forward(x)

                loss = self.criterion(out, label)
                pred, target = np.argmax(out, axis=1), np.argmax(label, axis=1)

                va_loss.append(loss)
                va_accu.append(np.sum(pred==target) / x.shape[0])

            print(f"Epoch: {ep+1} \t Training Loss:{np.mean(tr_loss):.6f} \t Training Accuracy: {np.mean(tr_accu):.6f}")
            print(f"Epoch: {ep+1} \t Validate Loss:{np.mean(va_loss):.6f} \t Validate Accuracy: {np.mean(va_accu):.6f}")

    def train(self, train_loader, valid_loader, epochs):

        for ep in range(epochs):
            tr_loss, tr_accu = [], []
            va_loss, va_accu = [], []

            for x, label in train_loader:
                
                self.reset_gradients()
    
                out = self.forward(x)
                loss = self.criterion(out, label)
                pred, target = np.argmax(out, axis=1), np.argmax(label, axis=1)
                
                self.backward(self.criterion.backward())
                self.update() # SGD step

                tr_loss.append(loss)
                tr_accu.append(np.sum(pred==target) / x.shape[0])

            for x, label in valid_loader:

                out = self.forward(x)

                loss = self.criterion(out, label)
                pred, target = np.argmax(out, axis=1), np.argmax(label, axis=1)

                va_loss.append(loss)
                va_accu.append(np.sum(pred==target) / x.shape[0])

            print(f"Epoch: {ep+1} \t Training Loss:{np.mean(tr_loss):.6f} \t Training Accuracy: {np.mean(tr_accu):.6f}")
            print(f"Epoch: {ep+1} \t Validate Loss:{np.mean(va_loss):.6f} \t Validate Accuracy: {np.mean(va_accu):.6f}")

    def test2(self, test_set):
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

        test_loss = []
        correct = [0] * 10
        size = [0] * 10
        accu = []

        for x, label in test_loader:

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
