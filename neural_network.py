import torch
from tqdm import tqdm

#Binary Classification
class Perceptron():
    def __init__(self):
        self.weights = None
        self.biases = None
    
    @staticmethod
    def get_ovr_labels(y_train, y_test, no_classes):
        y_train_ovr = {}
        y_test_ovr = {}
        for clas in range(no_classes):
            y_train_ovr[clas] = torch.tensor([1 if x == clas else 0 for x in y_train])
            y_test_ovr[clas] = torch.tensor([1 if x == clas else 0 for x in y_test])
        
        return y_train_ovr, y_test_ovr
    
    @staticmethod
    def sigmoid(X):
        return torch.exp(-torch.logaddexp(torch.zeros_like(X), -X))
    
    def activation(self, X):
        return self.sigmoid(X.matmul(self.weights) + self.biases)
    
    def initialize_parameters(self, length):
        self.weights = torch.zeros(length)
        self.biases = torch.zeros(1)
        
    def propagate(self, X, y):
        m = torch.tensor([X.shape[0]])

        def forward():
            A = self.activation(X)
            loss = -(y * torch.log(A) + (1 - y) * torch.log(1 - A)).sum() / m 
            return A, loss

        def backward(A):
            error = A.sub(y)
            dw = error.matmul(X) / m # weights gradient
            db = A.sub(y).sum() / m # bias gradient
            return dw, db

        A, loss = forward()
        dw, db = backward(A)

        return dw, db, loss
    
    def fit(self, X, y, epoch, learning_rate, print_loss=False):
        self.initialize_parameters(4)

        for i in tqdm(range(epoch)):
            dw, db, loss = self.propagate(X, y)
            self.weights -= learning_rate * dw
            self.biases -= learning_rate * db
            
            if print_loss:
                print(f"Iteration {i}, loss: {loss}")

    
    def predict(self, X):
        A = self.activation(X)
        y_pred = torch.where(A >= 0.5, 1, 0)
        return y_pred
