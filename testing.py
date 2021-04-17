import torch 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from neural_network import Perceptron
from naive_bayes import GaussianNaiveBayes
from linear_model import LinearRegressor, SGDRegressor

def test_bayes():

    iris = load_iris()
    #dataframe input required
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    X, y = df.drop('target', 1), df[['target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1, stratify=y, shuffle=True)

    clas = GaussianNaiveBayes()
    clas.fit(X_train, y_train)

    pred2 = clas.predict(X_test)
    print("F1: ", f1_score(pred2, y_test, average='micro'))

def test_perceptron():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, random_state=1, stratify=y)
    
    X_train = torch.tensor(X_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    
    model = Perceptron()
    y_train_ovr, y_test_ovr = model.get_ovr_labels(y_train, y_test, 3)
    model.fit(X_train, y_train_ovr[1], 2000, 0.5)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)


    print("Train accuracy: {} %".format(accuracy_score(y_train_ovr[1], y_pred_train)))
    print("Test accuracy: {} %".format(accuracy_score(y_test_ovr[1], y_pred_test)))
    
def test_linear_regression():
    
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    lin_reg = LinearRegressor()
    lin_reg.fit(X, y)
    predicted = lin_reg.predict(X)
    
    plt.scatter(X, y, label="Original")
    plt.plot(X, predicted, color="red", label="Predicted")
    plt.legend()
    plt.show()
   
   
def test_SGDRegressor():
    
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    reg = SGDRegressor()
    reg.fit(X, y)
    predicted = reg.predict(X)
    
    plt.scatter(X, y, label="Original")
    plt.plot(X, predicted, color="red", label="Predicted")
    plt.legend()
    plt.show()
    
    
def main():
    test_bayes()
    
if __name__ == "__main__":
    main()
    