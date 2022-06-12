import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, learning_rate = 0.001, iterations = 100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.bias = None
        self.weight = None
        
        
    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        
        self.weight = np.zeros(X_train.shape[1])
        self.bias = 0
        
        for i in range(self.iterations):
            y_pred = pd.DataFrame(np.dot(X_train, self.weight) + self.bias)
            print(y_pred.shape, y_train.shape, X_train.shape)
            
            dw = (1 / X_train.shape[0]) * (np.dot(X_train.T, (y_pred - y_train)))
            db = (1 / X_train.shape[0]) * (np.sum(y_pred - y_train))
            
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    
    def predict(self, X):
        y_pred = np.dot(X, self.weight) + self.bias
        return y_pred

    def mean_squared_error(self, y_test, y_pred):
        return np.mean((y_test - y_pred) ** 2)