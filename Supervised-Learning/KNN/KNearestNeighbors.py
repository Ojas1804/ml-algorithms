import numpy as np
import pandas as pd
from collections import Counter 

class KNearestNeighbors:
    def __init__(self, k = 5):
        self.k = k


    def predict(self, X_pred):
        y_pred = []
        for x in X_pred:
            y_pred.append(self._predict(x))
        return np.array(y_pred)


    def _predict(self, x):
        distance = []
        for x_train in self.X_train:
            distance.append(self.eucledian_distance(x, x_train))
        distance = np.array(distance)

        indices = np.argsort(distance)
        k_indices = indices[: self.k]
        k_neighbors = []
        
        for i in k_indices:
            k_neighbors.append(self.y_train[i])
        most_common_label = Counter(k_neighbors).most_common(1)
        return most_common_label[0][0]


    def eucledian_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def train(self, X, y, normalize = False):
        self.X_train = X
        self.y_train = y
        self.normailze = normalize
        if(normalize):
            self.X_train = normalize(X_train)


    def accuracy(self, y_pred, y_test):
        correct_predictions = 0
        for (y1, y2) in zip(y_pred, y_test):
            if(y1 == y2):
                correct_predictions += 1
        return (correct_predictions / len(y_test) * 100)