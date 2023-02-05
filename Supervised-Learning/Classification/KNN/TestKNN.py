from KNearestNeighbors import KNearestNeighbors

from sklearn import datasets
from sklearn.model_selection import train_test_split 

def main():
    iris_dataset = datasets.load_iris()
    X = iris_dataset.data
    y = iris_dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
#     X_train, X_test, y_train, y_test = split_dataset(X, y, 0.3)

    knn = KNearestNeighbors(8)
    knn.train(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(knn.accuracy(y_pred, y_test))


if __name__ == "__main__":
    main()
