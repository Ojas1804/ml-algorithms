import pandas as pd
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split


def main():
    housing = pd.read_csv("USA_Housing.csv")
    housing.drop(columns = "Address", inplace = True)
    X = pd.DataFrame(housing.iloc[:1000 , 0])
    y = pd.DataFrame(housing.iloc[:1000 , -1])
    print(X.shape, y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.25)
    print(X_train.shape, y_train.shape)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    print(y_pred)

    print("Mean-Squared Error = ", lr.mean_squared_error(y_test, y_pred))
    plt.show()


if __name__ == "__main__":
    main()