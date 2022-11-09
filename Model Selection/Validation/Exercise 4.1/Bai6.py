import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def readData(folder , filename):
    data = np.loadtxt(os.path.join(folder, filename), delimiter = ',')
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    one = np.ones((X.shape[0], 1))
    X = np.concatenate((one, X), axis = 1)
    return X, y

def featureScalingSplit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train[:, 1:] = sc_X.fit_transform(X_train[:, 1:])
    X_test[:, 1:] = sc_X.transform(X_test[:, 1:])
    y_train = sc_y.fit_transform(y_train)
    y_test = sc_y.transform(y_test)
    return X_train, X_test, y_train, y_test

def kFoldCrossValiation(X_train, y_train, k):
    pass
        
def main():
    FOLDER = r'D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\Model Selection\Validation'
    X, y = readData(FOLDER, 'ex1data1.txt')
    X_train, X_test, y_train, y_test = featureScalingSplit(X, y)
    regressor = LinearRegression()
    scores = cross_validate(regressor, X_train, y_train, cv=10,
                        scoring='neg_mean_squared_error',
                        return_train_score=True)
    print(scores['test_score'])
    
    y_pred = cross_val_predict(regressor, X_test, y_test, cv=10)
    print(mean_squared_error(y_test, y_pred))
if __name__ == "__main__":
    main()