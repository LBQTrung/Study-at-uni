import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

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

def kFoldCrossValiation(X_train, y_train):
    kf = KFold(n_splits=10)
    step = 0
    valitdate_models = {}
    for train2_index, val_index in kf.split(X = X_train,y = y_train):
        step = step + 1
        regressor = LinearRegression()
        print('\tBước lặp huấn luyện thứ: ', step)
        X_train2, X_val = X_train[train2_index], X_train[val_index]
        y_train2, y_val = y_train[train2_index], y_train[val_index]
        regressor.fit(X_train2, y_train2)
        print('\t\tĐánh giá mô hình trên tập dữ liệu validation')
        y_pred = regressor.predict(X_val)
        print('\t\t\tMSE: ', mean_squared_error(y_val, y_pred))
        valitdate_models[mean_squared_error(y_val, y_pred)] =  regressor
    min_error = min(list(valitdate_models.keys()))
    return valitdate_models[min_error]

def validateTestSet(X_test, y_test, regressor):
    print("\nĐánh giá hiệu năng mô hình trên tập dữ liệu test:")
    y_pred = regressor.predict(X_test)
    print("\tMSE: ", mean_squared_error(y_test, y_pred))
    return mean_squared_error(y_test, y_pred)
        
def main():
    FOLDER = r'D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\Model Selection\Validation'
    X, y = readData(FOLDER, 'ex1data1.txt')
    X_train, X_test, y_train, y_test = featureScalingSplit(X, y)
    regressor = kFoldCrossValiation(X_train, y_train)
    validateTestSet(X_test, y_test, regressor)
    
if __name__ == "__main__":
    main()