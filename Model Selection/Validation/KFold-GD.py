import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def readData(folder , filename):
    data = np.loadtxt(os.path.join(folder, filename), delimiter = ',')
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    one = np.ones((X.shape[0], 1))
    X = np.concatenate((one, X), axis = 1)
    return X, y

def predict(X, w):
    return np.dot(X, w)

def calculateLoss(X, y, w):
    h = np.dot(X, w)
    m = X.shape[0]
    J = (1/ (2*m)) * np.sum(np.square(h-y))
    return J

def gradient(X, y, w):
    m = X.shape[0]
    h = np.dot(X, w)
    return (1/m) * np.dot(X.T, h - y)

def gradientDescent(X, y, w_init, alpha, n = 1500):
    w_optimal = w_init.reshape(-1, 1)
    loss_values = []
    for i in range(n):
        w_optimal = w_optimal - alpha * gradient(X, y, w_optimal)
        j = calculateLoss(X, y, w_optimal)
        loss_values.append(j)
    return w_optimal, loss_values

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
    w_init = np.zeros((X_train.shape[1], 1))
    step = 0
    valitdate_models = {}
    temp = []
    for train2_index, val_index in kf.split(X = X_train,y = y_train):
        step = step + 1
        print('\tBước lặp huấn luyện thứ: ', step)
        X_train2, X_val = X_train[train2_index], X_train[val_index]
        y_train2, y_val = y_train[train2_index], y_train[val_index]
        w_opt, J_history = gradientDescent(X=X_train2,y=y_train2, 
                                           w_init=w_init, alpha=0.01, n=1500)
        print('\t\tĐánh giá mô hình trên tập dữ liệu validation')
        y_pred = predict(X_val, w_opt)
        print('\t\t\tMSE: ', mean_squared_error(y_val, y_pred))
        valitdate_models[mean_squared_error(y_val, y_pred)] = w_opt 
    min_error = min(list(valitdate_models.keys()))
    return valitdate_models[min_error]

def validateTestSet(X_test, y_test, w_opt):
    print("\nĐánh giá hiệu năng mô hình trên tập dữ liệu test:")
    y_pred = predict(X_test, w_opt)
    print("\tMSE: ", mean_squared_error(y_test, y_pred))
    return mean_squared_error(y_test, y_pred)
        
def main():
    FOLDER = r'D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\Model Selection\Validation'
    X, y = readData(FOLDER, 'ex1data1.txt')
    X_train, X_test, y_train, y_test = featureScalingSplit(X, y)
    w_optimal = kFoldCrossValiation(X_train, y_train)
    validateTestSet(X_test, y_test, w_optimal)
    
if __name__ == "__main__":
    main()