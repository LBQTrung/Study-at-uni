import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def readData(folder , filename):
    data = np.loadtxt(os.path.join(folder, filename), delimiter = ',')
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    one = np.ones((X.shape[0], 1))
    X = np.concatenate((one, X), axis = 1)
    return X, y
    
def normScaling(X, y):
  for col in range(1, X.shape[1]):
    temp = X[:, col]
    X[:, col] = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
  temp = y[:, 0]
  y[:, 0] = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
 
def standardScaling(X, y):
  for col in range(1, X.shape[1]):
    temp = X[:, col]
    X[:, col] = (temp - np.mean(temp)) / (np.std(temp))
  temp = y[:, 0]
  y[:, 0] = (temp - np.mean(temp)) / (np.std(temp))

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
        
def main():
    FOLDER = r'D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\Model Selection\Validation\Exercise 4.1'
    # Đọc dữ liệu
    X, y = readData(FOLDER, 'ex1data2.txt')
    # Chia tập dữ liệu thành training set và test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state = 5)
    # Chuẩn hóa dữ liệu
    standardScaling(X_train, y_train)
    standardScaling(X_test, y_test)

    # Huấn luyện mô hình bằng gradient descent
    w_init = np.zeros((X_train.shape[1], 1))
    w_opt, loss = gradientDescent(X=X_train,y=y_train, 
                                           w_init=w_init, alpha=0.01, n=1500)
    # Dự đoán
    y_pred = predict(X_test, w_opt)

    # Đánh giá hiệu năng của mô hình
    print("Đánh giá hiệu năng mô hình")
    print("\tMSE: ",mean_squared_error(y_test, y_pred))
    print("\tRMSE: ",mean_squared_error(y_test, y_pred) ** (1/2))
if __name__ == "__main__":
    main()