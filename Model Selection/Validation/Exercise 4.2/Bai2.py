import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def readData(folder , filename):
    data = np.loadtxt(os.path.join(folder, filename), delimiter = ',')
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    one = np.ones((X.shape[0], 1))
    X = np.concatenate((one, X), axis = 1)
    return X, y
    
def normScaling(X):
  for col in range(1, X.shape[1]):
    temp = X[:, col]
    X[:, col] = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
 
def standardScaling(X):
  for col in range(1, X.shape[1]):
    temp = X[:, col]
    X[:, col] = (temp - np.mean(temp)) / (np.std(temp))

def predict(X, w):
    h_w = 1 / (1 + np.exp(- np.dot(X, w)))
    return (h_w >= 0.5).astype('int32')

def costFunction(X, y ,w):
    m = X.shape[0]
    h_w = 1 / (1 + np.exp(- np.dot(X, w)))
    J_w = (-1/m) * (np.dot(y.T, np.log(h_w)) + np.dot((1-y).T, np.log(1-h_w)))
    return J_w[0, 0]

def gradient(X, y, w):
    m = X.shape[0]
    h_w = 1 / (1 + np.exp(- np.dot(X, w)))
    return (1/m) * np.dot(X.T, h_w - y)

def gradientDescent(X, y, w_init, alpha, n = 1500):
    w_old = w_init.reshape(-1, 1)
    cost_values = []
    for i in range(n):
        w_new = w_old - alpha * gradient(X, y, w_old)
        cost_values.append(costFunction(X, y, w_new))
        w_old = w_new
    return w_new, cost_values
        
def main():
    FOLDER = r'D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\Model Selection\Validation\Exercise 4.2'
    # Đọc dữ liệu
    X, y = readData(FOLDER, 'ex2data1.txt')
    # Chia tập dữ liệu thành training set và test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state = 5)
    # Chuẩn hóa dữ liệu
    standardScaling(X_train)
    standardScaling(X_test)

    # Huấn luyện mô hình bằng gradient descent
    w_init = np.zeros((X_train.shape[1], 1))
    w_opt, loss = gradientDescent(X=X_train,y=y_train, 
                                           w_init=w_init, alpha=0.01, n=1500)
    # Dự đoán
    y_pred = predict(X_test, w_opt)
    
    # Đánh giá hiệu năng của mô hình
    print("Đánh giá hiệu năng mô hình")
    print("\tAccuracy: ", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()