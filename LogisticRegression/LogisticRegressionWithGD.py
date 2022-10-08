import os
import numpy as np

# Preprocessing the dataset
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

# Build Logistic Regression Model
def predict(x, w):
    h_w = 1 / (1 + np.exp(- np.dot(x, w)))
    if h_w[0, 0] >= 0.5:
        return 1
    else:
        return 0

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
    return w_new

def main():
    FOLDER = r'D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\LogisticRegression'
    X, y = readData(FOLDER, 'ex2data1.txt')
    standardScaling(X)
    w_init = np.zeros((X.shape[1], 1))
    w_result = gradientDescent(X, y, w_init, 0.00001)
    print(predict(X[5, :].reshape(1, -1), w_result))

if __name__ == "__main__":
    main()