import os
import numpy as np
from sklearn.linear_model import LinearRegression

def readData(folder , filename):
  data = np.loadtxt(os.path.join(folder, filename), delimiter = ',')
  X = data[:, :-1]
  y = data[:, -1].reshape(-1, 1)
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

def findModel(X, y):
    regressor = LinearRegression()
    regressor.fit(X, y)
    return regressor

def main():
   FOLDER = r'D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\LinearRegression\Multiple Linear Regression'
   X, y = readData(FOLDER, 'ex1data2.txt')
   print(X.shape)
   normScaling(X, y)
   w = findModel(X, y)
   for i in range(42):
    print(w.predict(X[i, :].reshape(1, -1))[0, 0], y[i, 0])

if __name__ == "__main__":
    main()