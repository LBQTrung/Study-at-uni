import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def readData(folder , filename):
  data = np.loadtxt(os.path.join(folder, filename), delimiter = ',')
  X = data[:, :-1]
  y = data[:, -1].reshape(-1, 1)
  one = np.ones((X.shape[0], 1))
  X = np.concatenate((one, X), axis = 1)
  return X, y

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
  w_old = w_init.reshape(-1, 1)
  loss_values = []
  for i in range(n):
    w_new = w_old - alpha * gradient(X, y, w_old)
    j = calculateLoss(X, y, w_new)
    w_old = w_new
    loss_values.append(j)
  return w_new, loss_values

def main():
   FOLDER = r'D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\LinearRegression\Multiple Linear Regression'
   X, y = readData(FOLDER, 'ex1data2.txt')
   w_init = np.zeros((X.shape[1], 1))
   sc_X = StandardScaler()
   sc_y = StandardScaler()
   X[:, 1:] = sc_X.fit_transform(X[:, 1:])
   y = sc_y.fit_transform(y)
   w_optimal, loss_values = gradientDescent(X, y, w_init, alpha=0.001)
   print(f"w optimal: \n{w_optimal}")

if __name__ == "__main__":
  main()