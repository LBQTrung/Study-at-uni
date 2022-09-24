import os
import numpy as np
import matplotlib.pyplot as plt

def readData(folder , filename):
  data = np.loadtxt(os.path.join(folder, filename), delimiter = ',')
  X = data[:, 0]
  y = data[:, -1]
  one = np.ones((X.shape[0], 1))
  X = np.concatenate((one, X.reshape(X.shape[0], 1)), axis = 1)
  return X, y

def calculateLoss(X, y, w):
  h = np.dot(X, w)
  m = X.shape[0]
  J = (1/ (2*m)) * np.sum(np.square(h-y))
  return J

def gradientDescent(X, y, w, alpha, n = 1500):
  m = X.shape[0]
  w_optimal = []
  loss_values = []
  for i in range(n):
    w = w - (alpha/m) * (np.dot(X, w) - y ).dot(X)
    j = calculateLoss(X, y, w)
    w_optimal.append(w)
    loss_values.append(j)
  return w_optimal, loss_values

def visualizeModel(X, y, w):
    plt.scatter(X[:, -1], y, color = "red")
    plt.plot(X[:, -1], w[0] + w[1] * X[:, -1], color = "blue")
    plt.title("Simple Linear Regression")
    plt.show()

def main():
   FOLDER = r'D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\LinearRegression\GradientDescent'
   X, y = readData(FOLDER, 'ex1data1.txt')
   w, loss_values = gradientDescent(X, y, [-1, 5], 0.01, 1500)
   w_optimal = w[-1]
   print(f'w_0 = {w_optimal[0]}, w_1 = {w_optimal[1]}')
   visualizeModel(X, y, w_optimal)

if __name__ == "__main__":
    main()