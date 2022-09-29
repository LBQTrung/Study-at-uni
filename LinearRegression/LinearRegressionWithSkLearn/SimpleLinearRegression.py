import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def readData(folder , filename):
  data = np.loadtxt(os.path.join(folder, filename), delimiter = ',')
  X = data[:, 0]
  y = data[:, -1]
  return X.reshape(len(X), 1), y

def findModel(X, y):
    regressor = LinearRegression()
    regressor.fit(X, y)
    return  regressor.intercept_, regressor.coef_[0]

def visualizeModel(X, y, w):
    plt.scatter(X[:, -1], y, color = "red")
    plt.plot(X[:, -1], w[0] + w[1] * X[:, -1], color = "blue")
    plt.title("Simple Linear Regression")
    plt.show()

def main():
   FOLDER = r'D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\LinearRegression\GradientDescent'
   X, y = readData(FOLDER, 'ex1data1.txt')
   w = findModel(X, y)
   print(f'w_0 = {w[0]}, w_1 = {w[1]}')
   visualizeModel(X, y, w)

if __name__ == "__main__":
    main()