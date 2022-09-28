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

def gradient(X, y, w):
  m = X.shape[0]
  return  (1/m) * (np.dot(X, w) - y ).dot(X)

def gradientDescentMomentum(X, y, w_init, alpha = 0.01, gamma = 0.9, n = 1500):
  w = [w_init]
  v_old = np.zeros_like(w_init)
  for i in range(n):
    v_new = gamma*v_old + alpha * gradient(X, y, w[-1])
    w_new = w[-1] - v_new
    if calculateLoss(X, y, w_new) < 0.001:
      print(f"Get the result after {i+1} iterations")
      break
    w.append(w_new)
    v_old = v_new
  return w

def visualizeModel(X, y, w):
    plt.scatter(X[:, -1], y, color = "green")
    plt.plot(X[:, -1], w[0] + w[1] * X[:, -1], color = "blue")
    plt.title("SLR with GD momentum")
    plt.show()

def main():
   FOLDER = r'D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\LinearRegression\GradientDescent'
   X, y = readData(FOLDER, 'ex1data1.txt')
   omega = gradientDescentMomentum(X, y, [0, 0])
   [w_0, w_1] = omega[-1]
   print(f"w_0 = {w_0}, w_1 = {w_1}")
   visualizeModel(X, y, omega[-1])

if __name__ == "__main__":
    main()