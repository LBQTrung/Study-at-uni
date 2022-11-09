import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

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
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    # Dự đoán
    y_pred = regressor.predict(X_test)

    # Đánh giá hiệu năng của mô hình
    print("Đánh giá hiệu năng mô hình")
    print("\tMSE: ",mean_squared_error(y_test, y_pred))
    print("\tRMSE: ",mean_squared_error(y_test, y_pred) ** (1/2))
if __name__ == "__main__":
    main()