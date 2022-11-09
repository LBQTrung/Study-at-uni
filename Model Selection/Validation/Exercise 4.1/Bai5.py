import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression

def readData(folder , filename):
    data = np.loadtxt(os.path.join(folder, filename), delimiter = ',')
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    one = np.ones((X.shape[0], 1))
    X = np.concatenate((one, X), axis = 1)
    return X, y

def featureScaling(X_train, X_test, y_train, y_test):
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train[:, 1:] = sc_X.fit_transform(X_train[:, 1:])
    X_test[:, 1:] = sc_X.transform(X_test[:, 1:])
    y_train = sc_y.fit_transform(y_train)
    y_test = sc_y.transform(y_test)
    return X_train, X_test, y_train, y_test
        
def main():
    FOLDER = r'D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\Model Selection\Validation\Exercise 4.1'
    # Đọc dữ liệu
    X, y = readData(FOLDER, 'ex1data2.txt')
    # Chia tập dữ liệu thành training set và test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state = 5)

    # Chuẩn hóa dữ liệu
    X_train, X_test, y_train, y_test = featureScaling(X_train, X_test, y_train, y_test)

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