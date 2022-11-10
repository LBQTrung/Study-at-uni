import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def readData(folder , filename):
    data = np.loadtxt(os.path.join(folder, filename), delimiter = ',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def normScaling(X):
  for col in range(1, X.shape[1]):
    temp = X[:, col]
    X[:, col] = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
 
def standardScaling(X):
  for col in range(1, X.shape[1]):
    temp = X[:, col]
    X[:, col] = (temp - np.mean(temp)) / (np.std(temp))
        
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
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    # Dự đoán
    y_pred = classifier.predict(X_test)

    # Đánh giá hiệu năng của mô hình
    print("Đánh giá hiệu năng mô hình")
    print("\tAccuracy: ",accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()