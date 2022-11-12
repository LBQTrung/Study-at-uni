from sklearn.linear_model import LogisticRegressionCV
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def readData(folder, filename, delimiter= ","):
    D = np.loadtxt(os.path.join(folder, filename), delimiter=delimiter)
    X = D[:, :-1]
    y = D[:, -1]
    return X, y

def featureScaling(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

def main():
    #Bước 1: Đọc dữ liệu
    FOLDER = r"D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\Model Selection\Validation\BTKN4-2"
    X, y = readData(FOLDER, 'ex2data2.txt')

    #Bước 2: Phân chia train - test theo tỉ lệ 70% - 30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

    # Bước 3: Chuẩn hóa dữ liệu
    X_train, X_test = featureScaling(X_train, X_test)

    #Bước 4 Tạo mô hình và huấn luyen 10-foldCV
    model = LogisticRegressionCV(cv=10, random_state=15).fit(X_train, y_train)

    #Bước 5: Dự đoán và đánh giá hiệu năng
    y_pred = model.predict(X_test)

    #Bước 6 đánh giá hiệu năng mô hình
    print('Hiệu năng mô hình acc: ', accuracy_score(y_pred, y_test))

if __name__ == "__main__":
    main()