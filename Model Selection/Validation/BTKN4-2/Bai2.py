import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

def checkTypeDependentData(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    result = dict(zip(unique, counts))
    print(result)
    return result

def crossValScore(model, X_train, y_train, cv=10, scoring='accuracy'):
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    print('Kết quả huấn luyên 10-fold cv')
    print('\t', scores)
    return scores

def main():
    #Bước 1: Đọc dữ liệu
    FOLDER = r"D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\Model Selection\Validation\BTKN4-2"
    X, y = readData(FOLDER, 'ex2data2.txt')

    #Bước 2: Phân chia train - test theo tỉ lệ 90% - 10%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=15)

    # Bước 3: Chuẩn hóa dữ liệu
    X_train, X_test = featureScaling(X_train, X_test)

    #Số lượng k-fold được xác định tùy thuộc vào số lượng y_train
    result = checkTypeDependentData(y_train)

    #Bước 4: Khởi tạo mô hình hồi quy logistic, với thuật toán tối ưu là liblinear
    #Bước lặp 1500; multi_class = 'auto' để tự phát hiện nhãn lớp nhị phân hay đa nhãn lớp
    classifier = LogisticRegression(solver='liblinear', max_iter=1500, multi_class='auto')

    #Bước 5: Đặc tả 10-fold cv với k = 10, validation size = 20%
    cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=15)

    #Bước 6: Huấn luyện mô hình cv = 10 và độ đo là scoring='accuracy' và in kết quả
    scores = crossValScore(classifier, X_train, y_train, cv=10, scoring='accuracy')

if __name__ == "__main__":
    main()