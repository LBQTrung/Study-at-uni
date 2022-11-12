import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

def girdSeachCV(model, X_train, y_train, specified_parameters: list):
    parameters = {'C':[1, 10, 20, 50]}
    cv = GridSearchCV(model, parameters, cv=10)
    cv.fit(X_train, y_train)
    return cv

def main():
    #Bước 1: Đọc dữ liệu
    FOLDER = r"D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\Model Selection\Validation\BTKN4-2"
    X, y = readData(FOLDER, 'ex2data2.txt')
    print(X.shape[1])
    #Bước 2: Phân chia train - test theo tỉ lệ 70% - 30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

    # Bước 3: Chuẩn hóa dữ liệu
    X_train, X_test = featureScaling(X_train, X_test)

    #Bước 4: Khởi tạo mô hình hồi quy logistic, với thuật toán tối ưu là liblinear
    #Bước lặp 1500; multi_class = 'auto' để tự phát hiện nhãn lớp nhị phân hay đa nhãn lớp
    model = LogisticRegression(solver='liblinear', max_iter=1500, multi_class='auto')

    #Bước 5: Đặc tả 10-fold cv với k = 10 và huấn luyện mô hình
    cv = girdSeachCV(model, X_train, y_train, specified_parameters = [1, 10, 20, 50])

    #Bước 7: Thông báo kết quả tối ưu
    print('Kết quả huấn luyên 10-fold cv')
    print('\t', cv.best_params_)

    #Bước 8: Tạo mô hình LogisticRegression với best param
    #model.set_params(**cv.best_params_)
    #model.fit(X_train, y_train)
    y_pred = cv.predict(X_test)

    #Bước 9 đánh giá hiệu năng mô hình
    print('Hiệu năng mô hình acc: ', accuracy_score(y_pred, y_test))

if __name__ == "__main__":
    main()