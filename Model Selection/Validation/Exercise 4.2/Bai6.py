import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def readData(folder , filename):
    data = np.loadtxt(os.path.join(folder, filename), delimiter = ',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def featureScalingSplit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    return X_train, X_test, y_train, y_test

def kFoldCrossValiation(X_train, y_train):
    kf = KFold(n_splits=10)
    step = 0
    valitdate_models = {}
    for train2_index, val_index in kf.split(X = X_train,y = y_train):
        step = step + 1
        classifier = LogisticRegression()
        X_train2, X_val = X_train[train2_index], X_train[val_index]
        y_train2, y_val = y_train[train2_index], y_train[val_index]
        classifier.fit(X_train2, y_train2)
        y_pred = classifier.predict(X_val)
        valitdate_models[accuracy_score(y_val, y_pred)] =  classifier
    min_error = min(list(valitdate_models.keys()))
    return valitdate_models[min_error]

def validateTestSet(X_test, y_test, classifier):
    print("Đánh giá hiệu năng mô hình trên tập dữ liệu test:")
    y_pred = classifier.predict(X_test)
    print("\tAccuracy: ", accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred)
        
def main():
    FOLDER = r'D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\Model Selection\Validation\Exercise 4.2'
    X, y = readData(FOLDER, 'ex2data1.txt')
    X_train, X_test, y_train, y_test = featureScalingSplit(X, y)
    classifier = kFoldCrossValiation(X_train, y_train)
    validateTestSet(X_test, y_test, classifier)
    
if __name__ == "__main__":
    main()