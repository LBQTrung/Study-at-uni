import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def readData(folder , filename):
    data = np.loadtxt(os.path.join(folder, filename), delimiter = ',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def visualizeModel(X, y):
    from matplotlib.colors import ListedColormap
    X_set, y_set = sc.inverse_transform(X), y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                        np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
    plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Logistic Regression (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    FOLDER = r'D:\Học Kì 1 - Năm 2\Học máy 1\Baitap\LogisticRegression'
    X, y = readData(FOLDER, 'ex2data1.txt')
    sc = StandardScaler()
    X = sc.fit_transform(X)
    print(X[3, :])
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X, y)
    print(classifier.predict(sc.transform([[60.18259938620976,86.30855209546826]])))
    visualizeModel(X, y)