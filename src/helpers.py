import cv2
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def preview(images: list):
    fig = plt.figure(figsize=(20, 20))
    for i, image_path in enumerate(images):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        fig.add_subplot(4, 4, i + 1)
        plt.imshow(image, cmap='gray')
    plt.show()

def plt_test(xs, ys):
    males = xs[ys[:,0] == 1]
    females = xs[ys[:,0] == 0]
    for f1 in range(0, len(males[0])):
        print(f1)
        plt.plot(males[:, f1], ys[ys[:,0] == 1], 'bo')
        plt.plot(females[:, f1], ys[ys[:,0] == 0], 'rx')
        plt.show()

def svm_test(xs, ys, times=100, test_size=0.2, **kwargs):
    tr_ac, ts_ac, mal, fem = 0, 0, 0, 0
    for _ in range(times):
        X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=test_size)
        clf = SVC(**kwargs)
        clf.fit(X_train, y_train)
        tr_ac += clf.score(X_train, y_train)
        ts_ac += clf.score(X_test, y_test)
        mal += np.sum(clf.predict(xs) == 1) / len(xs)
        fem += np.sum(clf.predict(xs) == 0) / len(xs)
    print(f"""
          male percentage = {mal * 100 / times:.2f}%
          female percentage = {fem * 100 / times:.2f}%
          train accuracy = {tr_ac * 100 / times:.2f}%
          test accuracy = {ts_ac * 100 / times:.2f}%
          """)

def ann_test(xs, ys, times=100, test_size=0.2, **kwargs):
    mlp_kwargs = {
        'solver': 'lbfgs',
        'hidden_layer_sizes': (5, 2),
        'activation': 'identity',
        'max_iter': 10000,
    }
    mlp_kwargs.update(kwargs)
    tr_ac, ts_ac, mal, fem = 0, 0, 0, 0
    for _ in range(times):
        X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=test_size)
        clf = MLPClassifier(**kwargs)
        clf.fit(X_train, y_train)
        tr_ac += clf.score(X_train, y_train)
        ts_ac += clf.score(X_test, y_test)
        mal += np.sum(clf.predict(xs) == 1) / len(xs)
        fem += np.sum(clf.predict(xs) == 0) / len(xs)
    print(f"""
          male percentage = {mal * 100 / times:.2f}%
          female percentage = {fem * 100 / times:.2f}%
          train accuracy = {tr_ac * 100 / times:.2f}%
          test accuracy = {ts_ac * 100 / times:.2f}%
          """)
