#from model import ANN, RF, SVM
from utils import split, loading

import cv2
import numpy as np
import matplotlib.pyplot as plt


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

def c_test(xs, ys, clf, count=100, test_size=0.2):
    tr_ac, ts_ac, mal, fem = 0, 0, 0, 0
    loading(0, count)
    for i in range(count):
        X_train, X_test, y_train, y_test = split(xs, ys, test_size)
        clf.fit(X_train, y_train)
        tr_ac += clf.score(X_train, y_train)
        ts_ac += clf.score(X_test, y_test)
        mal += np.sum(clf.predict(xs) == 1) / len(xs)
        fem += np.sum(clf.predict(xs) == 0) / len(xs)
        loading(i + 1, count)
    print(f"""
          male percentage = {mal * 100 / count:.2f}%
          female percentage = {fem * 100 / count:.2f}%
          train accuracy = {tr_ac * 100 / count:.2f}%
          test accuracy = {ts_ac * 100 / count:.2f}%
          """)

# def svm_test(xs, ys, count=100, test_size=0.2, **kwargs):
#     return c_test(xs, ys, SVM(**kwargs), count, test_size)

# def ann_test(xs, ys, count=100, test_size=0.2, **kwargs):
#     return c_test(xs, ys, ANN(**kwargs), count, test_size)

# def rfc_test(xs, ys, count=100, test_size=0.2, **kwargs):
#     return c_test(xs, ys, RF(**kwargs), count, test_size)
