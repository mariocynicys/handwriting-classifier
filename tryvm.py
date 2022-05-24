#############################################################
############################LIL DEV ENV######################
#############################################################

import collections
import cv2
import sys
import time
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.measure
from sklearn import svm
from skimage.filters import gabor
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def main(show):
    if not show:
        cv2.imshow = lambda *args: None

    CMP23_DIR = 'cmp23/'
    ICDAR_DIR = 'icdar/'

    PREPROCESSING_GAUSSIAN_BLUR_KERNEL_SIZE = (9, 9)
    PREPROCESSING_DILATION_ITERATIONS = 8
    # We need more dilation into the vertical axis
    PREPROCESSING_DILATION_SIZE = (15, 20)
    PREPROCESSING_THRESH_BLOCK_SIZE = 101
    PREPROCESSING_THRESH_C = 30
    X_CUT_PERCENT = 0.1
    Y_CUT_PERCENT = 0.5
    RESIZE = (500, 700)
    RESIZE_CROPPED = (5000, 3000)


    def cmp(gender: str, id: int):
        return CMP23_DIR + gender + '/' + str(id) + '.jpg'


    def icd(gender: str, id: int):
        return ICDAR_DIR + gender + '/' + str(id) + '.jpg'


    image_paths = [f'dataset/{sys.argv[1]}.jpg']

    for image_path in image_paths:
        print(f'Working on image {image_path}')
        # start_time = time.time()
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(image.shape)

        if image is None:
            continue
        height, width = image.shape
        if X_CUT_PERCENT:
            start_x, end_x = int(width / (100/X_CUT_PERCENT)
                                 ), width - int(width / (100/X_CUT_PERCENT))
            image = image[:, start_x:end_x]
        if Y_CUT_PERCENT:
            start_y, end_y = int(height / (100/Y_CUT_PERCENT)
                                 ), height - int(height / (100/Y_CUT_PERCENT))
            image = image[start_y:end_y, :]
        # cv2.imshow('original', cv2.resize(image, RESIZE))

        # read_time = time.time()
        img = cv2.GaussianBlur(image, PREPROCESSING_GAUSSIAN_BLUR_KERNEL_SIZE, 0)
        # blur_time = time.time()

        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, PREPROCESSING_THRESH_BLOCK_SIZE, PREPROCESSING_THRESH_C)

        # image = img
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, PREPROCESSING_THRESH_BLOCK_SIZE, PREPROCESSING_THRESH_C)

        # thresh_time = time.time()
        # cv2.imshow('thresh', cv2.resize(img, RESIZE))

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, PREPROCESSING_DILATION_SIZE)
        img = cv2.dilate(img, kernel, iterations=PREPROCESSING_DILATION_ITERATIONS)
        # dilation_time = time.time()
        # cv2.imshow('dilation', cv2.resize(img, RESIZE))

        contours = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # contour_time = time.time()

        #biggest_contour = functools.reduce(lambda c1, c2: c1 if cv2.contourArea(c1) > cv2.contourArea(c2) else c2, contours)
        biggest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        x, y, w, h = cv2.boundingRect(biggest_contour)
        # cv2.imshow('Text', cv2.resize(cv2.rectangle(image, (x, y), (x + w, y + h), 255, 3), RESIZE))
        image = image[y:y + h, x:x + w]

        image = cv2.threshold(image, 255 / 2, 255, cv2.THRESH_BINARY)[1]

        # cv2.imshow('final', cv2.resize(image, RESIZE_CROPPED))
        # write_time = time.time()


    # image = np.where(image > 255 / 2, 1, 0)

    cv2.waitKey(0)

main(show=True)