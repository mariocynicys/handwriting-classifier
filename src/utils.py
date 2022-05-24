from processing import norm

import os
import cv2
import glob
import numpy as np
from sklearn.decomposition import PCA


def cmp(gender: str, id: int):
    return os.path.join('cmp23', gender, f'{id:03}.jpg')

def gender(path: str):
    return 1 if 'female' not in path else 0

def pre(image_path: str):
    image_path = os.path.relpath(image_path, 'cmp23')
    return os.path.join('preprocessed', image_path)

def feat(feature: str):
    return os.path.join('features', feature + '.ft')

def imread(image_path: str):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def imwrite(image_path: str, image):
    cv2.imwrite(image_path, image)

def binarize(image):
    return np.where(image > 255 / 2, 1, 0)

def imread_bw(image_path: str):
    return binarize(imread(image_path))

def get_all_images():
    return _ALL_IMAGES

def get_avg_image_shape():
    resize_w, resize_h = (0, 0)
    for image_path in map(pre, _ALL_IMAGES):
        image = imread(image_path)
        resize_h += image.shape[0]
        resize_w += image.shape[1]
    return resize_w / len(_ALL_IMAGES), resize_h / len(_ALL_IMAGES)

def load_labels():
    return _LABELS

def load_feature(feature_name: str):
    return np.loadtxt(feat(feature_name))

def save_feature(feature_name: str, features):
    np.savetxt(feat(feature_name), norm(features))

def pca(xs):
    return PCA().fit_transform(xs)

def combine_features(*features):
    all_features = features[0]
    for feature in features[1:]:
        all_features = np.append(feature, all_features, axis=1)
    return all_features

_ALL_IMAGES = sorted(glob.glob('cmp23/*/*.jpg'))
_LABELS = np.array([gender(image_path) for image_path in _ALL_IMAGES])
