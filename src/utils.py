from processing import norm

import os
import cv2
import glob
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def hint(*args, **kwargs):
    """Just like print(), but its output is erased with the next hint()/print() call."""
    print(*args, **kwargs, end=100 * ' ' + '\r')

def loading(done: int, outof: int, loading_char='█'):
    percentage = done / outof * 100
    filled, left = int(percentage), 100 - int(percentage)
    hint(f"|{filled * loading_char}{left * ' '}| [{percentage:.2f}%]")

def cmp(gender: str, id: int):
    return os.path.join('cmp23', gender + 's', f'{id:03}.jpg')

def gender(path: str):
    return 1 if 'females' not in path else 0

def pre(image_path: str):
    image_path = os.path.relpath(image_path, 'cmp23')
    return os.path.join('preprocessed', image_path)

def feat(feature: str):
    return os.path.join('features', feature + '.ft')

def meta(feature: str, meta_info: str):
    return os.path.join('meta', f'{feature}-{meta_info}.inf')

def imread(image_path: str, apply_tresh=True):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if apply_tresh:
        image = cv2.threshold(image, 255 / 2, 255, cv2.THRESH_BINARY)[1]
    return image

def imwrite(image_path: str, image):
    cv2.imwrite(image_path, image)

def binarize(image):
    return np.where(image > 255 / 2, 1, 0)

def imread_np(image_path: str):
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
    features, cols_to_keep, scaler = norm(features)
    np.savetxt(feat(feature_name), features)
    np.savetxt(meta(feature_name, 'cols_to_keep'), cols_to_keep)
    pickle.dump(scaler, open(meta(feature_name, 'scaler'), 'wb'))

def pca(xs, feature_name):
    pca = PCA()
    pca.fit(xs)
    pickle.dump(pca, open(meta(feature_name, 'pca'), 'wb'))
    return pca.transform(xs)

def combine_features(*features):
    all_features = features[0]
    for feature in features[1:]:
        all_features = np.append(feature, all_features, axis=1)
    return all_features

def split(xs, ys, test_size=0.2):
    return train_test_split(xs, ys, test_size=test_size)

_ALL_IMAGES = sorted(glob.glob('cmp23/*/*.jpg'))
_LABELS = np.array([gender(image_path) for image_path in _ALL_IMAGES])
