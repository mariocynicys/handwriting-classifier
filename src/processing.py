import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler


PREPROCESSING_GAUSSIAN_BLUR_KERNEL_SIZE = (9, 9)
PREPROCESSING_DILATION_ITERATIONS = 8
PREPROCESSING_DILATION_SIZE = (15, 20) # We need more dilation into the vertical axis.
PREPROCESSING_THRESH_BLOCK_SIZE = 101 # 101 is quick enough, yet robust.
PREPROCESSING_THRESH_C = 30 # Note that we need a big C subtractor when we use a big block size.
X_CUT_PERCENT = 0.5
Y_CUT_PERCENT = 1


def prune_useless_feature_cols(features):
    same_cols = features[0] == features[1]
    for feature_vec in features:
        same_cols &= features[0] == feature_vec
    if np.any(same_cols):
        print(f'The following {np.sum(same_cols)} features were removed because they are not discriminative:')
        features_to_remove = np.where(same_cols, [x + 1 for x in range(len(same_cols))], -1)
        print(features_to_remove[features_to_remove != -1])
    return features[:, ~same_cols], ~same_cols

def norm(features):
    features, cols_to_keep = prune_useless_feature_cols(np.array(features))
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler.transform(features), cols_to_keep, scaler

def preprocess(image):
    # Cut some percentage of the images' edges. They are usually noisy.
    height, width = image.shape
    if X_CUT_PERCENT:
        start_x, end_x = int(width / (100 / X_CUT_PERCENT)), width - int(width / (100 / X_CUT_PERCENT))
        image = image[:, start_x:end_x]
    if Y_CUT_PERCENT:
        start_y, end_y = int(height / (100 / Y_CUT_PERCENT)), height - int(height / (100 / Y_CUT_PERCENT))
        image = image[start_y:end_y, :]
    # Blur the image to decrease sharpness. Good for thresholding.
    img = cv2.GaussianBlur(image, PREPROCESSING_GAUSSIAN_BLUR_KERNEL_SIZE, 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                PREPROCESSING_THRESH_BLOCK_SIZE, PREPROCESSING_THRESH_C)
    # Dilate the image to create a contour out of the handwritten text.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, PREPROCESSING_DILATION_SIZE)
    img = cv2.dilate(img, kernel, iterations=PREPROCESSING_DILATION_ITERATIONS)
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # Get the biggest contour that contains the text.
    biggest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # Get the original image in black and white. Note that this version is not blurred.
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                  PREPROCESSING_THRESH_BLOCK_SIZE, PREPROCESSING_THRESH_C)
    image = cv2.threshold(image, 255 / 2, 255, cv2.THRESH_BINARY)[1]
    # Crop only the text part.
    x, y, w, h = cv2.boundingRect(biggest_contour)
    return image[y:y + h, x:x + w]
