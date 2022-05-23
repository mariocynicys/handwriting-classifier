import cv2 as cv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

img1 = cv.imread('dataset/female_cmp_1.jpg', cv.IMREAD_GRAYSCALE)
img2 = mpimage.imread('dataset/female_cmp_1.jpg')
mat1 = np.array(img1)
mat2 = np.array(img2)
print(mat1.shape, mat2.shape)
# print(mat1 == mat2)
# np.savetxt('mat1', mat1)
# np.savetxt('mat2', mat2)
# cv.imshow('title', img1)
# cv.waitKey(1)
# plt.imshow(img1)
# plt.show()

image = cv.imread('dataset/female_cmp_8.jpg', cv.IMREAD_GRAYSCALE)
print(image.shape)
plt.imshow(image)
#plt.show()

image_paths = ['dataset/female_cmp_1.jpg']
fig = plt.figure(figsize=(9, 9))
for image_path in image_paths:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Skip this image if it's corrupt, like F87.jpg
    if img is None:
        continue
    #fig.add_subplot(9, 9, 1)
    #plt.imshow(img)

    blr = cv2.GaussianBlur(img, (9,9), 0)
    #fig.add_subplot(4, 4, 2)
    #plt.imshow(blr)

    the = cv2.adaptiveThreshold(blr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 30)
    #print(the)
    #fig.add_subplot(4, 4, 3)
    plt.imshow(the)
RESIZE = (500, 700)

thresh = cv2.resize(the, RESIZE, the)
plt.imshow(the)
plt.show()
cv2.imshow('so', thresh)
cv2.waitKey(0)
# cv2.imshow('', img)
# cv2.waitKey(1)
# while True:
#   import time
#   time.sleep(1)
