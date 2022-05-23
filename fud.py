import os
from tkinter.dialog import DIALOG_ICON

path = 'male_cmp_89.jpg'
import cv2

ITER = 10
RESIZE = (500, 700)
DILATE_SIZE = (15, 20) # we don't need so much width dilation as hight.
# Load image, grayscale, Gaussian blur, adaptive threshold
image = cv2.imread(f'dataset/{path}', cv2.IMREAD_GRAYSCALE)
print(image.shape)
blur = cv2.GaussianBlur(image, (9,9), 0)
width = image.shape[1] * 2
size = width//2 if width//2 % 2 else width//2 + 1
thresh = cv2.adaptiveThreshold(blur, 255, # the image is inverted from this line
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV, size,
                            # we need a great subtractor since the size if the images width.
                            30)

# Dilate to combine adjacent text contours  
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_SIZE)
dilate = cv2.dilate(thresh, kernel, iterations=3) # consider closing instead.

open = thresh#cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# for size in range(15, 1, -1):
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
#     open = cv2.morphologyEx(open, cv2.MORPH_OPEN, kernel)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_SIZE)
dilate = cv2.dilate(open, kernel, iterations=ITER)
# Find contours, highlight text areas, and extract ROIs
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

ROI_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > 10000:
        x,y,w,h = cv2.boundingRect(c)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
        # cv2.drawContours(image, [c], -1, (0,255,0), 3)
        # ROI = image[y:y+h, x:x+w]
        # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        # ROI_number += 1

thresh = cv2.resize(thresh, RESIZE, thresh)
dilate = cv2.resize(dilate, RESIZE, dilate)
open = cv2.resize(open, RESIZE, open)
image = cv2.resize(image, RESIZE, image)

cv2.imshow('thresh', thresh)
cv2.imshow('dilate', dilate)
#cv2.imshow('open', open)
cv2.imshow('image', image)
cv2.waitKey(0)


