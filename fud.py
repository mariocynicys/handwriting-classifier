import os

sh  = os.listdir('dataset')
import random
random.shuffle(sh)
for path in sh[:1]:
    import cv2

    RESIZE = (500, 700)
    DILATE_SIZE = (20, 20)
    # Load image, grayscale, Gaussian blur, adaptive threshold
    image = cv2.imread(f'dataset/{path}', cv2.IMREAD_GRAYSCALE)
    # ##
    # wid_start = image.shape[0]//40
    # wid_end = image.shape[0] - wid_start

    # hig_start = image.shape[1]//40
    # hig_end = image.shape[1] - hig_start
    # print(image.shape)
    # image = image[ wid_start:wid_end, hig_start:hig_end,]
    # ##
    print(image.shape)
    blur = cv2.GaussianBlur(image, (11,11), 0)
    size = image.shape[1]//2 if image.shape[1]//2 % 2 else image.shape[1]//2 + 1
    thresh = cv2.adaptiveThreshold(blur, 255, # the image is inverted from this line
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, size, 10)

    # Dilate to combine adjacent text contours  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_SIZE)
    dilate = cv2.dilate(thresh, kernel, iterations=3) # consider closing instead.

    open = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    for size in range(15, 1, -1):
        for time in range(2):
            hight, width = (size, size*2) if time % 2 else (size*2, size)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hight, width))
            open = cv2.morphologyEx(open, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_SIZE)
    dilate = cv2.dilate(open, kernel, iterations=8)
    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    ROI_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 10000:
            x,y,w,h = cv2.boundingRect(c)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
            # ROI = image[y:y+h, x:x+w]
            # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
            # ROI_number += 1

    thresh = cv2.resize(thresh, RESIZE, thresh)
    dilate = cv2.resize(dilate, RESIZE, dilate)
    open = cv2.resize(open, RESIZE, open)
    image = cv2.resize(image, RESIZE, image)

    cv2.imshow('thresh', thresh)
    cv2.imshow('dilate', dilate)
    cv2.imshow('open', open)
    cv2.imshow('image', image)
    cv2.waitKey(0)
