import cv2
import numpy as np


def erosion(img):
    kernel = np.ones((3, 3), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)

    return img_erosion

def dilation(img):
    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)

    return img_dilation

def close(img, kernelSize = (3,3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return closing