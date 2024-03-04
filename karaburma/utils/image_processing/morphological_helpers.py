import cv2
import numpy as np


def erosion(img, kernelSize=(3,3)):
    kernel = np.ones(kernelSize, np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    return img_erosion

def dilation(img, kernelSize=(3,3)):
    kernel = np.ones(kernelSize, np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    return img_dilation

def close(img, kernelSize=(3,3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing