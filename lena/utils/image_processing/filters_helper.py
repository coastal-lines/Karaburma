import inspect

import cv2
from decimal import Decimal
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage.filters import threshold_local, threshold_mean, threshold_minimum, threshold_otsu, threshold_li, \
    threshold_isodata, threshold_triangle, threshold_yen
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.draw import rectangle
from skimage.draw import rectangle_perimeter
import pickle
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, confusion_matrix
from imutils.object_detection import non_max_suppression
from sklearn.metrics import accuracy_score
from typing import overload
from loguru import logger
import traceback

from lena.utils.logging_manager import LoggingManager


def gamma_correction(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv.LUT(src, table)

def levels_correction(img, *args):
    if(len(args) == 5):
        input_min, input_max, output_min, output_max, gamma = args
    elif(len(args[0]) == 5):
        input_min, input_max, output_min, output_max, gamma = args[0]
    else:
        raise ValueError(LoggingManager().log_number_arguments_error(5, len(args)))

    inBlack = np.array([input_min], dtype=np.float32)
    inWhite = np.array([input_max], dtype=np.float32)
    inGamma = np.array([gamma], dtype=np.float32)
    outBlack = np.array([output_min], dtype=np.float32)
    outWhite = np.array([output_max], dtype=np.float32)

    img = np.clip((img - inBlack) / (inWhite - inBlack), 0, 255)
    img = (img ** (1 / inGamma)) * (outWhite - outBlack) + outBlack
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img

def bw_gamma_correction2(img):

    inBlack = np.array([14], dtype=np.float32)
    inWhite = np.array([114], dtype=np.float32)
    inGamma = np.array([0.25], dtype=np.float32)
    outBlack = np.array([0], dtype=np.float32)
    outWhite = np.array([255], dtype=np.float32)

    img = np.clip((img - inBlack) / (inWhite - inBlack), 0, 255)
    img = (img ** (1 / inGamma)) * (outWhite - outBlack) + outBlack
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img

def levels(img, inGamma, inBlack=0, inWhite=255, outBlack=0, outWhite=255):

    inBlack = np.array([inBlack], dtype=np.float32)
    inWhite = np.array([inWhite], dtype=np.float32)
    inGamma = np.array([inGamma], dtype=np.float32)
    outBlack = np.array([outBlack], dtype=np.float32)
    outWhite = np.array([outWhite], dtype=np.float32)

    img = np.clip((img - inBlack) / (inWhite - inBlack), 0, 255)
    img = (img ** (1 / inGamma)) * (outWhite - outBlack) + outBlack
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img

def blur(img, kernel):
    return cv2.GaussianBlur(img, kernel, 0)

def sharp(img, type):

    middle = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    strong = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    if(type == "middle"):
        img = cv2.filter2D(img, -1, middle)

    if(type == "strong"):
        img = cv2.filter2D(img, -1, strong)

    return img

def threshold(img, min, max):
    ret, th = cv2.threshold(img, min, max, cv2.THRESH_BINARY)
    return ret, th

def adaptive_threshold(img, max_value, block_size, constant):
    th = cv2.adaptiveThreshold(img ,max_value ,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY ,block_size ,constant)
    return th

def BlendedThreshold(img):
    ret, th1 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
    blended = cv2.addWeighted(src1=th1, alpha=0.6, src2=th2, beta=0.4, gamma=0)
    return blended

def try_threshold(grayscale_image, block_size_min=3, block_size_max=11):
    #local_thresh = threshold_local(grayscale_image, block_size=block_size, offset=3)

    binary_local = []

    for i in range(block_size_min, block_size_max, 2):
        try:
            binary_local.append(skimage.img_as_ubyte(grayscale_image > threshold_local(grayscale_image, block_size=i, offset=3)))
        except:
            binary_local.append(np.zeros((255,255), dtype=int))

    try:
        mean_binary = skimage.img_as_ubyte(grayscale_image.copy() > threshold_mean(grayscale_image))
    except:
        mean_binary = np.zeros((255,255), dtype=int)

    try:
        minimum_binary = skimage.img_as_ubyte(grayscale_image.copy() > threshold_minimum(grayscale_image))
    except:
        minimum_binary = np.zeros((255,255), dtype=int)

    try:
        otsu_binary = skimage.img_as_ubyte(grayscale_image.copy() > threshold_otsu(grayscale_image))
    except:
        otsu_binary= np.zeros((255, 255), dtype=int)

    try:
        li_binary = skimage.img_as_ubyte(grayscale_image.copy() > threshold_li(grayscale_image))
    except:
        li_binary = np.zeros((255, 255), dtype=int)

    try:
        isodata_binary = skimage.img_as_ubyte(grayscale_image.copy() > threshold_isodata(grayscale_image))
    except:
        isodata_binary = np.zeros((255, 255), dtype=int)

    try:
        triangle_binary = skimage.img_as_ubyte(grayscale_image.copy() > threshold_triangle(grayscale_image))
    except:
        triangle_binary = np.zeros((255, 255), dtype=int)

    try:
        yen_binary = skimage.img_as_ubyte(grayscale_image.copy() > threshold_yen(grayscale_image))
    except:
        yen_binary = np.zeros((255, 255), dtype=int)

    '''
    f, axarr = plt.subplots(3, 3)

    axarr[0, 0].imshow(isodata_binary, cmap='gray')
    axarr[0, 0].set_title("Isodata")

    axarr[0, 1].imshow(minimum_binary, cmap='gray')
    axarr[0, 1].set_title("Minimum")

    axarr[0, 2].imshow(binary_local[0], cmap='gray')
    axarr[0, 2].set_title("Binary")

    axarr[1, 0].imshow(mean_binary, cmap='gray')
    axarr[1, 0].set_title("Mean")

    axarr[1, 1].imshow(otsu_binary, cmap='gray')
    axarr[1, 1].set_title("Otsu")

    axarr[1, 2].imshow(grayscale_image, cmap='gray')
    axarr[1, 2].set_title("Image")

    axarr[2, 0].imshow(triangle_binary, cmap='gray')
    axarr[2, 0].set_title("Triangle")

    axarr[2, 1].imshow(yen_binary, cmap='gray')
    axarr[2, 1].set_title("Yen")

    axarr[2, 2].imshow(li_binary, cmap='gray')
    axarr[2, 2].set_title("li_binary")

    plt.show()
    '''

    return binary_local, mean_binary, minimum_binary, otsu_binary, li_binary, isodata_binary, triangle_binary, yen_binary