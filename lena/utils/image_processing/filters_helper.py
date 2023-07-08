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