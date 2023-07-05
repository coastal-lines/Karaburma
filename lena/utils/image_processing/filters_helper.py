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


def GammaCorrection(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv.LUT(src, table)

'''
def LevelsCorrection(img, input_min, input_max, output_min, output_max, gamma):

    inBlack = np.array([input_min], dtype=np.float32)
    inWhite = np.array([input_max], dtype=np.float32)
    inGamma = np.array([gamma], dtype=np.float32)
    outBlack = np.array([output_min], dtype=np.float32)
    outWhite = np.array([output_max], dtype=np.float32)

    img = np.clip((img - inBlack) / (inWhite - inBlack), 0, 255)
    img = (img ** (1 / inGamma)) * (outWhite - outBlack) + outBlack
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img
'''

def LevelsCorrection(img, *args):
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