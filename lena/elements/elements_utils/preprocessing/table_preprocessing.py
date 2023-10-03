import math
import cv2
import skimage
import numpy as np
import pandas as pd
from PIL import Image as PIL_Image
from skimage.filters import threshold_otsu
from skimage.util import img_as_ubyte
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from lena.elements.objects.roi_element import RoiElement
from lena.elements.objects.screenshot_element import ImageSourceObject
from lena.utils.config_manager import ConfigManager
from lena.utils.image_processing import filters_helper, contours_helper, morphological_helpers
from lena.utils import data_normalization, general_helpers


class TablePreprocessing:
