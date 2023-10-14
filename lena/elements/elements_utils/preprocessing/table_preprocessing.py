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
    def __init__(self):
        self.image_source = None

    @property
    def image_source(self):
        return self.__image_source

    @image_source.setter
    def image_source(self, screenshot: ImageSourceObject):
        print("Setting value")
        self.__image_source = screenshot

    def __resize_source_image_and_apply_filter(self, roi):

        grey_ = filters_helper.convert_to_grayscale(roi)
        imnp = np.array(grey_) / 255
        gamma = math.log(imnp.mean()) / math.log(0.1)
        new = ((imnp ** (1 / gamma)) * 255).astype(np.uint8)
        new = np.array(PIL_Image.fromarray(new).resize(ConfigManager().config.elements_parameters.table["fixed_size_for_preprocessing"], PIL_Image.BICUBIC))
        er = img_as_ubyte(morphological_helpers.erosion(new))
        er = morphological_helpers.erosion(er)
        dl = morphological_helpers.dilation(er)

        return dl
