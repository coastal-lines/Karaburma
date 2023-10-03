import numpy as np
import skimage
import skimage as sk
import cv2
from skimage.filters import threshold_mean

from lena.elements.objects.roi_element import RoiElement
from lena.utils.config_manager import ConfigManager
from lena.utils.image_processing import filters_helper, morphological_helpers, contours_helper, harris


class BasicPreprocessing:
