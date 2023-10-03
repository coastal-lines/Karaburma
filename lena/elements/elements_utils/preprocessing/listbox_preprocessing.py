import cv2
import numpy as np
from PIL import Image as PIL_Image
from sklearn.preprocessing import MinMaxScaler

from lena.elements.features.scroll_element_features import ScrollElementDetectionsFeatures
from lena.elements.objects.roi_element import RoiElement
from lena.elements.objects.element import Element
from lena.elements.objects.listbox_element import ListBoxElement
from lena.utils.config_manager import ConfigManager
from lena.utils.image_processing import filters_helper, morphological_helpers


class ListboxPreprocessing:
