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
    def __init__(self, model_for_common_elements, model_for_listbox, common_element_features,
                 scroll_buttons_patterns, shift_threshold_for_scrolls):
        self.model_for_common_elements = model_for_common_elements
        self.__model_for_listbox = model_for_listbox
        self.__scroll_element_features = ScrollElementDetectionsFeatures(common_element_features,
                                                                         scroll_buttons_patterns,
                                                                         shift_threshold_for_scrolls)
