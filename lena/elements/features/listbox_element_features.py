from lena.elements.elements_utils.displacement_features import DisplacementFeatures
from lena.elements.elements_utils.preprocessing.listbox_preprocessing import ListboxPreprocessing
from lena.elements.elements_utils.scroll_actions_features import ScrollActionsFeatures
from lena.elements.elements_utils.stitching_features import StitchingFeatures
from lena.elements.objects.listbox_element import ListBoxElement
from lena.utils.config_manager import ConfigManager
from lena.utils.image_processing import augmentation, morphological_helpers
import time
import cv2
import numpy as np
import pyautogui
from skimage.metrics import structural_similarity as ssim
from lena.data.constants.enums.scroll_direction_enum import ScrollDirectionEnum
from lena.elements.objects.element import Element
from lena.elements.objects.roi_element import RoiElement
from lena.utils import general_helpers
from lena.utils.image_processing import ocr_helper, filters_helper
from lena.elements.features.scroll_element_features import ScrollElementDetectionsFeatures
from lena.utils.objects_tracking.displacement import OcrVerticalDisplacement


class ListboxElementFeatures(ListboxPreprocessing):
    def __init__(self, model_for_common_elements, model_for_listbox, common_element_features, scroll_buttons_patterns, shift_threshold_for_scrolls):
        super().__init__(model_for_common_elements, model_for_listbox, common_element_features, scroll_buttons_patterns, shift_threshold_for_scrolls)

        self.model_for_common_elements = model_for_common_elements
        self.__model_for_listbox = model_for_listbox

        self.__image_source = None

        #self.__scroll_element_features = ScrollElementDetectionsFeatures(common_element_features, scroll_buttons_patterns, shift_threshold_for_scrolls)


    def __extend_stiched_listbox_roi(self, temp_stiched_listbox):
        #исключаем скроллбар
        roi = temp_stiched_listbox[:, 0:temp_stiched_listbox.shape[1] - (temp_stiched_listbox.shape[1] // 5)]

        left = ConfigManager().config.elements_parameters.listbox.additional_borders["left"]
        right = ConfigManager().config.elements_parameters.listbox.additional_borders["right"]
        top = ConfigManager().config.elements_parameters.listbox.additional_borders["top"]
        bottom = ConfigManager().config.elements_parameters.listbox.additional_borders["bottom"]
        colour = ConfigManager().config.elements_parameters.listbox.additional_borders["colour"]

        roi = augmentation.extend_grayscale_roi(roi, left, right, top, bottom, colour)
        roi = roi.astype(np.uint8)

        return roi
