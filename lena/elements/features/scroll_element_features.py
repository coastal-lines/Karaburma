import cv2
import numpy as np
import pyautogui

from lena.data.constants.enums.element_types_enum import ElementTypesEnum
from lena.data.constants.enums.scroll_direction_enum import ScrollDirectionEnum
from lena.utils import general_helpers, files_helper
from lena.elements.objects.element import Element
from lena.elements.objects.scroll_element import ScrollElement
from lena.utils.image_processing import filters_helper, geometric_transformations, contours_helper, pattern_matching
from lena.elements.objects.roi_element import RoiElement

class TempScrollAttributes:
    def __init__(self, sorted_rectangles, index):
        self.current_x, self.current_y, self.current_w, self.current_h = contours_helper.extract_xywh(sorted_rectangles, index)
        self.current_center_x = contours_helper.get_x_center(self.current_x, self.current_w)
        self.current_center_y = contours_helper.get_y_center(self.current_y, self.current_h)
        self.next_x, self.next_y, self.next_w, self.next_h = contours_helper.extract_xywh(sorted_rectangles, index + 1)
        self.next_center_x = contours_helper.get_x_center(self.next_x, self.next_w)
        self.next_center_y = contours_helper.get_y_center(self.next_y, self.next_h)

class ScrollElementDetectionsFeatures():

    def __init__(self, common_element_features, scroll_buttons_patterns, shift_threshold_for_scrolls = 0):
        self.__common_element_features = common_element_features
        self.__scroll_buttons_patterns = scroll_buttons_patterns
        self.__shift_threshold_for_scrolls = shift_threshold_for_scrolls
