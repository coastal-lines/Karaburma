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

    def __get_rectangles_by_patterns(self, temp_roi: np.array, similarity_threshold: float=0.8) -> list:
        scroll_rectangles = []

        for pattern in self.__scroll_buttons_patterns:
            template_height, template_width = pattern.shape
            _, max_val, _, max_loc = pattern_matching.calculate_min_max(temp_roi, pattern)

            top_left, bottom_right = None, None
            if max_val >= similarity_threshold:
                top_left = max_loc
                bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

            if top_left is not None and bottom_right is not None:
                scroll_rectangles.append((top_left[0], top_left[1], template_width, template_height))

        return scroll_rectangles

    def __get_rectangles(self, temp_table_roi):
        ref_gray = filters_helper.convert_to_grayscale(temp_table_roi)
        ref_gray_sharp = filters_helper.Sharp(ref_gray, "strong")
        ret, thresh = cv2.threshold(ref_gray_sharp, 180, 255, 0)
        filtered_rectangles = contours_helper.get_contours_after_approximation(thresh, 3)

        return filtered_rectangles
