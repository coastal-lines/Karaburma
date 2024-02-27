import cv2
import numpy as np
import pyautogui

from karaburma.data.constants.enums.element_types_enum import ElementTypesEnum
from karaburma.data.constants.enums.scroll_direction_enum import ScrollDirectionEnum
from karaburma.utils import general_helpers, files_helper
from karaburma.elements.objects.element import Element
from karaburma.elements.objects.scroll_element import ScrollElement
from karaburma.utils.image_processing import filters_helper, geometric_transformations, contours_helper, pattern_matching
from karaburma.elements.objects.roi_element import RoiElement

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
        ref_gray_sharp = filters_helper.sharp(ref_gray, "strong")
        ret, thresh = cv2.threshold(ref_gray_sharp, 180, 255, 0)
        filtered_rectangles = contours_helper.get_contours_after_approximation(thresh, 3)

        return filtered_rectangles

    def __prepare_rectangle_for_vertical_scroll(self, temp_attributes, y_max, y_min):
        x = np.min([temp_attributes.current_x, temp_attributes.next_x]) - self.__shift_threshold_for_scrolls
        y = y_min - self.__shift_threshold_for_scrolls
        w = temp_attributes.current_w + self.__shift_threshold_for_scrolls
        h = temp_attributes.current_h + self.__shift_threshold_for_scrolls + (y_max - y_min)
        return x, y, w, h

    def __prepare_rectangle_for_horizontal_scroll(self, temp_attributes, x_max, x_min):
        x = x_min - self.__shift_threshold_for_scrolls
        y = np.min([temp_attributes.current_y, temp_attributes.next_y]) - self.__shift_threshold_for_scrolls
        w = temp_attributes.current_w + self.__shift_threshold_for_scrolls + (x_max - x_min)
        h = temp_attributes.current_h + self.__shift_threshold_for_scrolls
        return x, y, w, h

    def __init_scroll_button(self, source_element, x, y, w, h, button_type, button_side):
        button_absolute_x, button_absolute_y = general_helpers.calculate_absolute_coordinates(source_element, x, y)
        roi_element = RoiElement(source_element.get_roi()[y:y + h, x:x + h, :], button_absolute_x, button_absolute_y, button_side, button_side)
        return Element(button_type, 1.0, roi_element)

    def __init_v_scroll(self, source_element, x, y, w, h):
        temp_v_scroll_up_button = self.__init_scroll_button(source_element, x, y, w, h, ScrollDirectionEnum.UP.name, w)

        current_down_button_y = (y + h) - w
        temp_v_scroll_down_button = self.__init_scroll_button(source_element, x, current_down_button_y, w, h, ScrollDirectionEnum.DOWN.name, w)

        return self.__init_scroll(source_element, temp_v_scroll_up_button, temp_v_scroll_down_button, x, y, w, h, ElementTypesEnum.v_scroll.name)

    def __init_h_scroll(self, source_element, x, y, w, h):
        temp_h_scroll_left_button = self.__init_scroll_button(source_element, x, y, w, h, ScrollDirectionEnum.LEFT.name, h)

        current_right_button_x = (x + w) - h
        temp_h_scroll_right_button = self.__init_scroll_button(source_element, current_right_button_x, y, w, h, ScrollDirectionEnum.RIGHT.name, h)

        return self.__init_scroll(source_element, temp_h_scroll_left_button, temp_h_scroll_right_button, x, y, w, h, ElementTypesEnum.h_scroll.name)

    def __init_scroll(self, source_element, first_button, second_button, x, y, w, h, scroll_type):
        temp_scroll_absolute_x, temp_scroll_absolute_y = general_helpers.calculate_absolute_coordinates(source_element, x, y)
        temp_scroll_roi = RoiElement(source_element.get_roi()[y:y + h, x:x + w, :], temp_scroll_absolute_x, temp_scroll_absolute_y, w, h, scroll_type)
        return ScrollElement(scroll_type, 0.0, temp_scroll_roi, first_button, second_button)

    def __try_to_detect_scrolls(self, rectangles, source_element: RoiElement):
        temp_h_scrolls = []
        temp_v_scrolls = []

        '''
        # calculate the length and width of the window and take 85%
        # according to logic, the scroll cannot be smaller than these dimensions
        '''
        w_border = int((source_element.get_w() / 100) * 70)
        h_border = int((source_element.get_h() / 100) * 50)

        rectangles = np.array(rectangles)
        sorted_rectangles_by_x = rectangles[rectangles[:, 0].argsort()]
        sorted_rectangles_by_y = rectangles[rectangles[:, 1].argsort()]

        for i in range(len(sorted_rectangles_by_x) - 1):
            temp_attributes = TempScrollAttributes(sorted_rectangles_by_x, i)

            if temp_attributes.current_center_x - 1 <= temp_attributes.next_center_x <= temp_attributes.current_center_x + 1:
                y_max = np.max([temp_attributes.current_y, temp_attributes.next_y])
                y_min = np.min([temp_attributes.current_y, temp_attributes.next_y])
                x, y, w, h = self.__prepare_rectangle_for_vertical_scroll(temp_attributes, y_max, y_min)

                # additionally, check that the height is sufficiently large and that
                # the target point is approximately on the right
                if (h > h_border and x > w_border):
                    x = x - self.__shift_threshold_for_scrolls
                    y = y - self.__shift_threshold_for_scrolls
                    temp_v_scrolls.append(self.__init_v_scroll(source_element, x, y, w, h))

        for i in range(len(sorted_rectangles_by_y) - 1):
            temp_attributes = TempScrollAttributes(sorted_rectangles_by_y, i)

            if (temp_attributes.current_center_y - 1 <= temp_attributes.next_center_y <= temp_attributes.current_center_y + 1):
                x_min = np.min([temp_attributes.current_x, temp_attributes.next_x])
                x_max = np.max([temp_attributes.current_x, temp_attributes.next_x])
                x, y, w, h = self.__prepare_rectangle_for_horizontal_scroll(temp_attributes, x_max, x_min)

                # additionally, check that the width is sufficiently large and that
                # the target point is approximately at the bottom
                if (w > w_border and y > h_border):
                    x = x - self.__shift_threshold_for_scrolls
                    y = y - self.__shift_threshold_for_scrolls

                    if(w < source_element.get_w()):
                        temp_h_scrolls.append(self.__init_h_scroll(source_element, x, y, w, h))

        return temp_h_scrolls, temp_v_scrolls

    def __classify_horizontal_scroll(self, potential_h_scrolls):
        for i in range(len(potential_h_scrolls)):
            label, prediction_value = self.__common_element_features.calculate_scores_for_element(potential_h_scrolls[i].get_roi_element())

            if label == ElementTypesEnum.h_scroll.name:
                potential_h_scrolls[i].update_prediction_value(prediction_value)

        return max(potential_h_scrolls, key=lambda obj: obj.get_prediction_value())

    def __classify_vertical_scroll(self, potential_v_scrolls):
        for i in range(len(potential_v_scrolls)):
            original_v_scroll_roi = potential_v_scrolls[i].get_roi_element().get_roi()
            turned_left_temp_v_scroll = geometric_transformations.turn_left(potential_v_scrolls[i].get_roi_element().get_roi())
            potential_v_scrolls[i].get_roi_element().set_roi(turned_left_temp_v_scroll)
            label, prediction_value = self.__common_element_features.calculate_scores_for_element(potential_v_scrolls[i].get_roi_element())

            if label == ElementTypesEnum.h_scroll.name:
                #TODO - for debuging only
                if(prediction_value != 1.0):
                    potential_v_scrolls[i].get_roi_element().set_roi(original_v_scroll_roi)
                    potential_v_scrolls[i].update_prediction_value(prediction_value)

        return max(potential_v_scrolls, key=lambda obj: obj.get_prediction_value())

    def __classify_potential_scrolls(self, potential_h_scrolls, potential_v_scrolls):
        h_scroll, v_scroll = None, None

        if len(potential_h_scrolls) > 0:
            h_scroll = self.__classify_horizontal_scroll(potential_h_scrolls)

        if len(potential_v_scrolls) > 0:
            v_scroll = self.__classify_vertical_scroll(potential_v_scrolls)

        return h_scroll, v_scroll

    def find_scrolls(self, temp_roi: RoiElement) -> tuple:
        rectangles = []
        rectangles.extend(self.__get_rectangles(temp_roi.get_roi()))
        rectangles.extend(self.__get_rectangles_by_patterns(temp_roi.get_roi(), 0.6))

        if len(rectangles) > 0:
            temp_h_scrolls, temp_v_scrolls = self.__try_to_detect_scrolls(rectangles, temp_roi)
            h_scroll, v_scroll = self.__classify_potential_scrolls(temp_h_scrolls, temp_v_scrolls)

            return h_scroll, v_scroll
        else:
            return None, None