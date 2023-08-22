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

    def __image_preprocessing_for_text_reading(self, roi):
        #roi = augmentation.bicubic_resize(roi, (roi.shape[0] * 3, roi.shape[1] * 1))
        roi = augmentation.bicubic_resize(roi, (roi.shape[0] * 3, roi.shape[1] * 3))
        roi = filters_helper.Blur(roi, (3, 3))

        return roi

    def __get_stitched_list_box(self, temp_listbox, list_area):
        scroll_features = ScrollActionsFeatures(temp_listbox, list_area, ScrollDirectionEnum.DOWN.name)
        displacement_features = DisplacementFeatures(OcrVerticalDisplacement(), scroll_features)
        stitching_features = StitchingFeatures(displacement_features)
        stitched_listbox = stitching_features.vertical_stitch()
        #general_helpers.show(temp_stitched_listbox)

        return stitched_listbox


    def __get_text_list_for_listbox(self, temp_listbox, list_area):

        #scroll = ScrollFeatures(temp_listbox, list_area, ScrollDirectionEnum.DOWN.name, OcrVerticalDisplacement(), None, None, None)
        #temp_stiched_listbox = scroll.vertical_scroll_and_accumulate_rois_difference()

        temp_stitched_listbox = self.__get_stitched_list_box(temp_listbox, list_area)

        extended_temp_stiched_listbox = self.__extend_stiched_listbox_roi(temp_stitched_listbox)
        prepared_roi_for_reading_text = self.__image_preprocessing_for_text_reading(extended_temp_stiched_listbox)
        general_helpers.show(prepared_roi_for_reading_text)
        text_list = ocr_helper.get_text(prepared_roi_for_reading_text, "--psm 6 --oem 3")
        #print(text_list)

        return text_list

    def find_listboxes(self, image_source):
        self.__image_source = image_source
        list_listboxes = super().listbox_element_classification(self.__image_source.get_current_image_source())
        image_source.add_elements(list_listboxes)

        return list_listboxes
    