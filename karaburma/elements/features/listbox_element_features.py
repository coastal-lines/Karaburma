import numpy as np

from karaburma.elements.elements_utils.displacement_features import DisplacementFeatures
from karaburma.elements.elements_utils.preprocessing.listbox_preprocessing import ListboxPreprocessing
from karaburma.elements.elements_utils.scroll_actions_features import ScrollActionsFeatures
from karaburma.elements.elements_utils.stitching_features import StitchingFeatures
from karaburma.utils.config_manager import ConfigManager
from karaburma.utils.image_processing import augmentation
from karaburma.data.constants.enums.scroll_direction_enum import ScrollDirectionEnum
from karaburma.elements.objects.element import Element
from karaburma.elements.objects.roi_element import RoiElement
from karaburma.utils.image_processing import filters_helper
from karaburma.utils.ocr import ocr_helper
from karaburma.utils.objects_tracking.displacement import OcrVerticalDisplacement


class ListboxElementFeatures(ListboxPreprocessing):
    def __init__(self, model_for_common_elements, model_for_listbox, common_element_features, scroll_buttons_patterns, shift_threshold_for_scrolls):
        super().__init__(model_for_common_elements, model_for_listbox, common_element_features, scroll_buttons_patterns, shift_threshold_for_scrolls)

        self.model_for_common_elements = model_for_common_elements
        self.__model_for_listbox = model_for_listbox


    def __extend_stiched_listbox_roi(self, temp_stiched_listbox):
        # Exclude scrollbar element from listbox roi - take 80% of listbox's width
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
        roi = augmentation.bicubic_resize(roi, (roi.shape[0] * 3, roi.shape[1] * 3))
        roi = filters_helper.blur(roi, (3, 3))

        return roi

    def __get_stitched_list_box(self, temp_listbox, list_area):
        scroll_features = ScrollActionsFeatures(temp_listbox, list_area, ScrollDirectionEnum.DOWN.name)
        displacement_features = DisplacementFeatures(OcrVerticalDisplacement(), scroll_features)
        stitching_features = StitchingFeatures(displacement_features)
        stitched_listbox = stitching_features.vertical_stitch()

        return stitched_listbox

    def __get_text_list_for_listbox(self, extended_temp_stiched_listbox):
        prepared_roi_for_reading_text = self.__image_preprocessing_for_text_reading(extended_temp_stiched_listbox)
        text_list = ocr_helper.get_text(prepared_roi_for_reading_text, "--psm 6 --oem 3")

        return str(text_list)

    def find_listboxes(self, image_source):
        list_listboxes = super().listbox_element_classification(image_source.get_current_image_source())
        image_source.add_elements(list_listboxes)

    def find_listbox_and_expand(self, image_source, listbox_index):
        list_listboxes = self.find_listboxes(image_source.get_current_image_source())
        image_source.add_elements(list_listboxes)

        if (len(list_listboxes) > 0):
            stitched_listbox_roi = self.__get_stitched_list_box(list_listboxes[listbox_index], list_listboxes[listbox_index].textarea)
            list_listboxes[listbox_index].full_text_area = Element("listbox_full_text_area", 1.0,
                                                                   RoiElement(stitched_listbox_roi, 0, 0,
                                                                              stitched_listbox_roi.shape[1],
                                                                              stitched_listbox_roi.shape[0]))

            text = self.__get_text_list_for_listbox(list_listboxes[listbox_index].full_text_area.get_roi_element().get_roi())
            list_listboxes[listbox_index].add_list_text(text)