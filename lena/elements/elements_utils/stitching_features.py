import copy
import numpy as np

from lena.data.constants.enums.scroll_direction_enum import ScrollDirectionEnum
from lena.utils import files_helper


class StitchingFeatures:
    def __init__(self, displacement_features, horizontal_stiching_shift=None, horizontal_roi_shift=None, vertical_stiching_shift=None):
        self.displacement_features = displacement_features

        self.horizontal_stitching_shift = horizontal_stiching_shift
        self.vertical_stitching_shift = vertical_stiching_shift

        if (horizontal_roi_shift != None):
            self.horizontal_roi_shift = horizontal_roi_shift
            self.displacement_features.scroll_features.nested_element.get_roi_element().update_w(self.horizontal_roi_shift)
            #TODO - try to use delegate here?

        self.x_displacement, self.y_displacement = self.__prepare_displacement_values()

        self.nested_element_w, self.nested_element_h = self.displacement_features.scroll_features.nested_element.get_roi_element().get_shape()
        print("")

    def __prepare_displacement_values(self) -> tuple[int, int]:
        x_displacement, y_displacement = self.displacement_features.try_to_find_displacement()

        if (self.horizontal_stitching_shift != None):
            x_displacement += self.horizontal_stitching_shift

        if (self.vertical_stitching_shift != None):
            y_displacement += self.vertical_stitching_shift

        print("x_displacement", x_displacement,"y_displacement", y_displacement)
        return x_displacement, y_displacement

    def __get_crop_after_scroll(self, temp_nested_element):
        match self.displacement_features.scroll_features.direction:
            case ScrollDirectionEnum.RIGHT.name:
                roi_after = temp_nested_element[:, self.nested_element_w - self.x_displacement: self.nested_element_w, :]
                return roi_after

            case ScrollDirectionEnum.DOWN.name:
                roi_after = temp_nested_element[self.nested_element_h - self.y_displacement: self.nested_element_h, :, :]
                return roi_after
