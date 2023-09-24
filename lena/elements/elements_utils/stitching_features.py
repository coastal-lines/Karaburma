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
