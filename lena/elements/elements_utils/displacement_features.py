import copy

from lena.data.constants.enums.scroll_direction_enum import ScrollDirectionEnum
from lena.utils.image_processing import filters_helper
from lena.utils.objects_tracking.displacement import DisplacementManager


class DisplacementFeatures(DisplacementManager):

    def __init__(self, displacement_features, scroll_features, horizontal_border: int = 0, vertical_border: int = 0):
        super().__init__(displacement_features)
        self.__scroll_features = scroll_features
        self.__horizontal_border = horizontal_border
        self.__vertical_border = vertical_border

    @property
    def scroll_features(self):
        return self.__scroll_features

    def __calculate_horizontal_displacement(self, before, after) -> int:
        before = before[self.__vertical_border:before.shape[0] - self.__vertical_border, self.__horizontal_border:before.shape[1] - self.__horizontal_border]
        after = after[self.__vertical_border:after.shape[0] - self.__vertical_border, self.__horizontal_border:after.shape[1] - self.__horizontal_border]
        x_displacement, _ = super().calculate_displacement(before, after)

        return x_displacement
