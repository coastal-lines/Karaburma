import copy

from karaburma.data.constants.enums.scroll_direction_enum import ScrollDirectionEnum
from karaburma.utils.image_processing import filters_helper
from karaburma.utils.objects_tracking.displacement import DisplacementManager


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

    def __calculate_vertical_displacement(self, before, after) -> int:
        before = before[self.__vertical_border:before.shape[0] - self.__vertical_border, self.__horizontal_border:before.shape[1] - self.__horizontal_border]
        after = after[self.__vertical_border:after.shape[0] - self.__vertical_border, self.__horizontal_border:after.shape[1] - self.__horizontal_border]
        _, y_displacement = super().calculate_displacement(before, after)

        return y_displacement

    def calculate_displacement_for_scrolling(self, before, after, direction: str) -> tuple[int, int]:
        x_displacement = 0
        y_displacement = 0

        match direction:
            case ScrollDirectionEnum.DOWN.name | ScrollDirectionEnum.UP.name:
                y_displacement = self.__calculate_vertical_displacement(before, after)

            case ScrollDirectionEnum.RIGHT.name | ScrollDirectionEnum.LEFT.name:
                x_displacement = self.__calculate_horizontal_displacement(before, after)

        return x_displacement, y_displacement

    def try_to_find_displacement(self):
        before = filters_helper.convert_to_grayscale(copy.copy(self.__scroll_features.element_with_scroll.get_roi_element().get_roi()))

        if self.__scroll_features.direction == ScrollDirectionEnum.RIGHT_DOWN.name:
            was_scrolled, roi_element_after_scroll = self.__scroll_features.scroll_element(ScrollDirectionEnum.RIGHT.name)
            x_displacement, _ = self.calculate_displacement_for_scrolling(before, roi_element_after_scroll, ScrollDirectionEnum.RIGHT.name)
            self.__scroll_features.scroll_element(ScrollDirectionEnum.LEFT.name)

            was_scrolled, roi_element_after_scroll = self.__scroll_features.scroll_element(ScrollDirectionEnum.DOWN.name)
            _, y_displacement = self.calculate_displacement_for_scrolling(before, roi_element_after_scroll, ScrollDirectionEnum.DOWN.name)
            self.__scroll_features.scroll_element(ScrollDirectionEnum.UP.name)

        else:
            was_scrolled, roi_element_after_scroll = self.__scroll_features.scroll_element()
            x_displacement, y_displacement = self.calculate_displacement_for_scrolling(before, roi_element_after_scroll, self.__scroll_features.direction)
            self.__scroll_features.scroll_element(self.scroll_features.reversed_direction)

        return x_displacement, y_displacement