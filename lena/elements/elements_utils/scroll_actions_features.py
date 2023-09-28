from lena.data.constants.enums.scroll_direction_enum import ScrollDirectionEnum
from lena.utils import general_helpers, mouse_actions


class ScrollActionsFeatures:

    def __init__(self, element, nested_element, direction):

        self.element_with_scroll = element
        self.nested_element = nested_element
        self.__direction = direction

        self.nested_element_w, self.nested_element_h = self.nested_element.get_roi_element().get_shape()
        self.reversed_direction = self.__get_reversed_direction(direction)

    @property
    def direction(self):
        return self.__direction

    @direction.setter
    def direction(self, direction):
        self.__direction = direction
