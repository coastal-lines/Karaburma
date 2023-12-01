from karaburma.data.constants.enums.scroll_direction_enum import ScrollDirectionEnum
from karaburma.utils import general_helpers, mouse_actions


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

    def __get_reversed_direction(self, direction):
        match direction:
            case ScrollDirectionEnum.RIGHT.name:
                return ScrollDirectionEnum.LEFT.name

            case ScrollDirectionEnum.DOWN.name:
                return ScrollDirectionEnum.UP.name

            case ScrollDirectionEnum.RIGHT_DOWN.name:
                return ScrollDirectionEnum.LEFT_UP.name

    def __was_scrolled(self, before, after, current_direction=None):
        direction = self.direction if current_direction == None else current_direction

        match direction:
            case ScrollDirectionEnum.RIGHT.name:
                similarity = general_helpers.calculate_similarity(before[self.nested_element_h - (self.nested_element_h // 5):self.nested_element_h, :, :],
                                                                  after[self.nested_element_h - (self.nested_element_h // 5):self.nested_element_h, :, :])
                return True if similarity < 1.0 else False

            case ScrollDirectionEnum.DOWN.name:
                #similarity = general_helpers.calculate_similarity(before[:, self.nested_element_w - (self.nested_element_w // 4):self.nested_element_w],
                #                                                  after[:, self.nested_element_w - (self.nested_element_w // 4):self.nested_element_w])

                similarity = general_helpers.calculate_similarity(before, after)

                return True if similarity < 1.0 else False

            case ScrollDirectionEnum.LEFT.name:
                similarity = general_helpers.calculate_similarity(before[:, self.nested_element_w - (self.nested_element_w // 4):self.nested_element_w],
                                                                  after[:, self.nested_element_w - (self.nested_element_w // 4):self.nested_element_w])
                return True if similarity < 1.0 else False

            case ScrollDirectionEnum.RIGHT_DOWN.name:
                similarity_down = general_helpers.calculate_similarity(before[:, 0:self.nested_element_w // 5],
                                                                  after[:, 0:self.nested_element_w // 5])

                similarity_left = general_helpers.calculate_similarity(before[:, 0:self.nested_element_w // 5],
                                                                  after[:, 0:self.nested_element_w // 5])

                return True if similarity_down < 1.0 and similarity_left < 1.0 else False

            case _:
                raise Exception("Date provided can't be in the past")

    def scroll_element(self, current_direction=None):
        before, after = None, None
        direction = self.direction if current_direction == None else current_direction

        match direction:
            case ScrollDirectionEnum.RIGHT.name:
                before, after = mouse_actions.click_and_return_difference(self.element_with_scroll, self.element_with_scroll.get_h_scroll().get_second_button().get_centroid())
                #general_helpers.click(self.element_with_scroll.get_h_scroll().get_second_button().get_centroid())

            case ScrollDirectionEnum.LEFT.name:
                #time.sleep(1)
                before, after = mouse_actions.click_and_return_difference(self.element_with_scroll, self.element_with_scroll.get_h_scroll().get_first_button().get_centroid())
                #general_helpers.click(self.element_with_scroll.get_h_scroll().get_first_button().get_centroid())
                #was_scrolled_result = self.was_scrolled(before, self.get_roi_element_after_update(element), element, direction)

            case ScrollDirectionEnum.DOWN.name:
                before, after = mouse_actions.click_and_return_difference(self.element_with_scroll, self.element_with_scroll.get_v_scroll().get_second_button().get_centroid())
                #general_helpers.click(self.element_with_scroll.get_v_scroll().get_second_button().get_centroid())

            case ScrollDirectionEnum.UP.name:
                before, after = mouse_actions.click_and_return_difference(self.element_with_scroll, self.element_with_scroll.get_v_scroll().get_first_button().get_centroid())
                #general_helpers.click(self.element_with_scroll.get_v_scroll().get_first_button().get_centroid())

            #case ScrollDirectionEnum.RIGHT_DOWN.name:
            #    general_helpers.click(self.element_with_scroll.get_v_scroll().get_second_button().get_centroid())
            #    general_helpers.click(self.element_with_scroll.get_h_scroll().get_second_button().get_centroid())

            #case ScrollDirectionEnum.LEFT_UP.name:
            #    general_helpers.click(self.element_with_scroll.get_v_scroll().get_first_button().get_centroid())
            #    general_helpers.click(self.element_with_scroll.get_h_scroll().get_first_button().get_centroid())

            case _:
                print("Something's wrong with the param")

        was_scrolled_result = self.__was_scrolled(before, after)

        return was_scrolled_result, after

    def scroll_until_its_possible(self, current_direction=None):
        direction = self.direction if current_direction == None else current_direction

        match direction:
            case ScrollDirectionEnum.LEFT.name:
                if self.scroll_element(ScrollDirectionEnum.LEFT.name)[0]:
                    # TODO - add log
                    print("click left")
                    self.scroll_until_its_possible(direction)