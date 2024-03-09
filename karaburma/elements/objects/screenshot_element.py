import copy

from karaburma.data.constants.enums.element_types_enum import ElementTypesEnum
from karaburma.elements.objects.element import Element


class ImageSourceObject:
    def __init__(self, image_source):
        self.__image_source = image_source
        self.__image_source_copy = copy.deepcopy(image_source)
        self.__list_elements = []
        self.__list_table_elements = []

    def get_current_image_source(self):
        return self.__image_source

    def get_current_image_source_copy(self):
        return self.__image_source_copy

    def update_current_image_source(self, updated_image_source):
        self.__image_source = updated_image_source

    def get_current_image_source_dimension(self):
        return self.__image_source.shape[1], self.__image_source.shape[0]

    def add_element(self, element: Element):
        self.__list_elements.append(element)

    def add_elements(self, elements: list):
        self.__list_elements.extend(elements)

    def get_elements(self) -> list:
        return self.__list_elements

    def get_table_elements(self) -> list:
        return [table for table in self.__list_elements if table.get_label() == ElementTypesEnum.table.name]

    def update_current_elements(self, new_elements: list):
        self.get_elements().clear()
        self.add_elements(new_elements)

    def update_text_for_all_elements(self):
        for element in self.get_elements():
            match element.get_label():
                case "listbox":
                    element.prepare_roi_and_set_text()

                    if (element.full_text_area is not None):
                        element.full_text_area.prepare_roi_and_set_text()
                case "table":
                    element.update_text_for_all_cells()

                    if (element.get_full_table_area() is not None):
                        element.get_full_table_area().update_text_for_all_cells()
                case _:
                    if (hasattr(element, 'get_text')):
                        element.prepare_roi_and_set_text()
