import copy
from lena.data.constants.enums.element_types_enum import ElementTypesEnum
from lena.elements.objects.element import Element


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

