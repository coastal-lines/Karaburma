from karaburma.data.constants.enums.element_types_enum import ElementTypesEnum
from karaburma.navigation.object_detection.base_find_actions import BaseFindActions


class CommonMlFindActions(BaseFindActions):

    def __init__(self, common_element_features, table_element_features, listbox_element_features):
        self.common_element_features = common_element_features
        self.table_element_features = table_element_features
        self.listbox_element_features = listbox_element_features

    def find_element(self, image_source, element_type):
        match element_type:
            case ElementTypesEnum.table.name:
                self.table_element_features.find_all_tables(image_source)
            case ElementTypesEnum.listbox.name:
                self.listbox_element_features.find_listboxes(image_source)
            case _:
                self.common_element_features.find_elements(image_source, element_type)

    def find_all_elements(self, image_source):
        self.table_element_features.find_all_tables(image_source)
        self.common_element_features.find_all_elements(image_source)
        self.listbox_element_features.find_listboxes(image_source)