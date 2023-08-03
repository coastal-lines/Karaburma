from lena.data.constants.enums.element_types_enum import ElementTypesEnum
from lena.navigation.object_detection.base_find_actions import BaseFindActions
from lena.utils import debug


class CommonMlFindActions(BaseFindActions):

    def __init__(self, common_element_features, table_element_features, listbox_element_features):
        self.common_element_features = common_element_features
        self.table_element_features = table_element_features
        self.listbox_element_features = listbox_element_features