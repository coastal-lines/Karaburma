import datetime
import os
import sys
import cv2
from loguru import logger

from karaburma.data.constants.enums.element_types_enum import ElementTypesEnum
from karaburma.navigation.find_manager import FindManager
from karaburma.navigation.object_detection.default_actions.common_ml_find_actions import CommonMlFindActions
from karaburma.navigation.object_detection.default_actions.screenshot_ml_find_actions import ScreenshotMlFindActions
from karaburma.data.constants.enums.models_enum import ModelsEnum
from karaburma.elements.features.basic_element_features import BasicElementFeatures
from karaburma.elements.features.listbox_element_features import ListboxElementFeatures
from karaburma.elements.features.table.table_element_features import TableElementFeatures
from karaburma.utils import files_helper, json_output, debug
from karaburma.utils.config_manager import ConfigManager


class Karaburma:
    def __init__(self, config_path, source_mode, detection_mode, logging):

        self.config = ConfigManager(config_path)
        self.source_mode = source_mode
        self.detection_mode = detection_mode
        self.logging = logging

        if self.logging == True:
            logger.add("app_{time:YYYY-MM-DD}.log", rotation="5 MB")
            logger.info(f"New session was started at {datetime.datetime.now()}")

        #self.scroll_buttons_patterns = files_helper.load_grayscale_images_from_folder(ConfigManager().config.patterns_path.scroll_buttons)
        scroll_buttons_path = os.path.join(files_helper.get_project_root_path(), ConfigManager().config.patterns_path.scroll_buttons)
        self.scroll_buttons_patterns = files_helper.load_grayscale_images_from_folder(scroll_buttons_path)

        self.models_dict = dict()
        for model_name in ConfigManager().config.models:
            model_path = os.path.join(files_helper.get_project_root_path(), ConfigManager().config.models[model_name])
            model = files_helper.load_model(model_path)
            self.models_dict[model_name] = model

        self.common_element_features = BasicElementFeatures(self.models_dict[ModelsEnum.basic_model.name])

        self.table_element_features = TableElementFeatures(self.models_dict[ModelsEnum.tables_model.name],
                                                           self.models_dict[ModelsEnum.basic_model.name],
                                                           self.common_element_features,
                                                           self.scroll_buttons_patterns,
                                                           ConfigManager().config.elements_parameters.table.scrollbar_shift_threshold)

        self.listbox_element_features = ListboxElementFeatures(self.models_dict[ModelsEnum.basic_model.name],
                                                               self.models_dict[ModelsEnum.listbox_model.name],
                                                               self.common_element_features,
                                                               self.scroll_buttons_patterns,
                                                               ConfigManager().config.elements_parameters.listbox.scrollbar_shift_threshold)

        self.source_modes = {
                        "screenshot":
                            {
                                "default": ScreenshotMlFindActions(self.common_element_features,
                                                          self.table_element_features,
                                                          self.listbox_element_features)
                            },
                        "file":
                            {
                                "default": CommonMlFindActions(self.common_element_features,
                                                          self.table_element_features,
                                                          self.listbox_element_features)
                            }
        }

        if self.source_mode in self.source_modes and self.detection_mode in self.source_modes[self.source_mode]:
            self.selected_mode = self.source_modes[self.source_mode][self.detection_mode]
            self.find_manager = FindManager(self.selected_mode)
        else:
            print("")

    def __check_source_mode(self, args):
        if self.source_mode == "screenshot" and len(args) > 0:
            raise ValueError("Incorrect source mode. Please try to use '--source_mode file'.")

    def find_all_elements(self, *args):
        self.__check_source_mode(args)
        image_source = self.find_manager.find_all_elements(*args)

        debug.draw_elements(image_source.get_current_image_source_copy(), image_source)

        return json_output.convert_object_into_json(image_source)

    def find_element(self, element_type, *args):
        self.__check_source_mode(args)
        #image_source = self.find_manager.find_element(element_type, *args)
        image_source = self.find_manager.find_all_elements(*args)
        image_source.update_current_elements([element for element in image_source.get_elements() if element.get_label() == element_type])

        #debug.draw_elements(image_source.get_current_image_source_copy(), image_source)
        return json_output.convert_object_into_json(image_source)

    def find_table_and_expand(self, table_index=0):
        image_source = self.find_manager.find_table_and_expand(table_index)
        return json_output.convert_object_into_json(image_source)

    def find_table_cell(self, column, row):
        self.find_manager.find_table_cell(column, row)

    def find_listbox_and_expand(self, listbox_index=0):
        image_source = self.find_manager.find_listbox_and_expand(listbox_index)
        return json_output.convert_object_into_json(image_source)

    def find_element_by_patterns(self, patterns, mode="normal", threshold=0.8, user_label="", *args):
        image_source = self.find_manager.find_element_by_patterns(patterns, mode, threshold, user_label, *args)
        debug.draw_elements(image_source.get_current_image_source_copy(), image_source)

        return json_output.convert_object_into_json(image_source)

    def find_all_elements_include_patterns(self, patterns, mode="normal", threshold=0.8, user_label="", *args):
        image_source = self.find_manager.find_all_elements_include_patterns(patterns, mode, threshold, user_label, *args)
        debug.draw_elements(image_source.get_current_image_source_copy(), image_source)
        return json_output.convert_object_into_json(image_source)

karaburma = None

if __name__ == "__main__":
    config_path = r"Karaburma\karaburma\config.json"
    source_mode = "file"
    detection_mode = "default"
    logging = False

    global_config = ConfigManager(config_path)

    karaburma = Karaburma(config_path, source_mode, detection_mode, logging)