import json
import os

from config import config_from_json

from lena.data.constants.enums.models_enum import ModelsEnum
from lena.utils.logging_manager import LoggingManager



class ConfigReader:
    def __init__(self, config_path="config.json"):
        self.__config_file_path = "config.json"

    def __load_config(self):
        file_dir = os.path.dirname(os.path.realpath('__file__'))
        self.__config_file_path = os.path.join(file_dir, self.__config_file_path)

        try:
            with open(self.__config_file_path, 'r') as config_file:
                return json.load(config_file), file_dir
        except FileNotFoundError:
            raise FileNotFoundError(LoggingManager().log_error(f"Config file was not found at {self.__config_file_path}"))

    def __travers_throw_config(self, object, desired_object_name):
        for key in object:
            if (key == desired_object_name):
                return object[key]

    def get_parameter(self, object_names, parameter_name, config_dictonary=None, start_index=0):
        if(config_dictonary == None):
            config_dictonary = self.__config

        for i in range(start_index, len(object_names)):
            for config_key in config_dictonary:
                if (config_key == object_names[i]):
                    try:
                        return config_dictonary[object_names[i]][parameter_name]
                    except KeyError:
                        return self.get_parameter(object_names, parameter_name, config_dictonary[object_names[i]], start_index + 1)

        raise Exception(f"Parameter '{parameter_name}' in '{object_names}' were not found in the config.")

    def get_model_file_path(self, model_name):
        try:
            return os.path.join(os.path.dirname(os.path.realpath('__file__')),
                                self.__config["models"][model_name]).replace("\\", "/")
        except KeyError:
            raise KeyError(LoggingManager().log_error(f"Parameter '{model_name}' was not found in the config."))

    def get_combined_files_path(self, files):
        return os.path.join(os.path.dirname(os.path.realpath('__file__')), files.replace("\\", "/"))