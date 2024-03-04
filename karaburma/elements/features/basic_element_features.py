import numpy as np

from karaburma.data.constants.enums.element_types_enum import ElementTypesEnum
from karaburma.elements.elements_utils.preprocessing.basic_preprocessing import BasicPreprocessing
from karaburma.elements.features.checkbox_element_features import CheckboxElementFeatures
from karaburma.elements.features.radiobutton_element_features import RadioButtonElementFeatures
from karaburma.elements.objects.button_element import ButtonElement
from karaburma.elements.objects.checkbox_element import CheckboxElement
from karaburma.elements.objects.combobox_element import ComboboxElement
from karaburma.elements.objects.input_element import InputElement
from karaburma.elements.objects.radiobutton_element import RadioButtonElement
from karaburma.elements.objects.slider_element import SliderElement
from karaburma.utils.config_manager import ConfigManager
from karaburma.utils.logging_manager import LoggingManager


class BasicElementFeatures(BasicPreprocessing):
    def __init__(self, basic_model):
        self.__image_source = None
        self.__basic_model = basic_model
        self.__dim = tuple(ConfigManager().config.elements_parameters.common_element.preprocessing["sample_dimension"])

        self.__checkbox_element_features = CheckboxElementFeatures()
        self.__radiobutton_element_features = RadioButtonElementFeatures()

    def __re_calculate_scores_for_element_with_feature(self, roi, predictions_proba):
        labels = [ElementTypesEnum.button.name,
                  ElementTypesEnum.checkbox.name,
                  ElementTypesEnum.combobox.name,
                  ElementTypesEnum.h_scroll.name,
                  ElementTypesEnum.input.name,
                  ElementTypesEnum.non.name,
                  ElementTypesEnum.radiobutton.name,
                  ElementTypesEnum.slider.name]

        if roi.get_class_feature() == ElementTypesEnum.radiobutton.name:
            roi_predictions_proba = ConfigManager().config.elements_parameters.common_element.weights["radiobutton"]
            temp_predictions = predictions_proba[0] * np.array(roi_predictions_proba)

            # Do normalize - each value divide by sum
            new_predictions = temp_predictions / np.sum(temp_predictions)
            index_of_max_scrore = np.where(new_predictions == new_predictions.max())[0][0]

            return labels[index_of_max_scrore], new_predictions[index_of_max_scrore]

        elif roi.get_class_feature() == ElementTypesEnum.checkbox.name:
            roi_predictions_proba = ConfigManager().config.elements_parameters.common_element.weights["checkbox"]
            temp_predictions = predictions_proba[0] * np.array(roi_predictions_proba)

            # Do normalize - each value divide by sum
            new_predictions = temp_predictions / np.sum(temp_predictions)
            index_of_max_scrore = np.where(new_predictions == new_predictions.max())[0][0]

            return labels[index_of_max_scrore], new_predictions[index_of_max_scrore]

        elif (roi.get_class_feature() == ElementTypesEnum.h_scroll.name or roi.get_class_feature() == ElementTypesEnum.v_scroll.name):
            roi_predictions_proba = ConfigManager().config.elements_parameters.common_element.weights["scroll"]
            temp_predictions = predictions_proba[0] * np.array(roi_predictions_proba)

            # Do normalize - each value divide by sum
            new_predictions = temp_predictions / np.sum(temp_predictions)
            index_of_max_scrore = np.where(new_predictions == new_predictions.max())[0][0]

            return labels[index_of_max_scrore], new_predictions[index_of_max_scrore]

        elif roi.get_class_feature() == None:
            roi_predictions_proba = ConfigManager().config.elements_parameters.common_element.weights["non"]
            temp_predictions = predictions_proba[0] * np.array(roi_predictions_proba)

            # Do normalize - each value divide by sum
            new_predictions = temp_predictions / np.sum(temp_predictions)
            index_of_max_scrore = np.where(new_predictions == new_predictions.max())[0][0]

            return labels[index_of_max_scrore], new_predictions[index_of_max_scrore]

    def calculate_scores_for_element(self, roi):
        feature = super().prepare_features_for_basic_elements(roi.get_roi(), self.__dim)
        img = [np.array(feature).flatten()]

        try:
            predictions_proba = self.__basic_model.predict_proba(img)
        except ValueError as ex:
            raise ValueError(LoggingManager().log_error(ex.args[0]))

        most_common_label, prediction_value = self.__re_calculate_scores_for_element_with_feature(roi, predictions_proba)
        return str(most_common_label), prediction_value

    def __element_classification(self, list_of_roi, type_of_element):
        list_elements = []

        for i in range(len(list_of_roi)):
            most_common_label, prediction_value = self.calculate_scores_for_element(list_of_roi[i])
            print(most_common_label, prediction_value)

            if(most_common_label == type_of_element):
                match most_common_label:
                    case ElementTypesEnum.button.name:
                        list_elements.append(ButtonElement(most_common_label, prediction_value, list_of_roi[i]))

                    case ElementTypesEnum.checkbox.name:
                        list_elements.append(CheckboxElement(most_common_label, prediction_value, list_of_roi[i]))

                    case ElementTypesEnum.combobox.name:
                        list_elements.append(ComboboxElement(most_common_label, prediction_value, list_of_roi[i]))

                    case ElementTypesEnum.input.name:
                        list_elements.append(InputElement(most_common_label, prediction_value, list_of_roi[i]))

                    case ElementTypesEnum.radiobutton.name:
                        list_elements.append(RadioButtonElement(most_common_label, prediction_value, list_of_roi[i]))

                    case ElementTypesEnum.slider.name:
                        list_elements.append(SliderElement(most_common_label, prediction_value, list_of_roi[i]))

        return list_elements

    def find_elements(self, image_source, type_of_element):
        temp_list_elements = []

        self.__image_source = image_source

        match type_of_element:
            case ElementTypesEnum.button.name | ElementTypesEnum.combobox.name | ElementTypesEnum.input.name | ElementTypesEnum.slider.name:
                list_of_roi = super().find_contours_for_common_elements(self.__image_source)
                temp_list_elements.extend(self.__element_classification(list_of_roi, type_of_element))

            case ElementTypesEnum.checkbox.name:
                list_of_roi = self.__checkbox_element_features.find_contours_for_checkbox_elements(self.__image_source)
                temp_list_elements.extend(self.__element_classification(list_of_roi, type_of_element))

            case ElementTypesEnum.radiobutton.name:
                list_of_roi = self.__radiobutton_element_features.find_roi_for_element(self.__image_source)
                temp_list_elements.extend(self.__element_classification(list_of_roi, type_of_element))

        self.__image_source.add_elements(temp_list_elements)

    def find_all_elements(self, screenshot_elements):
        for type in [ElementTypesEnum.button,
                     ElementTypesEnum.combobox,
                     ElementTypesEnum.input,
                     ElementTypesEnum.slider,
                     ElementTypesEnum.checkbox,
                     ElementTypesEnum.radiobutton]:

            self.find_elements(screenshot_elements, type.name)