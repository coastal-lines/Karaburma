import os
from skimage.io import imread as sk_image_reader
import numpy as np
import skimage
import skimage as sk
import imutils.object_detection
import cv2
import imutils.object_detection
from skimage.filters import threshold_mean
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from lena.data.constants.enums.element_types_enum import ElementTypesEnum
from lena.elements.elements_utils.preprocessing.basic_preprocessing import BasicPreprocessing
from lena.elements.features.checkbox_element_features import CheckboxElementFeatures
from lena.elements.features.radiobutton_element_features import RadioButtonElementFeatures
from lena.elements.objects.button_element import ButtonElement
from lena.elements.objects.checkbox_element import CheckboxElement
from lena.elements.objects.combobox_element import ComboboxElement
from lena.elements.objects.input_element import InputElement
from lena.elements.objects.radiobutton_element import RadioButtonElement
from lena.elements.objects.slider_element import SliderElement
from lena.utils.config_manager import ConfigManager
from lena.utils.logging_manager import LoggingManager


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
