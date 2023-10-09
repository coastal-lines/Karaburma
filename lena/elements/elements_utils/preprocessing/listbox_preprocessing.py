import cv2
import numpy as np
from PIL import Image as PIL_Image
from sklearn.preprocessing import MinMaxScaler

from lena.elements.features.scroll_element_features import ScrollElementDetectionsFeatures
from lena.elements.objects.roi_element import RoiElement
from lena.elements.objects.element import Element
from lena.elements.objects.listbox_element import ListBoxElement
from lena.utils.config_manager import ConfigManager
from lena.utils.image_processing import filters_helper, morphological_helpers


class ListboxPreprocessing:
    def __init__(self, model_for_common_elements, model_for_listbox, common_element_features,
                 scroll_buttons_patterns, shift_threshold_for_scrolls):
        self.model_for_common_elements = model_for_common_elements
        self.__model_for_listbox = model_for_listbox
        self.__scroll_element_features = ScrollElementDetectionsFeatures(common_element_features,
                                                                         scroll_buttons_patterns,
                                                                         shift_threshold_for_scrolls)

    def image_processing_for_listbox(self, image):
        grey_ = filters_helper.convert_to_grayscale(image)
        grey_ = filters_helper.LevelsCorrection(grey_, ConfigManager().config.elements_parameters.listbox.preprocessing["level_correction_1"])
        grey_ = filters_helper.LevelsCorrection(grey_, ConfigManager().config.elements_parameters.listbox.preprocessing["level_correction_2"])
        grey_ = morphological_helpers.erosion(grey_)
        grey_ = filters_helper.LevelsCorrection(grey_, ConfigManager().config.elements_parameters.listbox.preprocessing["level_correction_3"])
        grey_ = filters_helper.LevelsCorrection(grey_, ConfigManager().config.elements_parameters.listbox.preprocessing["level_correction_4"])

        ret, grey_ = filters_helper.threshold(grey_, ConfigManager().config.elements_parameters.listbox.preprocessing["threshold_min"],
                                              ConfigManager().config.elements_parameters.listbox.preprocessing["threshold_max"])

        return grey_

    def prepare_features_for_listbox(self, image):
        grey_ = filters_helper.convert_to_grayscale(image)
        colours = filters_helper.calculate_white_colour(grey_)
        prepared_img = self.image_processing_for_listbox(grey_)

        #general_helpers.show(prepared_img)

        train_image_dimension = ConfigManager().config.elements_parameters.listbox.preprocessing["sample_dimension"]
        resized_prepared_img = np.array(PIL_Image.fromarray(prepared_img).resize(tuple(train_image_dimension), PIL_Image.BICUBIC))

        scaler = MinMaxScaler()
        resized_prepared_img = scaler.fit_transform(resized_prepared_img)

        image_features = np.array(resized_prepared_img)
        image_features = image_features.reshape(image_features.shape[0], -1).flatten()

        concatenated_features = np.concatenate((image_features, colours))

        return concatenated_features
    