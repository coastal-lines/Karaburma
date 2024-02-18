import cv2
import numpy as np
from PIL import Image as PIL_Image
from sklearn.preprocessing import MinMaxScaler

from karaburma.elements.features.scroll_element_features import ScrollElementDetectionsFeatures
from karaburma.elements.objects.roi_element import RoiElement
from karaburma.elements.objects.element import Element
from karaburma.elements.objects.listbox_element import ListBoxElement
from karaburma.utils.config_manager import ConfigManager
from karaburma.utils.image_processing import filters_helper, morphological_helpers


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
        grey_ = filters_helper.levels_correction(grey_, ConfigManager().config.elements_parameters.listbox.preprocessing["level_correction_1"])
        grey_ = filters_helper.levels_correction(grey_, ConfigManager().config.elements_parameters.listbox.preprocessing["level_correction_2"])
        grey_ = morphological_helpers.erosion(grey_)
        grey_ = filters_helper.levels_correction(grey_, ConfigManager().config.elements_parameters.listbox.preprocessing["level_correction_3"])
        grey_ = filters_helper.levels_correction(grey_, ConfigManager().config.elements_parameters.listbox.preprocessing["level_correction_4"])

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

    def get_contours_for_listbox(self, image: np.array):
        results = []

        grey_ = self.image_processing_for_listbox(image)

        min_w = ConfigManager().config.elements_parameters.listbox.contours_parameters["min_w"]
        max_w = ConfigManager().config.elements_parameters.listbox.contours_parameters["max_w"]
        min_h = ConfigManager().config.elements_parameters.listbox.contours_parameters["min_h"]
        max_h = ConfigManager().config.elements_parameters.listbox.contours_parameters["max_h"]
        contours, hierarchy = cv2.findContours(grey_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if (min_w < w < max_w and min_h < h < max_h):
                parent_index = hierarchy[0][i][3]
                #print(parent_index)

                # the parent contour may turn out to be the global contour, i.e., it can be excluded
                # check the parent even if its parent index is not equal to "-1"
                if(parent_index != -1):
                    temp_parent_x, temp_parent_y, temp_parent_w, temp_parent_h = cv2.boundingRect(contours[parent_index])
                    if(temp_parent_w > w * 1.5 or temp_parent_h > h * 1.5):
                        results.append((x, y, w, h))
                        #cv2.drawContours(screenshot, [contours[i]], -1, (0, 255, 0), 1)

        #general_helpers.show(screenshot)

        #DEBUG - most likely no need to filter. Cleared through contours.
        #filtered_contours = contours_helper.remove_similar_contours(results, 0.1)
        filtered_contours = results

        return filtered_contours

    def get_label_for_roi(self, cnt, image, shift):
        x, y, w, h = cnt[0], cnt[1], cnt[2], cnt[3]
        roi = image[y - shift:y + h + shift, x - shift:x + w + shift, :]
        concatenated_features = self.prepare_features_for_listbox(roi)

        predictions = self.__model_for_listbox.predict([concatenated_features])
        predictions_proba = self.__model_for_listbox.predict_proba([concatenated_features])
        unique_labels, counts = np.unique(predictions, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]

        return most_common_label, predictions_proba

    def prepare_roi_element(self, cnt, image, shift):
        x_ = cnt[0] + shift
        y_ = cnt[1] + shift
        x2_ = cnt[0] + cnt[2] - shift
        y2_ = cnt[1] + cnt[3] - shift
        w_ = x2_ - x_
        h_ = y2_ - y_
        roi_without_shift = image[y_:y2_, x_:x2_, :]
        temp_roi = RoiElement(roi_without_shift, x_, y_, w_, h_)

        return temp_roi

    def listbox_element_classification(self, image: np.array):
        list_listboxes = []

        contours = self.get_contours_for_listbox(image)
        shift = 3 #TODO - move to config

        for cnt in contours:
            most_common_label, predictions_proba = self.get_label_for_roi(cnt, image, shift)

            if (most_common_label == 0):
                temp_roi = self.prepare_roi_element(cnt, image, shift)
                temp_listbox = ListBoxElement("listbox", predictions_proba[0][0], temp_roi)

                _, v_scroll = self.__scroll_element_features.find_scrolls(temp_listbox.get_roi_element())
                if v_scroll is not None:
                    temp_listbox.add_scroll("v_scroll", v_scroll)

                    text_area = temp_listbox.get_roi_element().get_roi()[:, 0:temp_listbox.get_roi_element().get_w() - v_scroll.get_roi_element().get_w(), :]
                    temp_listbox.textarea = Element("listbox",
                                                    1.0,
                                                    RoiElement(text_area, cnt[0] + shift, cnt[1] + shift,
                                                               temp_listbox.get_roi_element().get_w() - v_scroll.get_roi_element().get_w(),
                                                               temp_listbox.get_roi_element().get_h()))

                list_listboxes.append(temp_listbox)

        return list_listboxes
