import math
import cv2
import skimage
import numpy as np
import pandas as pd
from PIL import Image as PIL_Image
from skimage.filters import threshold_otsu
from skimage.util import img_as_ubyte
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from karaburma.elements.objects.roi_element import RoiElement
from karaburma.elements.objects.screenshot_element import ImageSourceObject
from karaburma.utils.config_manager import ConfigManager
from karaburma.utils.image_processing import filters_helper, contours_helper, morphological_helpers
from karaburma.utils import data_normalization


class TablePreprocessing:
    def __init__(self):
        self.image_source = None

    @property
    def image_source(self):
        return self.__image_source

    @image_source.setter
    def image_source(self, screenshot: ImageSourceObject):
        self.__image_source = screenshot

    def __resize_source_image_and_apply_filter(self, roi):

        grey_ = filters_helper.convert_to_grayscale(roi)
        imnp = np.array(grey_) / 255
        gamma = math.log(imnp.mean()) / math.log(0.1)
        new = ((imnp ** (1 / gamma)) * 255).astype(np.uint8)
        new = np.array(PIL_Image.fromarray(new).resize(ConfigManager().config.elements_parameters.table["fixed_size_for_preprocessing"], PIL_Image.BICUBIC))
        er = img_as_ubyte(morphological_helpers.erosion(new))
        er = morphological_helpers.erosion(er)
        dl = morphological_helpers.dilation(er)

        return dl

    def __find_structure(self, resized_image, features_size):
        otsu_binary = skimage.img_as_ubyte(resized_image.copy() > threshold_otsu(resized_image))
        contours, _ = cv2.findContours(otsu_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (w < len(otsu_binary[0]) // 2):
                otsu_binary[y:y + h, x:x + w] = 255

        otsu_binary = otsu_binary[30:len(otsu_binary[1]) - 60, 30:len(otsu_binary[0]) - 60]
        otsu_binary_as_feature = np.array(PIL_Image.fromarray(otsu_binary).resize(features_size, PIL_Image.BICUBIC))

        return otsu_binary_as_feature

    def __table_image_processing_otsu16(self, roi):
        resized_image = self.__resize_source_image_and_apply_filter(roi)
        preprocessed_image = self.__find_structure(resized_image, (16, 16))
        scaled_image = data_normalization.scaling_by_min_max(preprocessed_image)

        return scaled_image

    def __table_image_processing(self, roi):
        points = []

        resized_image = self.__resize_source_image_and_apply_filter(roi)
        preprocessed_image = self.__find_structure(resized_image, (800, 800))

        harris_corners = cv2.cornerHarris(preprocessed_image, 2, 3, 0.04)
        threshold = 0.01 * harris_corners.max()
        corner_response_thresholded = harris_corners > threshold
        current_points = np.argwhere(corner_response_thresholded)

        num_clusters = 16
        # random_state=102 - It is important because, for tasks like Harris corner detection and table recognition,
        # the accuracy of contour identification is critical.
        # Inconsistent or excessive points in the contours can lead to inaccurate results,
        # affecting the performance of algorithms that rely on precise contour information.
        kmeans = KMeans(n_clusters=num_clusters, random_state=102)

        try:
            kmeans.fit(current_points)
            cluster_centers = kmeans.cluster_centers_.astype(int)

            X = np.sort(cluster_centers[:, 0])
            Y = np.sort(cluster_centers[:, 1])

            original_min = np.min(X)
            original_max = np.max(X)
            desired_min = 0
            desired_max = 800
            X_converted_values = (X - original_min) * (desired_max - desired_min) / (
                        original_max - original_min) + desired_min

            original_min = np.min(Y)
            original_max = np.max(Y)
            desired_min = 0
            desired_max = 800
            Y_converted_values = (Y - original_min) * (desired_max - desired_min) / (original_max - original_min) + desired_min

            combined_array = np.column_stack((X_converted_values, Y_converted_values))

        except:
            print("Error. No cluster centres for: ")
            combined_array = [np.array([0, 0]) for _ in range(num_clusters)]

        scaler = MinMaxScaler()
        combined_array = scaler.fit_transform(combined_array)

        return combined_array

    def __find_best_contours_for_table(self):
        list_of_roi = []

        grey_ = filters_helper.convert_to_grayscale(self.image_source.get_current_image_source())
        grey_ = filters_helper.levels_correction(grey_, ConfigManager().config.elements_parameters.table.preprocessing["level_correction_1"])
        kernel1 = np.array(ConfigManager().config.elements_parameters.table.preprocessing["kernel1"])
        grey_ = cv2.filter2D(src=grey_, ddepth=-1, kernel=kernel1)
        grey_ = morphological_helpers.dilation(grey_)

        # Get contours
        contours, hierarchy = contours_helper.get_contours_by_canny(grey_, 0, 255, True)
        filtered_contours = contours

        # Adjust this threshold according to your specific use case
        rectangles = []
        for i in range(len(filtered_contours)):
            x, y, w, h = cv2.boundingRect(filtered_contours[i])
            rectangles.append((x, y, w, h))

        min_w = ConfigManager().config.elements_parameters.table.contours_parameters["min_w"]
        max_w = ConfigManager().config.elements_parameters.table.contours_parameters["max_w"]
        min_h = ConfigManager().config.elements_parameters.table.contours_parameters["min_h"]
        max_h = ConfigManager().config.elements_parameters.table.contours_parameters["max_h"]

        result_rectangles = []
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])

            if min_w < w < max_w and min_h < h < max_h:
                result_rectangles.append((x, y, w, h))

        if(len(result_rectangles) > 0):
            contours_threshold_for_x = ConfigManager().config.elements_parameters.table.contours_parameters["contours_threshold_for_x"]
            groups_x = []
            df = pd.DataFrame(result_rectangles)
            sorted_df = df.sort_values(by=0)
            groups_x.append(sorted_df.iloc[0, 0])
            for row_index in range(1, sorted_df.shape[0], 1):
                if(groups_x[len(groups_x) - 1] + contours_threshold_for_x < sorted_df.iloc[row_index, 0]):
                    groups_x.append(sorted_df.iloc[row_index, 0])

            super_result_rectangles = []
            for i in range(len(groups_x)):
                for row_index in range(0, sorted_df.shape[0], 1):
                    if(groups_x[i] == sorted_df.iloc[row_index, 0]):
                        super_result_rectangles.append(list(sorted_df.iloc[row_index, :]))
                        break

            if(len(super_result_rectangles) > 0):
                # cut images
                for i in range(len(super_result_rectangles)):
                    shift = ConfigManager().config.elements_parameters.table.contours_parameters["roi_shift"]

                    x = super_result_rectangles[i][0]
                    y = super_result_rectangles[i][1]
                    w = super_result_rectangles[i][2]
                    h = super_result_rectangles[i][3]

                    if(y - shift > 0 and x - shift > 0):
                        temp_image = self.image_source.get_current_image_source()[y - shift:y + h + shift, x - shift:x + w + shift, :]
                        list_of_roi.append(RoiElement(temp_image, x, y, w, h))
                    else:
                        temp_image = self.image_source.get_current_image_source()[y:y + h + shift, x:x + w + shift, :]
                        list_of_roi.append(RoiElement(temp_image, x, y, w, h))
        else:
            print("Potential table was not found")

        return list_of_roi

    def prepare_features_for_table_image(self, roi: np.ndarray):
        image_as_scaled_data = self.__table_image_processing_otsu16(roi)
        colours = filters_helper.calculate_white_colour(roi)
        filtered_centers = self.__table_image_processing(roi)
        filtered_centers = np.squeeze(filtered_centers.reshape((filtered_centers.shape[0], -1))).flatten()

        for j in range(len(filtered_centers)):
            if (math.isnan(filtered_centers[j])):
                print("NaN for: " + str(" roi"))
                filtered_centers[j] = 0.0

        image_features = np.array(image_as_scaled_data)
        image_features = image_features.reshape(image_features.shape[0], -1).flatten()
        concatenated_features = np.concatenate((filtered_centers, image_features, colours))

        return concatenated_features

    def table_element_classification(self, tables_model, image_source):
        self.image_source = image_source

        all_tables = []

        list_of_roi = self.__find_best_contours_for_table()

        #find tables
        for i in range(len(list_of_roi)):
            concatenated_features = self.prepare_features_for_table_image(list_of_roi[i].get_roi())

            predictions = tables_model.predict([concatenated_features])
            predictions_proba = tables_model.predict_proba([concatenated_features])

            unique_labels, counts = np.unique(predictions, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]

            if (most_common_label == 0):
                list_of_roi[i].update_element_roi_area_by_image(self.image_source.get_current_image_source())
                all_tables.append((list_of_roi[i], predictions_proba[0][0]))

        #group similar tables
        threshold_distance_between_tables = ConfigManager().config.elements_parameters.table["threshold_distance_between_tables"]
        tables_groups = []
        for i in range(len(all_tables)):
            x,y,w,h = all_tables[i][0].get_element_features()
            rect = [x,y,w,h]
            centroid = contours_helper.get_rect_centroid(rect)

            # Find a group for the current rectangle
            found_group = False
            for table_group in tables_groups:
                x_, y_, w_, h_ = table_group[0][0].get_element_features()

                # Get centroid of the first rectangle in the group
                group_centroid = contours_helper.get_rect_centroid([x_, y_, w_, h_])

                distance = np.sqrt((centroid[0] - group_centroid[0]) ** 2 + (centroid[1] - group_centroid[1]) ** 2)
                if distance <= threshold_distance_between_tables:
                    table_group.append(all_tables[i])
                    found_group = True
                    break

            # If no suitable group is found, create a new group
            if not found_group:
                tables_groups.append([all_tables[i]])

        return tables_groups