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
from lena.elements.objects.roi_element import RoiElement
from lena.elements.objects.screenshot_element import ImageSourceObject
from lena.utils.config_manager import ConfigManager
from lena.utils.image_processing import filters_helper, contours_helper, morphological_helpers
from lena.utils import data_normalization, general_helpers


class TablePreprocessing:
    def __init__(self):
        self.image_source = None

    @property
    def image_source(self):
        return self.__image_source

    @image_source.setter
    def image_source(self, screenshot: ImageSourceObject):
        print("Setting value")
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

        #cv2.imwrite("Projects\\test6" + "\\" + f, otsu_binary)
        #general_helpers.show(otsu_binary)

        harris_corners = cv2.cornerHarris(preprocessed_image, 2, 3, 0.04)
        threshold = 0.01 * harris_corners.max()
        corner_response_thresholded = harris_corners > threshold
        current_points = np.argwhere(corner_response_thresholded)

        #points.append(len(current_points))
        #print("Points: " + str(len(current_points)))

        num_clusters = 16
        # random_state=102 - It is important because, for tasks like Harris corner detection and table recognition,
        # the accuracy of contour identification is critical.
        # Inconsistent or excessive points in the contours can lead to inaccurate results,
        # affecting the performance of algorithms that rely on precise contour information.
        kmeans = KMeans(n_clusters=num_clusters, random_state=102)

        try:
            kmeans.fit(current_points)
            cluster_centers = kmeans.cluster_centers_.astype(int)
            #print("Centres: " + str(len(cluster_centers)))

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
            #cv2.imwrite("Projects\\test6" + "\\" + image_path.split("\\")[-1], otsu_binary)
            combined_array = [np.array([0, 0]) for _ in range(num_clusters)]

        scaler = MinMaxScaler()
        combined_array = scaler.fit_transform(combined_array)

        return combined_array
