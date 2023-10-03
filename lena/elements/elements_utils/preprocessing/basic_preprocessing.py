import numpy as np
import skimage
import skimage as sk
import cv2
from skimage.filters import threshold_mean

from lena.elements.objects.roi_element import RoiElement
from lena.utils.config_manager import ConfigManager
from lena.utils.image_processing import filters_helper, morphological_helpers, contours_helper, harris


class BasicPreprocessing:

    def prepare_image(self, image_source):
        gr = filters_helper.convert_to_grayscale(image_source.get_current_image_source())
        sh = filters_helper.Sharp(gr, "strong")
        er = morphological_helpers.erosion(sh)
        th = er.copy() > sk.filters.threshold_local(er,
                                                    block_size=ConfigManager().config.
                                                    elements_parameters.common_element.
                                                    preprocessing.contours_parameters["threshold_block_size"],
                                                    offset=ConfigManager().config.elements_parameters.common_element.
                                                    preprocessing.contours_parameters["threshold_offset"])
        th = sk.img_as_ubyte(th)

        return th

    def load_parameters(self):
        min_w = ConfigManager().config.elements_parameters.common_element.preprocessing.contours_parameters["min_w"]
        min_h = ConfigManager().config.elements_parameters.common_element.preprocessing.contours_parameters["min_h"]
        max_w = ConfigManager().config.elements_parameters.common_element.preprocessing.contours_parameters["max_w"]
        max_h = ConfigManager().config.elements_parameters.common_element.preprocessing.contours_parameters["max_h"]
        return min_w, min_h, max_w, max_h

    def prepare_contours(self, contours):
        min_w, min_h, max_w, max_h = self.load_parameters()

        result_contours = []
        result_rectangles = []
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if ((w > h) and (w > min_w) and (h < max_h) and (h > min_h) and (w < max_w)):
                result_contours.append(contours[i])
                result_rectangles.append((x, y, w, h))

        result_contours = contours_helper.filter_very_similar_contours(result_rectangles)

        return result_contours

    def find_contours_for_common_elements(self, image_source):
        list_of_roi = []

        th = self.prepare_image(image_source)
        contours, hierarchy = contours_helper.get_contours(th)
        result_contours = self.prepare_contours(contours)

        shift = ConfigManager().config.elements_parameters.common_element.preprocessing.contours_parameters["roi_shift"]
        for i in range(len(result_contours)):
            x, y, w, h = result_contours[i][0], result_contours[i][1], result_contours[i][2], result_contours[i][3]
            temp_image = image_source.get_current_image_source()[y:y + h, x:x + w, :]
            temp_image_with_board = np.ones((h + (shift * 2), w + (shift * 2), 3), dtype=np.uint8) * 255
            temp_image_with_board[shift:temp_image_with_board.shape[0] - shift,
            shift:temp_image_with_board.shape[1] - shift, :] = temp_image

            list_of_roi.append(RoiElement(temp_image_with_board, x, y, w, h))

        return list_of_roi

    def prepare_features_for_basic_elements(self, roi, dim):
        #number of harris coordinates for each sample
        harris_array = ConfigManager().config.elements_parameters.common_element.preprocessing["harris_array_size"]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, dim)
        mean_binary = gray.copy() > threshold_mean(gray)
        mean_binary = skimage.img_as_ubyte(mean_binary)
        coordinates = harris.applay_harris_and_get_coordinates(mean_binary)
        coords = np.zeros((harris_array, 2))
        #fill coordinates according "harris_array_size" parameter
        coords[:coordinates.shape[0], :] = coordinates[0:harris_array, :]

        image_flat = gray.flatten()
        coords_flat = coords.flatten()

        feature = []
        for th in image_flat:
            feature.append(th)

        for coord in coords_flat:
            feature.append(coord)

        return feature