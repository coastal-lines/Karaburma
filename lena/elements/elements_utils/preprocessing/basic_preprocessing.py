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
