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
