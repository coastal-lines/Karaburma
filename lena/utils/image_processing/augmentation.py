import numpy as np
from PIL import Image as PIL_Image

from lena.utils.image_processing import filters_helper


def extend_grayscale_roi(roi, left=10, right=10, top=10, bottom=10, colour=255):
    roi = filters_helper.convert_to_grayscale(roi)

    extended_roi = np.ones((roi.shape[0] + top + bottom, roi.shape[1] + left + right)) * colour
    extended_roi[top : extended_roi.shape[0] - bottom, left : extended_roi.shape[1] - right] = roi

    return extended_roi

def bicubic_resize(roi, size):
    new = np.array(PIL_Image.fromarray(roi).resize(size, PIL_Image.BICUBIC))
    return new