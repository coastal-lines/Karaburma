from typing import Sequence

import cv2
import numpy as np
from skimage import exposure
from skimage import feature

from karaburma.utils.image_processing import filters_helper


def compute_hog(image: np.ndarray) -> Sequence[float]:
    win_size = (8, 8)  # Choose an appropriate window size
    block_size = (2, 2)  # Choose an appropriate block size
    block_stride = (2, 2)
    cell_size = (2, 2)
    num_bins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    hog_features = hog.compute(image)

    return hog_features

def compute_hog_and_draw_on_image(image: np.ndarray) -> np.ndarray:
    hog_features_vector, hog_image = feature.hog(image,
                                                 orientations=9,
                                                 pixels_per_cell=(8, 8),
                                                 cells_per_block=(2, 2),
                                                 transform_sqrt=True,
                                                 block_norm="L1",
                                                 visualize=True)
    hog_image = exposure.rescale_intensity(hog_image, out_range=(0, 255))

    return filters_helper.convert_image_to_negative(hog_image.astype("uint8"))

def compare_hog_features(hog_features1: np.ndarray, hog_features2: np.ndarray) -> float:
    """
    :param hog_features1:
    :param hog_features2:
    :return: Calculate the cosine similarity between the two HOG feature vectors
    """
    return cv2.compareHist(hog_features1, hog_features2, cv2.HISTCMP_CORREL)