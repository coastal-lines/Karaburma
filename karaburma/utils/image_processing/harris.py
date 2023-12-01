import cv2
import numpy as np


def applay_harris_and_get_coordinates(img_grayscale: np.array) -> np.array:
    harris_corners = cv2.cornerHarris(img_grayscale, 2, 3, 0.04)
    threshold = 0.01 * harris_corners.max()
    indices = np.where(harris_corners > threshold)
    rows = indices[0]
    cols = indices[1]
    coordinates = np.column_stack((cols, rows))

    return coordinates