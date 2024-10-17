import numpy as np
import cv2


def draw_points_on_empty_image(image: np.ndarray, points, radius=6, color=(255, 0, 0)) -> np.ndarray:
    empty_image_for_drawing = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
    for point in points:
        cv2.circle(empty_image_for_drawing, (point[0], point[1]), radius, color, -1)

    return empty_image_for_drawing

