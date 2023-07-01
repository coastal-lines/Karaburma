import cv2


def calculate_images_transformation(points1, points2, method = cv2.RANSAC):
    return cv2.findHomography(points1, points2, method)