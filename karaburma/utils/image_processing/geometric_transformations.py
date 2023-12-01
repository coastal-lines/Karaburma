import cv2

def turn_horizontal(image):
    return cv2.flip(image, 1)

def turn_vertical(image):
    return cv2.flip(image, 0)

def turn_horizontal_and_vertical(image):
    return cv2.flip(cv2.flip(image, 1), 0)

def turn_left(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)