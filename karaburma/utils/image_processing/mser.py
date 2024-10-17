import cv2


def get_regions_and_bounding_Boxes_by_MSER(image):
    img = image.copy()
    mser = cv2.MSER_create()
    regions, boundingBoxes = mser.detectRegions(img)

    return regions, boundingBoxes


def draw_rectangles_for_MSER(boundingBoxes, image):
    for box in boundingBoxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return image