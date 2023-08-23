import cv2
import numpy as np
import pytesseract
import skimage as sk
import imutils.object_detection
from skimage.filters import threshold_local
from lena.utils import general_helpers
from lena.utils.image_processing import filters_helper, contours_helper
from lena.elements.objects.roi_element import RoiElement

class CheckboxElementFeatures():

    def try_to_find_squares(self, image):
        min_w = 7
        min_h = 7
        max_w = 20
        max_h = 20

        gr = filters_helper.convert_to_grayscale(image)
        er = filters_helper.Sharp(gr, "strong")
        th = er.copy() > threshold_local(er, block_size=5, offset=3)
        th = sk.img_as_ubyte(th)
        contours, hierarchy = contours_helper.get_contours(th)

        valid_squares = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                if ((w > min_w) and (w < max_w) and (h > min_h) and (h < max_h)):
                    aspect_ratio = w / float(h)

                    # Check if the aspect ratio is close to 1 (a square)
                    if 0.9 < aspect_ratio < 1.1:
                        points = np.array(approx)
                        x, y, w, h = cv2.boundingRect(points)
                        valid_squares.append((x, y, w, h))

        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in valid_squares])
        pick = imutils.object_detection.non_max_suppression(rects, probs=None, overlapThresh=0.5)

        # contours_helper.DrawRectangleByRectangles(image, pick)
        # general_helpers.show(image)
        return pick
