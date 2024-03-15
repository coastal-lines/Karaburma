import cv2
import numpy as np

from karaburma.elements.objects.roi_element import RoiElement
from karaburma.utils import general_helpers
from karaburma.utils.ocr import ocr_helper


class RadioButtonElementFeatures():
    def find_roi_for_element(self, screenshot_elements):
        list_of_roi = []

        rects_for_circles = []
        gr = cv2.cvtColor(screenshot_elements.get_current_image_source(), cv2.COLOR_BGR2GRAY)

        circle_rectangles = []
        circles = cv2.HoughCircles(gr, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=30, minRadius=1, maxRadius=10)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                r = r + 3
                circle_rectangles.append((x - r, y - r, x + r, y + r))

        for circle in circle_rectangles:
            roi = general_helpers.get_roi(screenshot_elements.get_current_image_source(), circle[0], circle[1], circle[2] + 100, circle[3])
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, text_data = ocr_helper.get_text_and_text_data(gray, "--psm 10 --oem 3")

            # Step 3: Text Contour Extraction
            the_largest_W = 0
            for i, conf in enumerate(text_data['conf']):
                # Filter out non-text regions
                if conf > 0:
                    x = text_data['left'][i]
                    y = text_data['top'][i]
                    w = text_data['width'][i]
                    h = text_data['height'][i]
                    if(the_largest_W < x + w):
                        the_largest_W = x + w

            # If text was not found
            if(the_largest_W == 0):
                the_largest_W = 100

            rects_for_circles.append((circle[0], circle[1], circle[0] + the_largest_W, circle[3]))

        for rect in rects_for_circles:
            shift = 1  # 0 - it discard white borders
            x = rect[0] - shift
            y = rect[1]
            w = (rect[2] - rect[0]) - shift
            h = (rect[3] - rect[1]) + shift

            temp_image = screenshot_elements.get_current_image_source()[y:y + h, x:x + w, :]
            list_of_roi.append(RoiElement(temp_image, x, y, w, h, "radiobutton"))

        return list_of_roi