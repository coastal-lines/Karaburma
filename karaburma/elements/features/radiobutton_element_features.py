import os

import cv2
import numpy as np
import pytesseract

from karaburma.elements.objects.roi_element import RoiElement
from karaburma.utils import general_helpers
from karaburma.utils import files_helper


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
                #circle_rectangles.append((x - r, y - r, x + r, y + r))
                r = r + 3
                circle_rectangles.append((x - r, y - r, x + r, y + r))

        for circle in circle_rectangles:
            roi = general_helpers.get_roi(screenshot_elements.get_current_image_source(), circle[0], circle[1], circle[2] + 100, circle[3])
            #general_helpers.show(roi)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            pytesseract.pytesseract.tesseract_cmd = os.path.join(os.path.expanduser('~\\AppData'), "Local\\Programs\\Tesseract-OCR\\tesseract.exe")
            text = pytesseract.image_to_string(gray, lang='eng', config="--psm 10 --oem 3")
            text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang='eng', config="--psm 10 --oem 3")

            # Step 3: Text Contour Extraction
            #contours = []
            the_largest_W = 0
            for i, conf in enumerate(text_data['conf']):
                if conf > 0:  # Filter out non-text regions
                    x = text_data['left'][i]
                    y = text_data['top'][i]
                    w = text_data['width'][i]
                    h = text_data['height'][i]
                    if(the_largest_W < x + w):
                        the_largest_W = x + w

                    #contours_helper.DrawRectangle(screen.screen_image, circle[0]+x, circle[1]+y, circle[0]+x + w, circle[1]+y+h)
            #general_helpers.show(screen.screen_image)
            #print(text, the_largest_W)

            #если текст не был найден
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
            #general_helpers.show(temp_image)
            #temp_image_with_board = np.ones((h + (shift * 2) + 0, w + (shift * 2) + 0, 3), dtype=np.uint8) * 255
            #temp_image_with_board[shift:temp_image_with_board.shape[0] - shift,
            #shift:temp_image_with_board.shape[1] - shift, :] = temp_image

            list_of_roi.append(RoiElement(temp_image, x, y, w, h, "radiobutton"))
            #files_helper.save_image(temp_image)
            #general_helpers.show(temp_image)

        return list_of_roi