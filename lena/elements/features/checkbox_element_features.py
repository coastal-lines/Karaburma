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

    def try_to_find_text(self, image, squares):
        #pytesseract.pytesseract.tesseract_cmd = 'c:\\Users\\User\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
        #text = pytesseract.image_to_string(gray, lang='eng', config="--psm 10 --oem 3")
        #text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

        final_rects = []
        for square in squares:
            roi = general_helpers.get_roi(image, square[0], square[1], square[2] + 100, square[3])

            # image = cv2.imread(r"F:\Data\Work\OwnProjects\Python\Demo_Screens\2_4.bmp")
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            pytesseract.pytesseract.tesseract_cmd = 'c:\\Users\\User\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
            text = pytesseract.image_to_string(gray, lang='eng', config="--psm 6 --oem 3")
            # text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang='eng', config="--psm 10 --oem 3")

            # Step 3: Text Contour Extraction
            the_extreme_right_point = 0
            #contours = []
            for i, conf in enumerate(text_data['conf']):
                if conf > 0:  # Filter out non-text regions
                    x = text_data['left'][i]
                    y = text_data['top'][i]
                    w = text_data['width'][i]
                    h = text_data['height'][i]
                    #if (w < 250 and w > 10 and h < 40 and h > 5):
                    #contours.append((x, y, w, h))

                    if (the_extreme_right_point < x + w):
                        the_extreme_right_point = x + w

                    #если текст не был найден
                    if (the_extreme_right_point == 0):
                        the_extreme_right_point = 100

            #rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in contours])
            # without strong text borders checking - just plus W and H to curent checkbox
            #if (len(rects) > 0):
                # final_rects.append((square[0], square[1], rects[-1][0] + rects[-1][2], rects[-1][1] + rects[-1][3]))

            #ПРОБА! - добавлен небольшой дополнителньый сдвиг
            final_rects.append((square[0] - 2, square[1] - 2, square[0] + the_extreme_right_point - 2, square[3] + 2))

        #contours_helper.DrawRectangleByRectangles(image, final_rects[:1])
        #general_helpers.show(image)

        return final_rects

    #not used for simple checkbox solution
    def get_rectangle_centre(self, rect):
        x_centre = (rect[0] + rect[2]) / 2
        y_centre = (rect[1] + rect[3]) / 2
        return x_centre, y_centre
