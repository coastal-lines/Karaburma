import cv2
import numpy as np
import pytesseract
import skimage as sk
import imutils.object_detection
from skimage.filters import threshold_local
from karaburma.utils import general_helpers
from karaburma.utils.image_processing import filters_helper, contours_helper
from karaburma.elements.objects.roi_element import RoiElement

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

    #not used for simple checkbox solution
    def combine_text_rectangles(self, squares, text_rects):

        final_checkbox_rect = []

        temp_current_rectangles = []
        for square in squares:
            for i in range(len(text_rects)):
                x_square_centre, y_square_centre = self.get_rectangle_centre(square)
                x_text_centre, y_text_centre = self.get_rectangle_centre(text_rects[i])

                x_difference = abs(x_text_centre - x_square_centre)
                y_difference = abs(y_text_centre - y_square_centre)

                if (x_difference < 100 and y_difference < 10):
                    temp_current_rectangles.append(square)
                    temp_current_rectangles.append(text_rects[i])

            if (len(temp_current_rectangles) > 0):
                x1, y1, x2, y2 = temp_current_rectangles[0][0], square[1] - 2, temp_current_rectangles[0][2], square[3] + 2
                np_temp_current_rectangles = np.array(temp_current_rectangles)
                for rect in np_temp_current_rectangles:
                    if (rect[0] < x1):
                        x1 = rect[0]
                    if (rect[2] > x2):
                        x2 = rect[2]

                final_checkbox_rect.append((x1, y1, x2, y2))
                print((x1, y1, x2, y2))
                temp_current_rectangles.clear()

        return final_checkbox_rect

    def find_contours_for_checkbox_elements(self, screenshot_elements):

        list_of_roi = []

        squares = self.try_to_find_squares(screenshot_elements.get_current_image_source())
        text_rects = self.try_to_find_text(screenshot_elements.get_current_image_source(), squares)
        #final_checkbox_rect = self.combine_text_rectangles(squares, text_rects)

        #for rect in final_checkbox_rect:
        for rect in text_rects:
            #shift = 2
            shift = 0 #it discard white borders

            x = rect[0]
            y = rect[1]
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]

            temp_image = screenshot_elements.get_current_image_source()[y - 2:y + h + 2, x - 2:x + w + 2, :]
            temp_image_with_board = np.ones((h + (shift * 2) + 4, w + (shift * 2) + 4, 3), dtype=np.uint8) * 255
            temp_image_with_board[shift:temp_image_with_board.shape[0] - shift, shift:temp_image_with_board.shape[1] - shift, :] = temp_image
            #general_helpers.show(temp_image_with_board)

            #screenshot_elements.add_roi(RoiElement(temp_image_with_board, x, y, w, h, "checkbox"))
            list_of_roi.append(RoiElement(temp_image_with_board, x, y, w, h, "checkbox"))

        return list_of_roi