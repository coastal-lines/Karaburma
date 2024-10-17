import cv2
import numpy as np
import skimage as sk
import imutils.object_detection
from skimage.filters import threshold_local
from karaburma.utils import general_helpers
from karaburma.utils.ocr import ocr_helper
from karaburma.utils.image_processing import filters_helper, contours_helper
from karaburma.elements.objects.roi_element import RoiElement


class CheckboxElementFeatures():
    def try_to_find_squares(self, image):
        min_w = 7
        min_h = 7
        max_w = 20
        max_h = 20

        gr = filters_helper.convert_to_grayscale(image)
        er = filters_helper.sharp(gr, "strong")
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
        return pick

    def try_to_find_text(self, image, squares):
        final_rects = []
        for square in squares:
            roi = general_helpers.get_roi(image, square[0], square[1], square[2] + 100, square[3])
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            _, text_data = ocr_helper.get_text_and_text_data(gray, "--psm 6 --oem 3")

            # Step 3: Text Contour Extraction
            the_extreme_right_point = 0
            for i, conf in enumerate(text_data['conf']):
                if conf > 0:  # Filter out non-text regions
                    x = text_data['left'][i]
                    y = text_data['top'][i]
                    w = text_data['width'][i]
                    h = text_data['height'][i]

                    if (the_extreme_right_point < x + w):
                        the_extreme_right_point = x + w

                    # If text was not found
                    if (the_extreme_right_point == 0):
                        the_extreme_right_point = 100

            # Add additional little shift
            final_rects.append((square[0] - 2, square[1] - 2, square[0] + the_extreme_right_point - 2, square[3] + 2))

        return final_rects

    # Not used for simple checkbox solution
    def get_rectangle_centre(self, rect):
        x_centre = (rect[0] + rect[2]) / 2
        y_centre = (rect[1] + rect[3]) / 2
        return x_centre, y_centre

    # Not used for simple checkbox solution
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

        # For rect in final_checkbox_rect:
        for rect in text_rects:
            shift = 0 #it discard white borders

            x = rect[0]
            y = rect[1]
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]

            temp_image = screenshot_elements.get_current_image_source()[y - 2:y + h + 2, x - 2:x + w + 2, :]
            temp_image_with_board = np.ones((h + (shift * 2) + 4, w + (shift * 2) + 4, 3), dtype=np.uint8) * 255
            temp_image_with_board[shift:temp_image_with_board.shape[0] - shift, shift:temp_image_with_board.shape[1] - shift, :] = temp_image

            list_of_roi.append(RoiElement(temp_image_with_board, x, y, w, h, "checkbox"))

        return list_of_roi
