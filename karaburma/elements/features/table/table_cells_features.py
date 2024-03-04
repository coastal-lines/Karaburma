import cv2
import numpy as np
import skimage
from skimage.filters import threshold_mean

from karaburma.elements.objects.table.table_cell_element import TableCell
from karaburma.elements.objects.table.table_cells_element import TableCellsElement
from karaburma.utils import general_helpers
from karaburma.utils.image_processing import filters_helper, contours_helper, morphological_helpers
from karaburma.elements.objects.roi_element import RoiElement


class TableCellsFeatures():
    def __preprocessing_for_table_cells(self, temp_table_roi):
        temp_table_roi_gr = filters_helper.convert_to_grayscale(temp_table_roi)

        filter_kernel = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [-1, -1, 1, -1, -1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 4, 0, 0]])

        temp_table_roi_gr = cv2.filter2D(src=temp_table_roi_gr, ddepth=-1, kernel=filter_kernel)
        temp_table_roi_gr = morphological_helpers.erosion(temp_table_roi_gr)
        temp_table_roi_gr = morphological_helpers.dilation(temp_table_roi_gr)
        temp_table_roi_gr = filters_helper.levels_correction(temp_table_roi_gr, 86, 159, 0, 255, 0.04)
        temp_table_roi_gr = cv2.medianBlur(temp_table_roi_gr, 7)

        return temp_table_roi_gr

    def __remove_noise_from_table(self, prepared_table_roi, original_table_roi):
        all, _ = contours_helper.get_contours_by_canny(prepared_table_roi, 0, 255)
        original_table_roi = filters_helper.convert_to_grayscale(original_table_roi)

        for contour in all:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            x, y, w, h = cv2.boundingRect(approx)
            if (w > 3 and h > 3 and w < 100 and h < 30):
                original_table_roi[y - 0:y + h + 0, x - 0:x + w + 0] = 255

        return original_table_roi

    def __get_contours_for_table_cells(self, prepared_table_roi, original_table_roi):
        cleaned_table_roi = self.__remove_noise_from_table(prepared_table_roi, original_table_roi)

        grey_ = filters_helper.levels_correction(cleaned_table_roi, 190, 236, 0, 111, 0.38)
        grey_ = filters_helper.levels_correction(grey_, 121, 160, 98, 255, 0.4)
        grey_ = morphological_helpers.erosion(grey_)
        grey_ = skimage.img_as_ubyte(grey_.copy() > threshold_mean(grey_))

        contours, hierarchy = cv2.findContours(grey_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        temp_contours = []

        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)

            if(w > h and w > 60 and h > 7 and w < 400 and h < 200):
                temp_contours.append((x, y, w, h))

        return temp_contours

    def __calculate_most_frequent_cell_dimension(self, prepared_cells_contours):
        #find the most similar contours
        all_width = [arr[2] for arr in prepared_cells_contours]
        most_frequent_width = np.bincount(all_width).argmax()

        all_height = [arr[3] for arr in prepared_cells_contours]
        most_frequent_height = np.bincount(all_height).argmax()

        return most_frequent_width, most_frequent_height

    def __calculate_cells_coordinates(self, prepared_cells_contours, most_frequent_width, most_frequent_height):
        #get coordinates of table
        all_x = np.sort([arr[0] for arr in prepared_cells_contours])
        x1 = all_x[0]
        x2 = all_x[-1] + most_frequent_width

        all_y = np.sort([arr[1] for arr in prepared_cells_contours])
        y1 = all_y[0]
        y2 = all_y[-1] + most_frequent_height

        return x1, y1, x2, y2

    def __calculate_columns_and_rows_number(self, cells_area_x2, cells_area_y2, most_frequent_width, most_frequent_height):
        table_columns = cells_area_x2 // most_frequent_width
        table_rows = cells_area_y2 // most_frequent_height
        return table_columns, table_rows

    def __prepare_list_cells(self, temp_table_roi, most_frequent_width, most_frequent_height, prepared_cells_contours):
        list_cells = []

        threshold = 1

        shift_from_find_best_contours_for_table_method = 3

        column_index = 0
        row_index = -1
        temp_cell_y = -1

        prepared_cells_contours.reverse()
        for cell_contour in prepared_cells_contours:
            contour_x, contour_y, contour_w, contour_h = cell_contour[0], cell_contour[1], cell_contour[2], cell_contour[3]

            if(temp_cell_y == contour_y):
                column_index += 1
            else:
                column_index = 0
                row_index += 1

            current_cell = None

            if (most_frequent_height - threshold <= contour_h <= most_frequent_height + threshold and most_frequent_width - threshold <= contour_w <= most_frequent_width + threshold):
                #TODO - move into one method

                current_roi = temp_table_roi.get_roi()[contour_y:contour_y + contour_h, contour_x:contour_x + contour_w]
                absolute_current_x, absolute_current_y = general_helpers.calculate_absolute_coordinates(temp_table_roi, contour_x, contour_y)

                absolute_current_x = absolute_current_x - shift_from_find_best_contours_for_table_method
                absolute_current_y = absolute_current_y - shift_from_find_best_contours_for_table_method

                current_cell = TableCell("table_cell", 1.0, RoiElement(current_roi, absolute_current_x, absolute_current_y, contour_w, contour_h), "", column_index, row_index)
            else:
                if (contour_w > 60 and contour_h > 8):
                    # TODO - move into one method. Code the same
                    current_roi = temp_table_roi.get_roi()[contour_y:contour_y + contour_h, contour_x:contour_x + contour_w]
                    absolute_current_x, absolute_current_y = general_helpers.calculate_absolute_coordinates(temp_table_roi, contour_x, contour_y)

                    absolute_current_x = absolute_current_x - shift_from_find_best_contours_for_table_method
                    absolute_current_y = absolute_current_y - shift_from_find_best_contours_for_table_method

                    current_cell = TableCell("table_cell", 1.0, RoiElement(current_roi, absolute_current_x, absolute_current_y, contour_w, contour_h), "", column_index, row_index)

            if (current_cell != None):
                temp_cell_y = contour_y
                list_cells.append(current_cell)

        return list_cells

    def get_roi_rectangle_for_table_cells(self, temp_table_roi):
        prepared_table_roi = self.__preprocessing_for_table_cells(temp_table_roi.get_roi())
        prepared_cells_contours = self.__get_contours_for_table_cells(prepared_table_roi, temp_table_roi.get_roi())

        min_x = min(rectangle[0] for rectangle in prepared_cells_contours)
        min_y = min(rectangle[1] for rectangle in prepared_cells_contours)
        max_x = max(rectangle[0] + rectangle[2] for rectangle in prepared_cells_contours)
        max_y = max(rectangle[1] + rectangle[3] for rectangle in prepared_cells_contours)

        width = max_x - min_x
        height = max_y - min_y
        combined_rectangle = [min_x, min_y, width, height]

        return combined_rectangle

    def find_table_cells(self, temp_table_roi):
        prepared_table_roi = self.__preprocessing_for_table_cells(temp_table_roi.get_roi())

        prepared_cells_contours = self.__get_contours_for_table_cells(prepared_table_roi, temp_table_roi.get_roi())

        if(len(prepared_cells_contours) > 0):
            most_frequent_width, most_frequent_height = self.__calculate_most_frequent_cell_dimension(prepared_cells_contours)
            cells_area_x1, cells_area_y1, cells_area_x2, cells_area_y2 = self.__calculate_cells_coordinates(prepared_cells_contours, most_frequent_width, most_frequent_height)

            list_cells = self.__prepare_list_cells(cells_area_y1, most_frequent_width, most_frequent_height, prepared_cells_contours)

            absolute_cells_area_x1, absolute_cells_area_y1 = general_helpers.calculate_absolute_coordinates(temp_table_roi, cells_area_x1, cells_area_y1)
            table_cells_element = TableCellsElement("table_cells", 1.0, RoiElement(temp_table_roi.get_roi()[cells_area_y1:cells_area_y2, cells_area_x1:cells_area_x2], absolute_cells_area_x1, absolute_cells_area_y1, cells_area_x2 - cells_area_x1, cells_area_y2 - cells_area_y1), list_cells)

            return table_cells_element
        else:
            print("No cells")
            return None