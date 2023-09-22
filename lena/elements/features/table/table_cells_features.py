import math
import cv2
import numpy as np
import skimage
from skimage.filters import threshold_otsu, threshold_mean
from skimage.util import img_as_ubyte

from lena.elements.objects.table.table_cell_element import TableCell
from lena.elements.objects.table.table_cells_element import TableCellsElement
from lena.utils import files_helper, general_helpers
from lena.utils.image_processing import filters_helper, contours_helper, morphological_helpers
from lena.elements.objects.roi_element import RoiElement

class TableCellsFeatures():

    def __preprocessing_for_table_roi(self, temp_table_roi):

        grey_ = filters_helper.convert_to_grayscale(temp_table_roi)
        imnp = np.array(grey_) / 255
        gamma = math.log(imnp.mean()) / math.log(0.3)
        new = ((imnp ** (1 / gamma)) * 255).astype(np.uint8)
        er = img_as_ubyte(morphological_helpers.erosion(new))
        #er = filters_helper.Erosion(er)
        dl = morphological_helpers.dilation(er)
        #general_helpers.show(dl)

        # Attempt to find everything that is not related to tables
        otsu_binary = skimage.img_as_ubyte(dl.copy() > threshold_otsu(dl))

        return otsu_binary

    def __preprocessing_for_table_cells(self, temp_table_roi):

        grey_ = filters_helper.convert_to_grayscale(temp_table_roi)

        kernel1 = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [-1, -1, 1, -1, -1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 4, 0, 0]])

        #grey_ = cv2.filter2D(src=grey_, ddepth=-1, kernel=kernel1)
        #general_helpers.show(grey_)

        imnp = np.array(grey_) / 255
        gamma = math.log(imnp.mean()) / math.log(0.3)
        new = ((imnp ** (1 / gamma)) * 255).astype(np.uint8)
        er = img_as_ubyte(morphological_helpers.erosion(new))
        #er = filters_helper.Erosion(er)
        dl = morphological_helpers.dilation(er)
        #general_helpers.show(er)

        # attempt to find everything that is not related to tables
        filters_helper.try_threshold(dl)
        otsu_binary = skimage.img_as_ubyte(dl.copy() > threshold_otsu(dl))

        #general_helpers.show(otsu_binary)

        return otsu_binary

    def __preprocessing_for_table_cells2(self, temp_table_roi):

        grey_ = filters_helper.convert_to_grayscale(temp_table_roi)

        kernel1 = np.array([[0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0],
                            [-1, -1, 1, 1, 1],
                            [0, 0, -1, 0, 0],
                            [0, 0, -1, 0, 0]])



        kernel2 = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 1, -1, -1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])

        grey_ = cv2.filter2D(src=grey_, ddepth=-1, kernel=kernel2)

        #grey_ = cv2.filter2D(src=grey_, ddepth=-1, kernel=kernel2)

        grey_ = filters_helper.LevelsCorrection(grey_, 80, 255, 0, 255, 0.25)

        #grey_ = filters_helper.Blur(grey_, (3 ,3))

        #otsu_binary = skimage.img_as_ubyte(grey_.copy() > threshold_otsu(grey_))

        otsu_binary = filters_helper.Blur(grey_, (3, 3))

        general_helpers.show(otsu_binary)

        return otsu_binary

    def __preprocessing_for_table_cells3(self, temp_table_roi):

        grey_ = filters_helper.convert_to_grayscale(temp_table_roi)

        #general_helpers.show(grey_)

        kernel1 = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [-1, -1, 1, -1, -1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 4, 0, 0]])

        grey_ = cv2.filter2D(src=grey_, ddepth=-1, kernel=kernel1)
        #general_helpers.show(grey_)

        imnp = np.array(grey_) / 255
        gamma = math.log(imnp.mean()) / math.log(0.3)
        new = ((imnp ** (1 / gamma)) * 255).astype(np.uint8)
        er = img_as_ubyte(morphological_helpers.erosion(new))
        #er = filters_helper.Erosion(er)
        dl = morphological_helpers.dilation(er)
        #general_helpers.show(dl)


        #general_helpers.show(dl)

        # attempt to find everything that is not related to tables
        #filters_helper.try_threshold2(dl)
        otsu_binary = skimage.img_as_ubyte(dl.copy() > threshold_otsu(dl))

        return otsu_binary

    def __preprocessing_for_table_cells4(self, temp_table_roi):

        grey_ = filters_helper.convert_to_grayscale(temp_table_roi)

        #files_helper.save_image("F:\Data\Work\OwnProjects\Python", grey_, "!test")

        #general_helpers.show(grey_)

        kernel1 = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [-1, -1, 1, -1, -1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 4, 0, 0]])

        grey_ = cv2.filter2D(src=grey_, ddepth=-1, kernel=kernel1)
        grey_ = morphological_helpers.erosion(grey_)
        grey_ = morphological_helpers.dilation(grey_)
        grey_ = filters_helper.LevelsCorrection(grey_, 86, 159, 0, 255, 0.04)
        grey_ = cv2.medianBlur(grey_, 7)

        #general_helpers.show(grey_)

        return grey_

    def __remove_noise_from_table(self, prepared_table_roi, original_table_roi):
        all, _ = contours_helper.GetContoursByCanny(prepared_table_roi, 0, 255)

        original_table_roi = filters_helper.convert_to_grayscale(original_table_roi)

        for contour in all:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            x, y, w, h = cv2.boundingRect(approx)
            #if (w > 3 and h > 3):
            if (w > 3 and h > 3 and w < 100 and h < 30):
                original_table_roi[y - 0:y + h + 0, x - 0:x + w + 0] = 255

        return original_table_roi

    def __get_contours_for_table_cells(self, prepared_table_roi, original_table_roi):
        #general_helpers.show(prepared_table_roi)
        #general_helpers.show(original_table_roi)
        cleaned_table_roi = self.__remove_noise_from_table(prepared_table_roi, original_table_roi)

        #general_helpers.show(cleaned_table_roi)

        grey_ = filters_helper.LevelsCorrection(cleaned_table_roi, 190, 236, 0, 111, 0.38)
        grey_ = filters_helper.LevelsCorrection(grey_, 121, 160, 98, 255, 0.4)

        grey_ = morphological_helpers.erosion(grey_)

        #DEBUG
        #grey_ = filters_helper.Sharp(grey_, "strong")
        kernel1 = np.array([[0, 0, 1, 0, 0],
                            [0, 0, 1, 3, 0],
                            [1, 0, 1, -1, -1],
                            [0, -2, -1, 0, 0],
                            [0, 0, 0, 0, 0]])
        #grey_ = cv2.filter2D(src=grey_, ddepth=-1, kernel=kernel1)

        #grey_ = filters_helper.Blur(grey_, (3, 3))


        grey_ = skimage.img_as_ubyte(grey_.copy() > threshold_mean(grey_))

        #grey_ = skimage.img_as_ubyte(grey_.copy() > threshold_minimum(grey_))

        #general_helpers.show(grey_)

        contours, hierarchy = cv2.findContours(grey_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        temp_contours = []

        for contour in contours:

            #cv2.drawContours(original_table_roi, [contour], -1, (0, 255, 0), 1)

            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)

            if(w > h and w > 60 and h > 7 and w < 400 and h < 200):
                temp_contours.append((x, y, w, h))

        #contours_helper.DrawRectangleByListXYWH(original_table_roi, temp_contours)
        #general_helpers.show(original_table_roi)

        return temp_contours

    def __calculate_most_frequent_cell_x_y(self, prepared_cells_contours):
        # find the most similar contours
        all_x = [arr[0] for arr in prepared_cells_contours]
        most_frequent_x = np.bincount(all_x).argmax()

        all_y = [arr[1] for arr in prepared_cells_contours]
        most_frequent_y = np.bincount(all_y).argmax()

        return most_frequent_x, most_frequent_y

    def __calculate_most_frequent_cell_dimension(self, prepared_cells_contours):
        #find the most similar contours
        all_width = [arr[2] for arr in prepared_cells_contours]
        most_frequent_width = np.bincount(all_width).argmax()

        all_height = [arr[3] for arr in prepared_cells_contours]
        most_frequent_height = np.bincount(all_height).argmax()

        return most_frequent_width, most_frequent_height

    def __calculate_most_frequent_cell_dimension2(self, prepared_cells_contours):
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

    def __prepare_list_cells(self, temp_table_roi, table_columns, table_rows, cells_area_x1, cells_area_y1, most_frequent_width, most_frequent_height, prepared_cells_contours):
        list_cells = []

        threshold = 12

        for column_index in range(table_columns):
            for row_index in range(table_rows):
                for cell_contour in prepared_cells_contours:
                    #TODO - try to simplify
                    contour_x, contour_y, contour_w, contour_h = cell_contour[0], cell_contour[1], cell_contour[2], cell_contour[3]

                    #DEBUG
                    #contour_w = (most_frequent_width + contour_w) // 2
                    #contour_h = (most_frequent_height + contour_h) // 2

                    current_x = cells_area_x1 + (most_frequent_width * column_index)
                    current_y = cells_area_y1 + (most_frequent_height * row_index)

                    # It is necessary to check for displacement, as cells may not necessarily have exactly the same width and height
                    if ((contour_x - threshold <= current_x <= contour_x + threshold) and (contour_y - threshold <= current_y <= contour_y + threshold)):
                        current_roi = temp_table_roi.get_roi()[contour_y:contour_y + most_frequent_height, contour_x:contour_x + most_frequent_width]
                        #current_text = ocr_helper.get_text(current_roi)
                        absolute_current_x, absolute_current_y = general_helpers.calculate_absolute_coordinates(temp_table_roi, contour_x, contour_y)
                        current_cell = TableCell("table_cell", "1", RoiElement(current_roi, absolute_current_x, absolute_current_y, contour_w, contour_h), "", column_index, row_index)

                        list_cells.append(current_cell)
                        break

                    else:
                        current_roi = temp_table_roi.get_roi()[current_y:current_y + most_frequent_height, current_x:current_x + most_frequent_width]
                        #current_text = ocr_helper.get_text(current_roi)

                        absolute_current_x, absolute_current_y = general_helpers.calculate_absolute_coordinates(temp_table_roi, current_x, current_y)
                        current_cell = TableCell("table_cell", "1", RoiElement(current_roi, absolute_current_x, absolute_current_y, contour_w, contour_h), "", column_index, row_index)

                        list_cells.append(current_cell)
                        break

        return list_cells

    '''
    def __custom_round(self, number):
        # Separate the integer and decimal parts
        integer_part = int(number)
        decimal_part = number - integer_part

        # Custom rounding logic
        if decimal_part < 0.5 or integer_part == 0:
            return integer_part
        else:
            return integer_part + 1
    '''

    def __get_most_frequent_features_of_nearest_cells(self, cell_contour, prepared_cells_contours, most_frequent_width, most_frequent_height):
        # take a cross-shaped matrix with the current contour in the center
        # calculate the most popular values and return them

        contour_x, contour_y, contour_w, contour_h = cell_contour[0], cell_contour[1], cell_contour[2], cell_contour[3]

        #left
        contour_1_x = contour_x - most_frequent_width
        contour_1_y = contour_y

        for temp_contour in prepared_cells_contours:
            if(temp_contour[0] <= contour_1_x <= temp_contour[0]):
                pass

        #up
        contour_1_x = contour_x
        contour_1_y = contour_y - most_frequent_height


        #right
        contour_1_x = contour_x + most_frequent_width
        contour_1_y = contour_y

        #bottom
        contour_1_x = contour_x
        contour_1_y = contour_y + most_frequent_height



        contour_1 = None
        contour_2 = None
        contour_3 = None
        contour_4 = None

    def __calculate_current_adress(self, x, y, most_frequent_width, most_frequent_height):
        row_index = general_helpers.custom_round(y / most_frequent_height)
        column_index = general_helpers.custom_round(x / most_frequent_width)

        test_y = row_index * most_frequent_height
        if(test_y != y and test_y > y):
            row_index -= 1

        print(row_index, column_index)

        return row_index, column_index

    def __prepare_list_cells2(self, temp_table_roi, table_columns, table_rows, cells_area_x1, cells_area_y1, most_frequent_width, most_frequent_height, prepared_cells_contours):
        list_cells = []

        threshold = 1

        shift_from_find_best_contours_for_table_method = 3

        for cell_contour in prepared_cells_contours:
            contour_x, contour_y, contour_w, contour_h = cell_contour[0], cell_contour[1], cell_contour[2], cell_contour[3]

            #DEBUG
            #self.__calculate_current_adress(14, 9, most_frequent_width, most_frequent_height)

            row_index = general_helpers.custom_round(contour_y / most_frequent_height)
            column_index = general_helpers.custom_round(contour_x / most_frequent_width)
            print(column_index, row_index)

            #row_index, column_index = self.__calculate_current_adress(contour_x, contour_y, most_frequent_width, most_frequent_height)

            #shift
            #contour_x -= 3
            #contour_y -= 3

            current_cell = None

            if(most_frequent_height - threshold <= contour_h <= most_frequent_height + threshold and most_frequent_width - threshold <= contour_w <= most_frequent_width + threshold):

                current_roi = temp_table_roi.get_roi()[contour_y:contour_y + contour_h, contour_x:contour_x + contour_w]
                absolute_current_x, absolute_current_y = general_helpers.calculate_absolute_coordinates(temp_table_roi, contour_x, contour_y)

                absolute_current_x = absolute_current_x - shift_from_find_best_contours_for_table_method
                absolute_current_y = absolute_current_y - shift_from_find_best_contours_for_table_method

                current_cell = TableCell("table_cell", "1", RoiElement(current_roi, absolute_current_x, absolute_current_y, contour_w, contour_h), "", column_index, row_index)
            else:
                #if(contour_h != most_frequent_height):
                #    contour_h = most_frequent_height

                #if(contour_w != most_frequent_width):
                #    contour_w = most_frequent_width

                if(contour_w > 60 and contour_h > 8):
                    current_roi = temp_table_roi.get_roi()[contour_y:contour_y + contour_h, contour_x:contour_x + contour_w]
                    absolute_current_x, absolute_current_y = general_helpers.calculate_absolute_coordinates(temp_table_roi, contour_x, contour_y)

                    absolute_current_x = absolute_current_x - shift_from_find_best_contours_for_table_method
                    absolute_current_y = absolute_current_y - shift_from_find_best_contours_for_table_method

                    current_cell = TableCell("table_cell", "1", RoiElement(current_roi, absolute_current_x, absolute_current_y, contour_w, contour_h), "", column_index, row_index)

            if(current_cell != None):
                list_cells.append(current_cell)

        return list_cells

    def __prepare_list_cells3(self, temp_table_roi, table_columns, table_rows, cells_area_x1, cells_area_y1, most_frequent_width, most_frequent_height, prepared_cells_contours):
        list_cells = []

        files_helper.save_image("Projects\!", temp_table_roi.get_roi(), "t")

        threshold = 1

        shift_from_find_best_contours_for_table_method = 3
        shift_for_cells = 2

        for i in range(0, table_columns):
            x_0 = (i * most_frequent_width) + cells_area_x1
            x_1 = ((i * most_frequent_width) + most_frequent_width) + cells_area_x1

            for j in range(0, table_rows):
                y_0 =(j * most_frequent_height) + cells_area_y1
                y_1 = ((j * most_frequent_height) + most_frequent_height) + cells_area_y1

                if(i == 0 and j == 0):
                    print("")

                for cell_contour in prepared_cells_contours:
                    contour_x, contour_y, contour_w, contour_h = cell_contour[0], cell_contour[1], cell_contour[2], cell_contour[3]

                    if (contour_x == 17 and contour_y == 18):
                        print("")

                    centre_x = contour_x + (contour_w // 2)
                    centre_y = contour_y + (contour_h // 2)

                    if (x_0 - shift_for_cells <= contour_x <= x_1 + shift_for_cells
                            and y_0 - shift_for_cells <= contour_y <= y_1 + shift_for_cells):

                        cv2.putText(temp_table_roi.get_roi(), str(i) + " " + str(j), (centre_x, centre_y), cv2.FONT_HERSHEY_PLAIN, 1,
                                    (0, 0, 0), 1, cv2.LINE_AA)

                        # DEBUG
                        # self.__calculate_current_adress(14, 9, most_frequent_width, most_frequent_height)

                        column_index = i
                        row_index = j
                        print(column_index, row_index)

                        # row_index, column_index = self.__calculate_current_adress(contour_x, contour_y, most_frequent_width, most_frequent_height)

                        # shift
                        # contour_x -= 3
                        # contour_y -= 3

                        current_cell = None

                        if (most_frequent_height - threshold <= contour_h <= most_frequent_height + threshold and most_frequent_width - threshold <= contour_w <= most_frequent_width + threshold):
                            current_roi = temp_table_roi.get_roi()[contour_y:contour_y + contour_h, contour_x:contour_x + contour_w]
                            absolute_current_x, absolute_current_y = general_helpers.calculate_absolute_coordinates(temp_table_roi, contour_x, contour_y)

                            absolute_current_x = absolute_current_x - shift_from_find_best_contours_for_table_method
                            absolute_current_y = absolute_current_y - shift_from_find_best_contours_for_table_method

                            current_cell = TableCell("table_cell", "1", RoiElement(current_roi, absolute_current_x, absolute_current_y, contour_w, contour_h), "", column_index, row_index)
                        else:
                            # if(contour_h != most_frequent_height):
                            #    contour_h = most_frequent_height

                            # if(contour_w != most_frequent_width):
                            #    contour_w = most_frequent_width

                            if (contour_w > 60 and contour_h > 8):
                                current_roi = temp_table_roi.get_roi()[contour_y:contour_y + contour_h, contour_x:contour_x + contour_w]
                                absolute_current_x, absolute_current_y = general_helpers.calculate_absolute_coordinates(temp_table_roi, contour_x, contour_y)

                                absolute_current_x = absolute_current_x - shift_from_find_best_contours_for_table_method
                                absolute_current_y = absolute_current_y - shift_from_find_best_contours_for_table_method

                                current_cell = TableCell("table_cell", "1", RoiElement(current_roi, absolute_current_x, absolute_current_y, contour_w, contour_h), "", column_index, row_index)

                        if (current_cell != None):
                            list_cells.append(current_cell)

        #general_helpers.show(temp_table_roi.get_roi())

        return list_cells

    def __prepare_list_cells4(self, temp_table_roi, table_columns, table_rows, cells_area_x1, cells_area_y1, most_frequent_width, most_frequent_height, prepared_cells_contours):
        list_cells = []

        files_helper.save_image("Projects\!", temp_table_roi.get_roi(), "t")

        threshold = 1

        shift_from_find_best_contours_for_table_method = 3
        shift_for_cells = 2

        temp_rows_numbers = []

        for i in range(0, table_columns):
            x_0 = (i * most_frequent_width) + cells_area_x1
            x_1 = ((i * most_frequent_width) + most_frequent_width) + cells_area_x1

            for j in range(0, table_rows):
                y_0 =(j * most_frequent_height) + cells_area_y1
                y_1 = ((j * most_frequent_height) + most_frequent_height) + cells_area_y1

                if(i == 0 and j == 0):
                    print("")

                for cell_contour in prepared_cells_contours:
                    contour_x, contour_y, contour_w, contour_h = cell_contour[0], cell_contour[1], cell_contour[2], cell_contour[3]

                    if (contour_x == 17 and contour_y == 18):
                        print("")

                    centre_x = contour_x + (contour_w // 2)
                    centre_y = contour_y + (contour_h // 2)

                    if (x_0 - shift_for_cells <= contour_x <= x_1 + shift_for_cells
                            and y_0 - shift_for_cells <= contour_y <= y_1 + shift_for_cells):

                        temp_rows_numbers.append(j)

                        print(contour_y)

                        #if():

                        cv2.putText(temp_table_roi.get_roi(), str(i) + " " + str(j), (centre_x, centre_y), cv2.FONT_HERSHEY_PLAIN, 1,
                                    (0, 0, 0), 1, cv2.LINE_AA)

                        # DEBUG
                        # self.__calculate_current_adress(14, 9, most_frequent_width, most_frequent_height)

                        column_index = i
                        row_index = j
                        #print(column_index, row_index)

                        # row_index, column_index = self.__calculate_current_adress(contour_x, contour_y, most_frequent_width, most_frequent_height)

                        # shift
                        # contour_x -= 3
                        # contour_y -= 3

                        current_cell = None

                        if (most_frequent_height - threshold <= contour_h <= most_frequent_height + threshold and most_frequent_width - threshold <= contour_w <= most_frequent_width + threshold):
                            current_roi = temp_table_roi.get_roi()[contour_y:contour_y + contour_h, contour_x:contour_x + contour_w]
                            absolute_current_x, absolute_current_y = general_helpers.calculate_absolute_coordinates(temp_table_roi, contour_x, contour_y)

                            absolute_current_x = absolute_current_x - shift_from_find_best_contours_for_table_method
                            absolute_current_y = absolute_current_y - shift_from_find_best_contours_for_table_method

                            current_cell = TableCell("table_cell", 1.0, RoiElement(current_roi, absolute_current_x, absolute_current_y, contour_w, contour_h), "", column_index, row_index)
                        else:
                            # if(contour_h != most_frequent_height):
                            #    contour_h = most_frequent_height

                            # if(contour_w != most_frequent_width):
                            #    contour_w = most_frequent_width

                            if (contour_w > 60 and contour_h > 8):
                                current_roi = temp_table_roi.get_roi()[contour_y:contour_y + contour_h, contour_x:contour_x + contour_w]
                                absolute_current_x, absolute_current_y = general_helpers.calculate_absolute_coordinates(temp_table_roi, contour_x, contour_y)

                                absolute_current_x = absolute_current_x - shift_from_find_best_contours_for_table_method
                                absolute_current_y = absolute_current_y - shift_from_find_best_contours_for_table_method

                                current_cell = TableCell("table_cell", "1", RoiElement(current_roi, absolute_current_x, absolute_current_y, contour_w, contour_h), "", column_index, row_index)

                        if (current_cell != None):
                            list_cells.append(current_cell)

        #general_helpers.show(temp_table_roi.get_roi())

        return list_cells

    def __prepare_list_cells5(self, temp_table_roi, table_columns, table_rows, cells_area_x1, cells_area_y1, most_frequent_width, most_frequent_height, prepared_cells_contours):
        list_cells = []

        #DEBUG
        #files_helper.save_image("Project\!", temp_table_roi.get_roi(), "t")

        threshold = 1

        shift_from_find_best_contours_for_table_method = 3
        shift_for_cells = 2

        column_index = 0
        row_index = -1
        temp_cell_y = -1

        prepared_cells_contours.reverse()
        for cell_contour in prepared_cells_contours:
            contour_x, contour_y, contour_w, contour_h = cell_contour[0], cell_contour[1], cell_contour[2], cell_contour[3]

            centre_x = contour_x + (contour_w // 2)
            centre_y = contour_y + (contour_h // 2)

            if(temp_cell_y == contour_y):
                column_index += 1
            else:
                column_index = 0
                row_index += 1

            #cv2.putText(temp_table_roi.get_roi(), str(column_index) + " " + str(row_index), (centre_x, centre_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)

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

        #general_helpers.show(temp_table_roi.get_roi())

        return list_cells

    def get_roi_rectangle_for_table_cells(self, temp_table_roi):
        prepared_table_roi = self.__preprocessing_for_table_cells4(temp_table_roi.get_roi())
        prepared_cells_contours = self.__get_contours_for_table_cells(prepared_table_roi, temp_table_roi.get_roi())

        min_x = min(rectangle[0] for rectangle in prepared_cells_contours)
        min_y = min(rectangle[1] for rectangle in prepared_cells_contours)
        max_x = max(rectangle[0] + rectangle[2] for rectangle in prepared_cells_contours)
        max_y = max(rectangle[1] + rectangle[3] for rectangle in prepared_cells_contours)

        width = max_x - min_x
        height = max_y - min_y

        combined_rectangle = [min_x, min_y, width, height]

        print("")

        return combined_rectangle

    def find_table_cells(self, temp_table_roi):
        prepared_table_roi = self.__preprocessing_for_table_cells4(temp_table_roi.get_roi())

        prepared_cells_contours = self.__get_contours_for_table_cells(prepared_table_roi, temp_table_roi.get_roi())

        if(len(prepared_cells_contours) > 0):
            most_frequent_width, most_frequent_height = self.__calculate_most_frequent_cell_dimension2(prepared_cells_contours)
            #most_frequent_x, most_frequent_y = self.__calculate_most_frequent_cell_x_y(prepared_cells_contours)
            cells_area_x1, cells_area_y1, cells_area_x2, cells_area_y2 = self.__calculate_cells_coordinates(prepared_cells_contours, most_frequent_width, most_frequent_height)
            table_columns, table_rows = self.__calculate_columns_and_rows_number(cells_area_x2, cells_area_y2, most_frequent_width, most_frequent_height)
            list_cells = self.__prepare_list_cells5(temp_table_roi, table_columns, table_rows, cells_area_x1, cells_area_y1, most_frequent_width, most_frequent_height, prepared_cells_contours)

            absolute_cells_area_x1, absolute_cells_area_y1 = general_helpers.calculate_absolute_coordinates(temp_table_roi, cells_area_x1, cells_area_y1)
            table_cells_element = TableCellsElement("table_cells", 1.0, RoiElement(temp_table_roi.get_roi()[cells_area_y1:cells_area_y2, cells_area_x1:cells_area_x2], absolute_cells_area_x1, absolute_cells_area_y1, cells_area_x2 - cells_area_x1, cells_area_y2 - cells_area_y1), list_cells)

            return table_cells_element
        else:
            print("No cells")
            return None