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