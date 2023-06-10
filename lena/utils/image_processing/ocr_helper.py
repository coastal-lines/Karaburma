import cv2
import pytesseract

from lena.utils import general_helpers
from lena.utils.image_processing import filters_helper


#setup tesseract
#check system folder user/app
#check "output_type=pytesseract.Output.DICT"
#

def get_text(roi, config="--psm 10 --oem 3"):
    grey_roi = filters_helper.convert_to_grayscale(roi)

    pytesseract.pytesseract.tesseract_cmd = 'tesseract.exe'
    text = pytesseract.image_to_string(grey_roi, lang='eng', config=config)

    return text

def get_text_and_text_data(grey_roi):
    pytesseract.pytesseract.tesseract_cmd = 'tesseract.exe'
    text = pytesseract.image_to_string(grey_roi, lang='eng', config="--psm 10 --oem 3")
    text_data = pytesseract.image_to_data(grey_roi, output_type=pytesseract.Output.DICT, lang='eng', config="--psm 10 --oem 3")

    return text, text_data

def calculate_scrolling_shift_by_text_position(img):
    list_subject1 = img
    list_subject1 = filters_helper.convert_to_grayscale(list_subject1)
    #general_helpers.show(list_subject1)

    th, threshed = cv2.threshold(list_subject1, 127, 255, cv2.THRESH_BINARY_INV)
    #general_helpers.show(threshed)
    non_zero_points = cv2.findNonZero(threshed)

    # cv2.REDUCE_AVG - the output is the mean vector of all rows/columns of the matrix
    # "1" - dimension index along which the matrix is reduced. 0 means that the matrix is reduced to a single row. 1 means that the matrix is reduced to a single column
    hist = cv2.reduce(threshed, 1, cv2.REDUCE_AVG)
    # reshape(-1) - The criterion to satisfy for providing the new shape is that 'The new shape should be compatible with the original shape
    # array = [[1], [2], [2]] -> arrary.shape = (3,1) -> array.reshape(-1) = [1,2,3]
    hist_reshaped = hist.reshape(-1)

    th = 2
    H, W = list_subject1.shape[:2]
    # uppers2 = [y for y in range(H-1) if hist_reshaped[y]<=th and hist_reshaped[y+1]>th]

    uppers = []
    for i in range(H - 1):
        if (hist_reshaped[i] <= th and hist_reshaped[i + 1] > th):
            uppers.append(i)

    #DEMO
    finish_image = cv2.cvtColor(list_subject1, cv2.COLOR_GRAY2BGR)
    #for y in uppers:
    #    cv2.line(finish_image, (0, y), (W, y), (255, 0, 0), 1)
    #general_helpers.show(finish_image)

    #text_height = uppers[0] + uppers[1]
    text_height = uppers[1]

    return 0, text_height

