import os
import cv2
import numpy as np
import pytesseract

from karaburma.utils.image_processing import filters_helper
from karaburma.utils.config_manager import ConfigManager
from karaburma.utils.image_processing import augmentation


#setup tesseract
#check system folder user/app
#check "output_type=pytesseract.Output.DICT"
#

def check_platform_before_processing():
    if (ConfigManager().config["platform"] == "windows"):
        pytesseract.pytesseract.tesseract_cmd = os.path.join(
            os.path.expanduser('~\\AppData'), "Local\\Programs\\Tesseract-OCR\\tesseract.exe"
        )

def get_text(roi, config="--psm 10 --oem 3"):
    check_platform_before_processing()
    grey_roi = filters_helper.convert_to_grayscale(roi)
    #pytesseract.pytesseract.tesseract_cmd = os.path.join(os.path.expanduser('~\\AppData'), "Local\\Programs\\Tesseract-OCR\\tesseract.exe")
    text = pytesseract.image_to_string(grey_roi, lang='eng', config=config)

    return text

def get_text_and_text_data(grey_roi):
    #pytesseract.pytesseract.tesseract_cmd = 'tesseract.exe'
    check_platform_before_processing()
    text = pytesseract.image_to_string(grey_roi, lang='eng', config="--psm 10 --oem 3")
    text_data = pytesseract.image_to_data(grey_roi, output_type=pytesseract.Output.DICT, lang='eng', config="--psm 10 --oem 3")

    return text, text_data

def calculate_scrolling_shift_by_text_position(roi):
    roi = filters_helper.convert_to_grayscale(roi)
    _, roi_thresholded = filters_helper.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)

    # cv2.REDUCE_AVG - the output is the mean vector of all rows/columns of the matrix
    # "1" - dimension index along which the matrix is reduced. 0 means that the matrix is reduced to a single row. 1 means that the matrix is reduced to a single column
    histogram = cv2.reduce(roi_thresholded, 1, cv2.REDUCE_AVG)

    # reshape(-1) - The criterion to satisfy for providing the new shape is that 'The new shape should be compatible with the original shape
    # array = [[1], [2], [2]] -> arrary.shape = (3,1) -> array.reshape(-1) = [1,2,3]
    histogram_reshaped = histogram.reshape(-1)

    histogram_threshold = 2
    h, _ = roi.shape[:2]

    uppers = []
    for i in range(h - 1):
        if (histogram_reshaped[i] <= histogram_threshold and histogram_reshaped[i + 1] > histogram_threshold):
            uppers.append(i)

    text_height = uppers[1]

    return 0, text_height

def update_text_for_element(roi: np.ndarray) -> str:
    img = filters_helper.convert_to_grayscale(roi)
    _, img = filters_helper.threshold(img, 127, 255)
    img = augmentation.bicubic_resize(img, (img.shape[1] * 1, img.shape[0] * 1))

    return get_text(img, "--psm 3 --oem 3").replace(" ", "").replace("\n", "")

def read_text_for_all_imagesource_elements(image_source):
    for element in image_source.get_elements():
        match element.get_label():
            case "listbox":
                pass
            case "table":
                pass
            case _:
                if (hasattr(element, 'get_text')):
                    element.
                    
                pass



