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