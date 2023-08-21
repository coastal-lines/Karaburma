from lena.elements.objects.element import Element
from lena.utils.image_processing import ocr_helper

class TableCell(Element):
    def __init__(self, label, prediction_value, roi, text, column_index, row_index):
        super().__init__(label, prediction_value, roi)
        self.__adress = [column_index, row_index]
        self.__text = text

    def read_text_from_cell(self):
        current_text = ocr_helper.get_text(self.get_roi_element().get_roi())
        return current_text

    def get_adress(self):
        return self.__adress

    def get_text(self):
        return self.__text