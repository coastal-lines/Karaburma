from karaburma.elements.objects.element import Element
from karaburma.utils.ocr import ocr_helper


class TableCell(Element):
    def __init__(self, label, prediction_value, roi, text, column_index, row_index):
        super().__init__(label, prediction_value, roi)
        self.__address = [column_index, row_index]
        self.__text = text

    def read_text_from_cell(self):
        current_text = ocr_helper.get_text(self.get_roi_element().get_roi())
        self.__text = current_text
        return current_text

    def get_adress(self):
        return self.__address

    def get_text(self):
        return self.__text