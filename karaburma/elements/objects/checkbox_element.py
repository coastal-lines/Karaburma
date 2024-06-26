from karaburma.elements.objects.element import Element
from karaburma.utils.ocr import ocr_helper


class CheckboxElement(Element):
    def __init__(self, label, prediction_value, roi):
        super().__init__(label, prediction_value, roi)
        self.__checkbox_square = None
        self.__checkbox_text = None
        self.__checbox_condition = None

    def add_text(self, checkbox_text):
        self.__checkbox_text = checkbox_text

    def prepare_roi_and_set_text(self):
        self.__checkbox_text = ocr_helper.update_text_for_element(super().get_roi_element().get_roi())
        print(f"{super().get_label()} text: ", self.__checkbox_text)

    def get_text(self):
        return self.__checkbox_text