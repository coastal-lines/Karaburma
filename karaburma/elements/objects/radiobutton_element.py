from karaburma.elements.objects.element import Element
from karaburma.utils.ocr import ocr_helper


class RadioButtonElement(Element):
    def __init__(self, label, prediction_value, roi):
        super().__init__(label, prediction_value, roi)
        self.__radiobutton_circle = None
        self.__radiobutton_text = None
        self.__radiobutton_condition = None

    def add_text(self, radiobutton_text):
        self.__radiobutton_text = radiobutton_text

    def prepare_roi_and_set_text(self):
        self.__radiobutton_text = ocr_helper.update_text_for_element(super().get_roi_element().get_roi())
        print(f"{super().get_label()} text: ", self.__radiobutton_text)

    def get_text(self):
        return self.__radiobutton_text