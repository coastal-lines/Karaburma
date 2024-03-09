from karaburma.elements.objects.element import Element
from karaburma.utils.ocr import ocr_helper


class ComboboxElement(Element):
    def __init__(self, label, prediction_value, roi):
        super().__init__(label, prediction_value, roi)
        self.__combobox_button = None
        self.__combobox_text = None
        self.__combobox_condition = None

    def add_text(self, combobox_text):
        self.__combobox_text = combobox_text

    def prepare_roi_and_set_text(self):
        self.__text = ocr_helper.update_text_for_element(super().get_roi_element().get_roi())
        print(f"{super().get_label()} text: ", self.__text)

    def get_text(self):
        return self.__combobox_text