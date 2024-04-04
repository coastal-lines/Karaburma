from karaburma.elements.objects.element import Element
from karaburma.utils.ocr import ocr_helper


class SliderElement(Element):
    def __init__(self, label, prediction_value, roi):
        super().__init__(label, prediction_value, roi)
        self.__slider_position = None
        self.__combobox_condition = None
        self.__slider_text = None

    def add_text(self, slider_text):
        self.__slider_text = slider_text

    def prepare_roi_and_set_text(self):
        self.__slider_text = ocr_helper.update_text_for_element(super().get_roi_element().get_roi())
        print(f"{super().get_label()} text: ", self.__slider_text)

    def get_text(self):
        return self.__slider_text