from karaburma.elements.objects.element import Element
from karaburma.utils.image_processing import augmentation, filters_helper
from karaburma.utils.ocr import ocr_helper


class ButtonElement(Element):
    def __init__(self, label, prediction_value, roi):
        super().__init__(label, prediction_value, roi)
        self.__text = None

    def prepare_roi_and_set_text(self):
        self.__text = ocr_helper.update_text_for_element(super().get_roi_element().get_roi())
        print(f"{super().get_label()} text: ", self.__text)

    def get_text(self):
        return self.__text