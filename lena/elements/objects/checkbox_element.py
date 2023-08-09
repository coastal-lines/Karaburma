from lena.elements.objects.element import Element

class CheckboxElement(Element):
    def __init__(self, label, prediction_value, roi):
        super().__init__(label, prediction_value, roi)
        self.__checkbox_square = None
        self.__checkbox_text = None
        self.__checbox_condition = None