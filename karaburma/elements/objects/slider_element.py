from karaburma.elements.objects.element import Element

class SliderElement(Element):
    def __init__(self, label, prediction_value, roi):
        super().__init__(label, prediction_value, roi)
        self.__slider_position = None
        self.__combobox_condition = None