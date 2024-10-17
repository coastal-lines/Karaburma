from karaburma.elements.objects.element import Element


class ScrollElement(Element):
    def __init__(self, label, prediction_value, roi, first_button, second_button):
        super().__init__(label, prediction_value, roi)
        self.__first_button = first_button
        self.___second_button = second_button

    def get_first_button(self):
        return self.__first_button

    def get_second_button(self):
        return self.___second_button