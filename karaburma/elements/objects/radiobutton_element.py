from karaburma.elements.objects.element import Element

class RadioButtonElement(Element):
    def __init__(self, label, prediction_value, roi):
        super().__init__(label, prediction_value, roi)
        self.__radiobutton_circle = None
        self.__radiobutton_text = None
        self.__radiobutton_condition = None

    def add_text(self, radiobutton_text):
        self.__radiobutton_text = radiobutton_text

    def get_text(self):
        return self.__radiobutton_text