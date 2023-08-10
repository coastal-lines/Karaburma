from lena.elements.objects.element import Element

class ComboboxElement(Element):
    def __init__(self, label, prediction_value, roi):
        super().__init__(label, prediction_value, roi)
        self.__combobox_button = None
        self.__combobox_text = None
        self.__combobox_condition = None

    def add_text(self, combobox_text):
        self.__combobox_text = combobox_text

    def get_text(self):
        return self.__combobox_text