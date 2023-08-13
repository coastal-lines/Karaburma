from typing import Optional

from lena.elements.objects.element import Element

class ListBoxElement(Element):
    def __init__(self, label, prediction_value, roi):
        super().__init__(label, prediction_value, roi)
        self.__h_scroll = None
        self.__v_scroll = None
        self.__list_text = ""

        self.__textarea = None
        self.__full_text_area = None

    def add_scroll(self, label, scroll):
        if(label == "h_scroll"):
            self.__h_scroll = scroll
        elif(label == "v_scroll"):
            self.__v_scroll = scroll

    def get_v_scroll(self):
        if (self.__v_scroll != None):
            return self.__v_scroll
        else:
            return None