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

    def add_list_text(self, list_text):
        self.__list_text = list_text

    def get_list_text(self):
        return self.__list_text

    @property
    def textarea(self):
        return self.__textarea

    @textarea.setter
    def textarea(self, textarea: Element):
        self.__textarea = textarea

    @property
    def full_text_area(self):
        return self.__full_text_area

    @full_text_area.setter
    def full_text_area(self, full_text_area: Element):
        self.__full_text_area = full_text_area
