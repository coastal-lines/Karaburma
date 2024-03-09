from karaburma.elements.objects.element import Element
from karaburma.utils.ocr import ocr_helper


class ListBoxElement(Element):
    def __init__(self, label, prediction_value, roi):
        super().__init__(label, prediction_value, roi)
        self.__h_scroll = None
        self.__v_scroll = None
        self.__list_text = ""

        self.__roi_without_scrolls = None

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

    @property
    def textarea(self):
        return self.__textarea

    @textarea.setter
    def textarea(self, textarea: Element):
        self.__textarea = textarea

    def add_list_text(self, list_text: str):
        self.__list_text = list_text

    def get_list_text(self) -> str:
        return self.__list_text

    def prepare_roi_and_set_text(self):
        self.add_list_text(ocr_helper.update_text_for_element(self.textarea.get_roi_element().get_roi()))
        print(f"{super().get_label()} text: ", self.get_list_text())

    @property
    def full_text_area(self) -> Element:
        return self.__full_text_area

    @full_text_area.setter
    def full_text_area(self, full_text_area: Element):
        self.__full_text_area = full_text_area
