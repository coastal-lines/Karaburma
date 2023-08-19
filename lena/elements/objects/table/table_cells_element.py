from lena.elements.objects.element import Element

class TableCellsElement(Element):
    def __init__(self, label, prediction_value, roi, list_cells):
        super().__init__(label, prediction_value, roi)
        self.__list_cells = list_cells

    def get_list_cells(self):
        return self.__list_cells