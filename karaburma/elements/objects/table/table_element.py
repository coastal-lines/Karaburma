from typing import Optional

from karaburma.data.constants.enums.element_types_enum import ElementTypesEnum
from karaburma.elements.objects.roi_element import RoiElement
from karaburma.elements.objects.element import Element
from karaburma.elements.objects.scroll_element import ScrollElement
from karaburma.elements.objects.table.table_cells_element import TableCellsElement


class TableElement(Element):
    def __init__(self, label: str, prediction_value: float, roi: RoiElement, h_scroll, v_scroll, cells_area_element: TableCellsElement):
        super().__init__(label, prediction_value, roi)
        self.__h_scroll = h_scroll
        self.__v_scroll = v_scroll
        self.__cells_area_element = cells_area_element
        self.__full_table_area: Optional[TableElement] = None

    def add_scroll(self, label: str, prediction_value: float, roi: RoiElement, first_button: Element,
                   second_button: Element):
        if(label == ElementTypesEnum.h_scroll.name):
            self.__h_scroll = ScrollElement(label, prediction_value, roi, first_button, second_button)

        elif(label == ElementTypesEnum.v_scroll.name):
            self.__v_scroll = ScrollElement(label, prediction_value, roi, first_button, second_button)

    def set_full_table_area(self, roi: RoiElement, cells_area_element):
        self.__full_table_area = TableElement("table", 1.0, roi, None, None, cells_area_element)

    def get_full_table_area(self):
        return self.__full_table_area

    def get_v_scroll(self):
        return self.__v_scroll

    def get_h_scroll(self):
        return self.__h_scroll

    def get_cells_area_element(self):
        if(self.__cells_area_element != None):
            return self.__cells_area_element
        else:
            #TODO
            print("EXCEPTION")

    def get_cell_by_adress(self, column_index: int, row_index: int):
        custom_cell = [cell for cell in self.get_cells_area_element().get_list_cells()
                       if cell.get_adress() == [column_index, row_index]][0]

        return custom_cell