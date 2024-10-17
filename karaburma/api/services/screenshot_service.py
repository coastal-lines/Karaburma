from karaburma.data.constants.enums.element_types_enum import ElementTypesEnum


class ScreenshotService:
    def __init__(self, karaburma_instance):
        self.__karaburma_instance = karaburma_instance

    def find_elements_in_screenshot(self, type_element, is_fully_expanded, is_read_text):
        result_json = ""

        match type_element:
            case ElementTypesEnum.table.name:
                if is_fully_expanded:
                    result_json = self.__karaburma_instance.find_table_and_expand(0)
            case ElementTypesEnum.listbox.name:
                if is_fully_expanded:
                    result_json = self.__karaburma_instance.find_listbox_and_expand(0)
                else:
                    result_json = self.__karaburma_instance.find_element(type_element)
            case "all":
                if is_read_text:
                    result_json = self.__karaburma_instance.find_all_elements_and_read_text()
                else:
                    result_json = self.__karaburma_instance.find_all_elements()
            case _:
                result_json = self.__karaburma_instance.find_element(type_element)

        return result_json

    def find_text_with_expanded_table_in_screenshot(self, table_number=0):
        return self.__karaburma_instance.find_table_and_expand(table_number, True)
