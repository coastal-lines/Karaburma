class FileService:
    def __init__(self, karaburma_instance):
        self.__karaburma_instance = karaburma_instance

    def find_element_in_file(self, type_element, is_read_text, image_file_path):
        result_json = dict()

        match type_element:
            case "all":
                if is_read_text:
                    result_json = self.__karaburma_instance.find_all_elements_and_read_text(image_file_path)
                else:
                    result_json = self.__karaburma_instance.find_all_elements(image_file_path)
            case _:
                result_json = self.__karaburma_instance.find_element(type_element, image_file_path)

        return result_json

    def find_element_by_pattern(self, image_pattern_type_element, image_file_path, image_pattern_file_path, is_all_elements, search_mode):
        result_json = dict()

        if (is_all_elements):
            result_json = self.__karaburma_instance.find_all_elements_include_patterns(
                [image_pattern_file_path],
                search_mode,
                0.8,
                image_pattern_type_element,
                image_file_path)
        else:
            result_json = self.__karaburma_instance.find_element_by_patterns(
                [image_pattern_file_path],
                "normal",
                0.8,
                image_pattern_type_element,
                image_file_path)

        return result_json