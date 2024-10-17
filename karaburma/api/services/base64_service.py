from karaburma.utils import files_helper


class Base64Service:
    def __init__(self, karaburma_instance):
        self.__karaburma_instance = karaburma_instance

    def find_element_in_base64image(self, type_element, is_read_text, base64_image):
        result_json = dict()

        user_image = files_helper.base64_to_image(base64_image)

        match type_element:
            case "all":
                if is_read_text:
                    result_json = self.__karaburma_instance.find_all_elements_in_base64image(user_image, True)
                else:
                    result_json = self.__karaburma_instance.find_all_elements_in_base64image(user_image)
            case _:
                result_json = self.__karaburma_instance.find_element_in_base64image(type_element, user_image)

        return result_json

    def find_all_elements_and_pattern_in_base64image(self, image_pattern_type_element, image_pattern_base64_image, base64_image):
        result_json = dict()

        user_image = files_helper.base64_to_image(base64_image)
        image_pattern = files_helper.base64_to_image(image_pattern_base64_image)

        result_json = self.__karaburma_instance.find_all_elements_include_patterns_in_base64image(
            image_pattern,
            "normal",
            0.8,
            image_pattern_type_element,
            user_image
        )

        return result_json