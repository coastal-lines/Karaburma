from lena.navigation.template_matching.template_matching import TemplateMatchingElement
from lena.elements.objects.screenshot_element import ImageSourceObject
from lena.navigation.object_detection.base_find_actions import BaseFindActions
from lena.utils import general_helpers, files_helper, debug


class FindManager(TemplateMatchingElement):

    def __init__(self, detection_mode: BaseFindActions):
        super().__init__()
        self.__detection_mode = detection_mode
