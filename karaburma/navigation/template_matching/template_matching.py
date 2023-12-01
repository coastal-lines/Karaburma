from karaburma.elements.objects.element import Element
from karaburma.elements.objects.roi_element import RoiElement
from karaburma.utils import files_helper
from karaburma.utils.config_manager import ConfigManager
from karaburma.utils.image_processing import pattern_matching


class TemplateMatchingElement:
    def find_element_by_patterns(self, patterns, mode, threshold, user_label, image_source):
        pattern_matching_result = None

        loaded_patterns = files_helper.load_grayscale_images(patterns)

        if (threshold == None):
            threshold = ConfigManager().config.elements_parameters.listbox.scrollbar_shift_threshold

        if(mode == "normal"):
            pattern_matching_result = pattern_matching.multi_match_for_list_patterns(image_source.get_current_image_source(), loaded_patterns, threshold)
        elif(mode == "extended"):
            pattern_matching_result = pattern_matching.multi_match_for_list_patterns_with_augmentation(image_source.get_current_image_source(), loaded_patterns, threshold)

        for rectangle in pattern_matching_result:
            x, y, w, h = rectangle[0], rectangle[1], rectangle[2], rectangle[3]
            roi = image_source.get_current_image_source()[y:y + h, x:x + w, :]
            image_source.add_element(Element(user_label, threshold, RoiElement(roi, x, y, w, h)))