from lena.utils import general_helpers
from lena.utils.image_processing import contours_helper

TABLE_COLOUR = (0, 246, 255)
TABLE_THICKNES = 4

TABLE_SCROLLS_COLOUR = (0, 136, 255)
TABLE_SCROLLS_THICKNES = 2

TABLE_CELLS_COLOUR = (2, 27, 255)
TABLE_CELLS_THICKNES = 1

BUTTONS_COLOUR = (255, 0, 114)
BUTTONS_THICKNES = 4

INPUT_COLOUR = (46, 255, 0)
INPUT_THICKNES = 4

RADIOBUTTON_COLOUR = (250, 255, 0)
RADIOBUTTON_THICKNES = 2

COMBOBOX_COLOUR = (135, 0, 255)
COMBOBOX_THICKNES = 4

CHECKBOX_COLOUR = (0, 255, 212)
CHECKBOX_THICKNES = 2

LISTBOX_COLOUR = (0, 0, 0)
LISTBOX_THICKNES = 2

SLIDER_COLOUR = (0, 46, 198)
SLIDER_THICKNES = 2

def draw_custom_elements(screenshot_copy_debug, elements, element_type, rectangle_colour, rectangle_thickness):
    for element in elements:
        if (element.get_label() == element_type):
            contours_helper.draw_rectangle_and_label_for_element(screenshot_copy_debug,
                                                                 element,
                                                                 rectangle_colour,
                                                                 rectangle_thickness)
            
def draw_pattern_matching_elements(screenshot_copy_debug, elements, rectangle_colour, rectangle_thickness):
    for element in elements:
        if (element.get_label() != "non" and
            element.get_label() != "table" and
            element.get_label() != "listbox" and
            element.get_label() != "button" and
            element.get_label() != "input" and
            element.get_label() != "radiobutton" and
            element.get_label() != "combobox" and
            element.get_label() != "checkbox" and
            element.get_label() != "slider"):
            contours_helper.draw_rectangle_and_label_for_element(screenshot_copy_debug,
                                                                 element,
                                                                 rectangle_colour,
                                                                 rectangle_thickness)