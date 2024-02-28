import numpy as np

from karaburma.utils import general_helpers
from karaburma.utils.image_processing import contours_helper

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

def draw_listbox(screenshot_copy_debug, screenshot_elements):
    for listbox in (item for item in screenshot_elements.get_elements() if item.get_label() == "listbox"):
        draw_custom_elements(screenshot_copy_debug, [listbox], "listbox", LISTBOX_COLOUR, LISTBOX_THICKNES)

        if(listbox.get_v_scroll() != None):
            draw_custom_elements(screenshot_copy_debug, [listbox.get_v_scroll()], "v_scroll", LISTBOX_COLOUR, TABLE_SCROLLS_THICKNES)
            draw_custom_elements(screenshot_copy_debug, [listbox.get_v_scroll().get_first_button()], "UP", LISTBOX_COLOUR, TABLE_SCROLLS_THICKNES)
            draw_custom_elements(screenshot_copy_debug, [listbox.get_v_scroll().get_second_button()], "DOWN", LISTBOX_COLOUR, TABLE_SCROLLS_THICKNES)

def draw_tables(screenshot_copy_debug, screenshot_elements):
    for table in screenshot_elements.get_table_elements():
        draw_custom_elements(screenshot_copy_debug, screenshot_elements.get_elements(), "table", TABLE_COLOUR, TABLE_THICKNES)

        if(table.get_v_scroll() != None):
            draw_custom_elements(screenshot_copy_debug, [table.get_v_scroll()], "v_scroll", TABLE_SCROLLS_COLOUR, TABLE_SCROLLS_THICKNES)
            draw_custom_elements(screenshot_copy_debug, [table.get_v_scroll().get_first_button()], "UP", TABLE_SCROLLS_COLOUR, TABLE_SCROLLS_THICKNES)
            draw_custom_elements(screenshot_copy_debug, [table.get_v_scroll().get_second_button()], "DOWN", TABLE_SCROLLS_COLOUR, TABLE_SCROLLS_THICKNES)

        if (table.get_h_scroll() != None):
            draw_custom_elements(screenshot_copy_debug, [table.get_h_scroll()], "h_scroll", TABLE_SCROLLS_COLOUR, TABLE_SCROLLS_THICKNES)
            draw_custom_elements(screenshot_copy_debug, [table.get_h_scroll().get_first_button()], "LEFT", TABLE_SCROLLS_COLOUR, TABLE_SCROLLS_THICKNES)
            draw_custom_elements(screenshot_copy_debug, [table.get_h_scroll().get_second_button()], "RIGHT", TABLE_SCROLLS_COLOUR, TABLE_SCROLLS_THICKNES)

        # draw custom cell
        #draw_custom_elements(screenshot_copy_debug, [table.get_cell_by_adress(1, 2)], "table_cell", TABLE_COLOUR, TABLE_SCROLLS_THICKNES)

        # draw all cells
        #for cell in table.get_cells_area_element().get_list_cells():
        #    contours_helper.DrawRectangleByListXYWH(screenshot_copy_debug, [cell.get_roi_element().get_element_features()], TABLE_CELLS_COLOUR, TABLE_CELLS_THICKNES)

def draw_elements(screenshot_copy_debug, screenshot_elements) -> np.ndarray:
    screenshot_copy_debug = general_helpers.extend_screenshot_by_rigth_border(screenshot_copy_debug, 120)

    draw_tables(screenshot_copy_debug, screenshot_elements)
    draw_listbox(screenshot_copy_debug, screenshot_elements)
    draw_custom_elements(screenshot_copy_debug, screenshot_elements.get_elements(), "button", BUTTONS_COLOUR, BUTTONS_THICKNES)
    draw_custom_elements(screenshot_copy_debug, screenshot_elements.get_elements(), "input", INPUT_COLOUR, INPUT_THICKNES)
    draw_custom_elements(screenshot_copy_debug, screenshot_elements.get_elements(), "radiobutton", RADIOBUTTON_COLOUR, RADIOBUTTON_THICKNES)
    draw_custom_elements(screenshot_copy_debug, screenshot_elements.get_elements(), "combobox", COMBOBOX_COLOUR, COMBOBOX_THICKNES)
    draw_custom_elements(screenshot_copy_debug, screenshot_elements.get_elements(), "checkbox", CHECKBOX_COLOUR, CHECKBOX_THICKNES)
    draw_custom_elements(screenshot_copy_debug, screenshot_elements.get_elements(), "slider", SLIDER_COLOUR, SLIDER_THICKNES)

    draw_pattern_matching_elements(screenshot_copy_debug, screenshot_elements.get_elements(), (0, 255, 0), 2)

    #general_helpers.show(screenshot_copy_debug)

    return screenshot_copy_debug
