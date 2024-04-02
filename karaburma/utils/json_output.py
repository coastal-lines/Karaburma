import json

from karaburma.utils import files_helper


def set_general_attributes_to_element(current_element, element, current_element_number):
    current_element["id"] = str(current_element_number)
    current_element["x"] = int(element.get_roi_element().get_x())
    current_element["y"] = int(element.get_roi_element().get_y())
    current_element["w"] = int(element.get_roi_element().get_w())
    current_element["h"] = int(element.get_roi_element().get_h())
    current_element["label"] = element.get_label()
    current_element["prediction"] = element.get_prediction_value()
    current_element["orig_img_base64"] = files_helper.image_to_base64(element.get_roi_element().get_roi())

def add_simple_element(element, current_element_number):
    current_element = dict()
    set_general_attributes_to_element(current_element, element, current_element_number)

    if element.get_text() is not None:
        current_element["text"] = element.get_text()
    else:
        current_element["text"] = ""

    return current_element

def get_v_scroll_element(element):
    current_up_button = dict()
    current_up_button["centre_x"], current_up_button["centre_y"] = element.get_v_scroll().get_first_button().get_centroid()

    current_down_button = dict()
    current_down_button["centre_x"], current_down_button["centre_y"] = element.get_v_scroll().get_second_button().get_centroid()

    current_v_scroll = dict()
    set_general_attributes_to_element(current_v_scroll, element, 0)

    current_v_scroll["first_button"] = current_up_button
    current_v_scroll["second_button"] = current_down_button

    return current_v_scroll

def get_h_scroll_element(element):
    current_left_button = dict()
    current_left_button["centre_x"], current_left_button["centre_y"] = element.get_h_scroll().get_first_button().get_centroid()

    current_rigth_button = dict()
    current_rigth_button["centre_x"], current_rigth_button["centre_y"] = element.get_h_scroll().get_second_button().get_centroid()

    current_h_scroll = dict()
    set_general_attributes_to_element(current_h_scroll, element, 0)

    current_h_scroll["first_button"] = current_left_button
    current_h_scroll["second_button"] = current_rigth_button

    return current_h_scroll

def add_listbox_element(element, current_element_number):
    current_element = dict()
    set_general_attributes_to_element(current_element, element, current_element_number)

    current_element["text"] = element.get_list_text()

    if (getattr(element, 'get_v_scroll', None)):
        if (element.get_v_scroll() is not None):
            current_element["v_scroll"] = get_v_scroll_element(element)

    if (getattr(element, 'get_h_scroll', None)):
        if (element.get_h_scroll() is not None):
            current_element["h_scroll"] = get_h_scroll_element(element)

    current_element["full_img_base64"] = files_helper.image_to_base64(element.get_roi_element().get_roi())

    if (element.full_text_area is not None):
        listbox_full_text_area_element = dict()

        set_general_attributes_to_element(listbox_full_text_area_element, element.full_text_area, 0)
        listbox_full_text_area_element["text"] = element.full_text_area.get_list_text()

        listbox_full_text_area_element["full_img_base64"] = files_helper.image_to_base64(element.full_text_area.get_roi_element().get_roi())

        current_element["full_listbox"] = listbox_full_text_area_element

    return current_element

def add_table_element(element, current_element_number):
    current_element = dict()
    set_general_attributes_to_element(current_element, element, current_element_number)

    if (getattr(element, 'get_v_scroll', None)):
        if (element.get_v_scroll() is not None):
            current_element["v_scroll"] = get_v_scroll_element(element)

    if (getattr(element, 'get_h_scroll', None)):
        if (element.get_h_scroll() is not None):
            current_element["h_scroll"] = get_h_scroll_element(element)

    current_table_cells = []
    for cell in element.get_cells_area_element().get_list_cells():
        current_cell = dict()
        current_cell["centre_x"], current_cell["centre_y"] = cell.get_centroid()
        current_cell["text"] = cell.get_text()
        current_cell["address"] = cell.get_adress()
        current_table_cells.append(current_cell)
    current_element["cells"] = current_table_cells

    return current_element

def convert_object_into_json(image_source, screenshot_copy_debug=None):
    dict_json = dict()
    dict_json["w"], dict_json["h"] = image_source.get_current_image_source_dimension()
    dict_json["elements"] = []
    dict_json["listbox_elements"] = []
    dict_json["table_elements"] = []

    for i in range(len(image_source.get_elements())):
        current_element = dict()
        current_label = image_source.get_elements()[i].get_label()

        if(current_label == "button"
            or current_label == "checkbox"
            or current_label == "combobox"
            or current_label == "input"
            or current_label == "radiobutton"
            or current_label == "slider"):
            dict_json["elements"].append(add_simple_element(image_source.get_elements()[i], i))

        if(current_label == "listbox"):
            dict_json["listbox_elements"].append(add_listbox_element(image_source.get_elements()[i], i))

        if(current_label == "table"):
            dict_json["table_elements"].append(add_table_element(image_source.get_elements()[i], i))

            # If table has expanded table
            if (image_source.get_elements()[i].get_full_table_area() is not None):
                expanded_table_element = add_table_element(image_source.get_elements()[i].get_full_table_area(), 0)
                dict_json["table_elements"][-1]["full_table"] = expanded_table_element

            else:
                dict_json["table_elements"][-1]["full_table"] = None

    # Put copy of the original image with debug information
    if(screenshot_copy_debug is not None):
        dict_json["debug_screenshot"] = files_helper.image_to_base64(screenshot_copy_debug)

    try:
        return json.loads(json.dumps(dict_json, indent=2))
    except TypeError as ex:
        print(ex)
        print("Error during converting ImageSource object into json format.\n Some of objects have non-serializable value.")