import json
from karaburma.utils import general_helpers, files_helper


def convert_object_into_json(image_source):

    dict_json = dict()
    dict_json["w"], dict_json["h"] = image_source.get_current_image_source_dimension()
    dict_json["elements"] = []

    for i in range(len(image_source.get_elements())):
        current_element = dict()
        current_label = image_source.get_elements()[i].get_label()

        if(current_label == "button"
            or current_label == "checkbox"
            or current_label == "checkbox"
            or current_label == "input"
            or current_element == "radiobutton"):

            if image_source.get_elements()[i].get_text() is not None:
                current_element["text"] = image_source.get_elements()[i].get_text()
            else:
                current_element["text"] = ""

        if(current_label == "listbox"):
            current_element["text"] = image_source.get_elements()[i].get_list_text()
            if (image_source.get_elements()[i].full_text_area is not None):
                current_element["full_img_base64"] = files_helper.image_to_base64(image_source.get_elements()[i].full_text_area.get_roi_element().get_roi())

        if(current_label == "table"):
            current_table_cells = []
            for cell in image_source.get_elements()[i].get_cells_area_element().get_list_cells():
                current_cell = dict()
                current_cell["adress"] = cell.get_adress()
                current_cell["text"] = cell.get_text()
                current_cell["centre_x"], current_cell["centre_y"] = cell.get_centroid()
                current_table_cells.append(current_cell)

            current_element["cells"] = current_table_cells

            # If table has expanded table
            if (image_source.get_elements()[i].get_full_table_area() is not None):

                expanded_table_element = dict()

                expanded_table_element["full_img_base64"] = files_helper.image_to_base64(
                    image_source.get_elements()[i].get_full_table_area().get_roi_element().get_roi())

                current_expanded_table_cells = []
                for cell in image_source.get_elements()[i].get_full_table_area().get_cells_area_element().get_cells_area_element().get_list_cells():
                    current_cell = dict()
                    current_cell["adress"] = cell.get_adress()
                    current_cell["text"] = cell.get_text()
                    current_cell["centre_x"], current_cell["centre_y"] = cell.get_centroid()
                    current_expanded_table_cells.append(current_cell)

                expanded_table_element["cells"] = current_expanded_table_cells

                current_element["full_table"] = expanded_table_element
            else:
                current_element["full_table"] = None

        if(getattr(image_source.get_elements()[i], 'get_v_scroll', None)):
            if (image_source.get_elements()[i].get_v_scroll() is not None):
                current_up_button = dict()
                current_up_button["centre_x"], current_up_button["centre_y"] = image_source.get_elements()[i].get_v_scroll().get_first_button().get_centroid()

                current_down_button = dict()
                current_down_button["centre_x"], current_down_button["centre_y"] = image_source.get_elements()[i].get_v_scroll().get_second_button().get_centroid()

                current_v_scroll = dict()
                current_v_scroll["w"] = image_source.get_elements()[i].get_roi_element().get_w()
                current_v_scroll["h"] = image_source.get_elements()[i].get_roi_element().get_h()
                current_v_scroll["x"] = image_source.get_elements()[i].get_roi_element().get_x()
                current_v_scroll["y"] = image_source.get_elements()[i].get_roi_element().get_y()
                current_v_scroll["up_button"] = current_up_button
                current_v_scroll["down_button"] = current_down_button

                current_element["v_scroll"] = current_v_scroll

        if (getattr(image_source.get_elements()[i], 'get_h_scroll', None)):
            if (image_source.get_elements()[i].get_h_scroll() is not None):
                current_left_button = dict()
                current_left_button["centre_x"], current_left_button["centre_y"] = image_source.get_elements()[i].get_h_scroll().get_first_button().get_centroid()

                current_rigth_button = dict()
                current_rigth_button["centre_x"], current_rigth_button["centre_y"] = image_source.get_elements()[i].get_h_scroll().get_second_button().get_centroid()

                current_h_scroll = dict()
                current_h_scroll["w"] = image_source.get_elements()[i].get_roi_element().get_w()
                current_h_scroll["h"] = image_source.get_elements()[i].get_roi_element().get_h()
                current_h_scroll["x"] = image_source.get_elements()[i].get_roi_element().get_x()
                current_h_scroll["y"] = image_source.get_elements()[i].get_roi_element().get_y()
                current_h_scroll["left_button"] = current_left_button
                current_h_scroll["rigth_button"] = current_rigth_button

                current_element["h_scroll"] = current_h_scroll

        current_element["id"] = str(i)
        current_element["label"] = current_label
        current_element["prediction"] = image_source.get_elements()[i].get_prediction_value()
        current_element["w"] = int(image_source.get_elements()[i].get_roi_element().get_w())
        current_element["h"] = int(image_source.get_elements()[i].get_roi_element().get_h())
        current_element["x"] = int(image_source.get_elements()[i].get_roi_element().get_x())
        current_element["y"] = int(image_source.get_elements()[i].get_roi_element().get_y())

        # Element image roi to base64
        current_element["orig_img_base64"] = files_helper.image_to_base64(image_source.get_elements()[i].get_roi_element().get_roi())

        dict_json["elements"].append(current_element)

    try:
        return json.loads(json.dumps(dict_json, indent=2))
    except TypeError as ex:
        print(ex)
        print("Some of objects have non-serializable value")
