class RootKaraburmaResponse:
    def __init__(self, w, h, elements, listbox_elements, table_elements,debug_screenshot):
        self.w = w
        self.h = h
        self.basic_elements = [Element(**element) for element in elements]
        self.listbox_elements = [ListBoxElement(**element) for element in listbox_elements]
        self.table_elements = [TableElement(**element) for element in table_elements]
        self.debug_screenshot = debug_screenshot


class Element:
    def __init__(self, id, x, y, w, h, label, prediction, orig_img_base64, text):
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.prediction = prediction
        self.orig_img_base64 = orig_img_base64
        self.text = text


class ListBoxElement(Element):
    def __init__(self, id, x, y, w, h, label, prediction, orig_img_base64, text, v_scroll, full_img_base64, full_listbox):
        super().__init__(id, x, y, w, h, label, prediction, orig_img_base64, text)
        self.v_scroll = v_scroll
        self.full_img_base64 = full_img_base64
        self.full_listbox = full_listbox


class TableElement(Element):
    def __init__(self, id, x, y, w, h, label, prediction, orig_img_base64, v_scroll, h_scroll, cells, full_table):
        super().__init__(id, x, y, w, h, label, prediction, orig_img_base64, "")
        self.v_scroll = v_scroll
        self.h_scroll = h_scroll
        self.cells = cells
        self.full_table = full_table


class Cell:
    def __init__(self, adress, text, centre_x, centre_y):
        self.adress = adress
        self.text = text
        self.centre_x = centre_x
        self.centre_y = centre_y


class Scroll:
    def __init__(self, w, h, x, y, first_button, second_button):
        self.w = w
        self.h = h
        self.x = x
        self.y = y
        self.first_button = Button(**first_button)
        self.second_button = Button(**second_button)


class Button:
    def __init__(self, centre_x, centre_y):
        self.centre_x = centre_x
        self.centre_y = centre_y