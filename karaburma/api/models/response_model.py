class RootKaraburmaResponse:
    def __init__(self, w, h, elements, debug_screenshot):
        self.w = w
        self.h = h
        self.elements = [Element(**element) for element in elements]
        self.debug_screenshot = debug_screenshot

class Element:
    def __init__(self, cells, full_table, v_scroll, h_scroll, id, label, prediction, w, h, x, y, orig_img_base64, text=None):
        self.cells = [Cell(**cell) for cell in cells]
        self.full_table = full_table
        self.v_scroll = Scroll(**v_scroll)
        self.h_scroll = Scroll(**h_scroll)
        self.id = id
        self.label = label
        self.prediction = prediction
        self.w = w
        self.h = h
        self.x = x
        self.y = y
        self.orig_img_base64 = orig_img_base64
        self.text = text

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