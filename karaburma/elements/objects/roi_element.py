from karaburma.utils import general_helpers


class RoiElement:
    def __init__(self, roi, x, y, w, h, class_feature=None):
        self._roi = roi
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._class_feature = class_feature

    def get_element_features(self):
        return self._x, self._y, self._w, self._h

    def get_class_feature(self):
        return self._class_feature

    def update_roi(self, updated_roi_area):
        self.set_roi(updated_roi_area)

    def get_roi(self):
        return self._roi

    def set_roi(self, roi):
        self._roi = roi

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def update_w(self, value):
        self._w += value

    def get_w(self):
        return self._w

    def get_h(self):
        return self._h

    def get_shape(self):
        return self._w, self._h

    def update_element_roi_area_by_screenshot(self):
        x, y, w, h = self.get_element_features()
        new_roi = general_helpers.do_screenshot()
        new_roi = new_roi[y:y + h, x:x + w, :]
        self.update_roi(new_roi)

    def update_element_roi_area_by_image(self, img):
        x, y, w, h = self.get_element_features()
        new_roi = img[y:y + h, x:x + w, :]
        self.update_roi(new_roi)
