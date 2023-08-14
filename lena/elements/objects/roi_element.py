from lena.utils import general_helpers


class RoiElement:
    def __init__(self, roi, x, y, w, h, class_feature=None):
        self.__roi = roi
        self.__x = x
        self.__y = y
        self.__w = w
        self.__h = h
        self.__class_feature = class_feature

    def get_element_features(self):
        return self.__x, self.__y, self.__w, self.__h

    def get_class_feature(self):
        return self.__class_feature

    def update_roi(self, updated_roi_area):
        self.set_roi(updated_roi_area)

    def get_roi(self):
        return self.__roi

    def set_roi(self, roi):
        self.__roi = roi

    def get_x(self):
        return self.__x

    def get_y(self):
        return self.__y

    def update_w(self, value):
        self.__w += value

    def get_w(self):
        return self.__w

    def get_h(self):
        return self.__h

    def get_shape(self):
        return self.__w, self.__h

    def update_element_roi_area_by_screenshot(self):
        x, y, w, h = self.get_element_features()
        new_roi = general_helpers.do_screenshot()
        new_roi = new_roi[y:y + h, x:x + w, :]
        self.update_roi(new_roi)

    def update_element_roi_area_by_image(self, img):
        x, y, w, h = self.get_element_features()
        new_roi = img[y:y + h, x:x + w, :]
        self.update_roi(new_roi)
