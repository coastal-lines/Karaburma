from lena.utils import general_helpers


class RoiElement:
    def __init__(self, roi, x, y, w, h, class_feature=None):
        self.__roi = roi
        self.__x = x
        self.__y = y
        self.__w = w
        self.__h = h
        self.__class_feature = class_feature
        