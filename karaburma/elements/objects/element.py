from typing import Optional

from karaburma.elements.objects.roi_element import RoiElement


class Element:
    def __init__(self, label: str, prediction_value: float, roi_element: RoiElement):
        self.__label = label
        self.__prediction_value = str("%.2f" % prediction_value)
        self.__roi_element = roi_element

    def get_label(self) -> str:
        return self.__label

    def get_prediction_value(self) -> str:
        return self.__prediction_value

    def get_roi_element(self) -> RoiElement:
        return self.__roi_element

    def get_centroid(self) -> tuple:
        return (int(self.get_roi_element().get_x() + (self.get_roi_element().get_w() // 2)),
                int(self.get_roi_element().get_y() + (self.get_roi_element().get_h() // 2)))

    def update_prediction_value(self, prediction_value: float):
        self.__prediction_value = str("%.2f" % prediction_value)
