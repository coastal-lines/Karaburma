from lena.elements.objects.roi_element import RoiElement


class Element:
    def __init__(self, label: str, prediction_value: float, roi_element: RoiElement):
        self.__label = label
        self.__prediction_value = str("%.2f" % prediction_value)
        self.__roi_element = roi_element

    def get_label(self) -> str:
        return self.__label

    def get_prediction_value(self) -> str:
        return self.__prediction_value
