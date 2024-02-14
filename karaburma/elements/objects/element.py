from karaburma.elements.objects.roi_element import RoiElement


class Element:
    def __init__(self, label: str, prediction_value: float, roi_element: RoiElement):
        self._label = label
        self._prediction_value = str("%.2f" % prediction_value)
        self._roi_element = roi_element

    def get_label(self) -> str:
        return self._label

    def get_prediction_value(self) -> str:
        return self._prediction_value

    def get_roi_element(self) -> RoiElement:
        return self._roi_element

    def get_centroid(self) -> tuple:
        return (int(self.get_roi_element().get_x() + (self.get_roi_element().get_w() // 2)),
                int(self.get_roi_element().get_y() + (self.get_roi_element().get_h() // 2)))

    def update_prediction_value(self, prediction_value: str):
        __prediction_value = str("%.2f" % prediction_value)
