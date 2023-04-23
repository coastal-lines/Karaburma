from abc import ABC, abstractmethod
import cv2
import numpy as np

from lena.utils import general_helpers
from lena.utils.image_processing import ocr_helper, filters_helper, orb_descriptor, key_feature_matchers, homography

#TODO -> tuple[int, int]

class DisplacementBase(ABC):
    @abstractmethod
    def calculate_displacement(self, before, after=None):
        pass
