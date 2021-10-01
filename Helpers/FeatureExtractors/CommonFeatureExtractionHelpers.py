import cv2
import numpy as np

class CommonFeatureExtractionHelpers():
    def MakeMaskForROI(image):
        mask = np.zeros(image.shape, dtype=np.uint8)
        # constants only for debugging
        # drawing rectangle only for debugging
        cv2.rectangle(mask, (13, 226), (273, 827), (255), thickness=-1)
        return mask