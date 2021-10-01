import cv2

class ImageLoaders():
    @staticmethod
    def LoadImage(path):
        image = cv2.imread(path, 0)
        return image