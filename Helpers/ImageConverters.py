import cv2

class ImageConverters():
    def ConvertToBW(image):
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image_bw

    def ConvertImageToNegative(image):
        image = cv2.bitwise_not(image)
        return image