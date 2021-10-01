import cv2
from Helpers.ImageConverters import ImageConverters

class MSER():

    def GetRegionsAndBoundingBoxesByMSER(image):
        img = image.copy()
        img = ImageConverters.ConvertToBW(img)

        mser = cv2.MSER_create()
        regions, boundingBoxes = mser.detectRegions(img)

        return regions, boundingBoxes

    def DrawRectanglesForMSER(boundingBoxes, image):
        for box in boundingBoxes:
            x, y, w, h = box;
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            #this is for checking each region
            #image = vis[y:y + h, x:x + w]