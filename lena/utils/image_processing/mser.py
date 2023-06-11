import cv2
import matplotlib.pyplot as plt

def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def GetRegionsAndBoundingBoxesByMSER(image):
    img = image.copy()
    mser = cv2.MSER_create()
    regions, boundingBoxes = mser.detectRegions(img)

    return regions, boundingBoxes


def DrawRectanglesForMSER(boundingBoxes, image):
    for box in boundingBoxes:
        x, y, w, h = box;
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # this is for checking each region
        # image = vis[y:y + h, x:x + w]

    return image

query_img = cv2.imread(r's4.bmp')
query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
regions, boundingBoxes = GetRegionsAndBoundingBoxesByMSER(query_img)
image = DrawRectanglesForMSER(boundingBoxes, query_img)
show(image)