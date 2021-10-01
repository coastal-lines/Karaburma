class CommonMethods():

    def CropImage(image, x, y, w, h):
        img = image[y:y + h, x:x + w]
        return img