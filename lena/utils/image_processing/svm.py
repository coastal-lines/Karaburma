from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import Bunch

from lena.utils.image_processing import filters_helper
from lena.utils.image_processing.filters_helper import convert_to_grayscale

d = (30, 100)

def prepare_image(image):
    image = filters_helper.Erosion(image)
    #image = image.copy() > threshold_local(image, block_size=9, offset=9)
    image = resize(image, d, anti_aliasing=False, mode='reflect')
    result = image

    return result

def load_image_files(container_path):

    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            image = imread(file)
            gr = convert_to_grayscale(image)
            img = prepare_image(gr)
            flat_data.append(img.flatten())
            images.append(img)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    image_dataset = Bunch(data=flat_data,target=target,target_names=categories,images=images,DESCR=descr)
    return image_dataset, categories