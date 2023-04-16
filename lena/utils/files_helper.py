import os
import pickle
import cv2
from lena.utils import general_helpers
from lena.utils.logging_manager import LoggingManager


def get_absolute_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath("__file__")), relative_path).replace("\\", "/").replace("api/", "")

def load_model(model_path, absolut_path=False):
    try:
        if absolut_path:
            model_file = open(get_absolute_path(model_path), "rb")
        else:
            model_file = open(model_path, "rb")

        model = pickle.load(model_file)
        model_file.close()

        return model

    except FileNotFoundError as ex:
        raise FileNotFoundError(LoggingManager().log_error(ex.strerror + ":" + ex.filename))

def save_model(model, model_path):
    dbfile = open(f"{model_path}.model", "ab")
    pickle.dump(model, dbfile)
    dbfile.close()

def load_image(file_path):
    try:
        return cv2.imread(file_path)
    except FileNotFoundError as ex:
        raise FileNotFoundError(LoggingManager().log_error(ex.strerror + ":" + ex.filename))

def load_grayscale_image(file_path):
    try:
        return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError as ex:
        raise FileNotFoundError(LoggingManager().log_error(ex.strerror + ":" + ex.filename))

def load_grayscale_images(file_paths):
    images = []

    for path in file_paths:
        images.append(load_grayscale_image(path))

    return images

def load_grayscale_images_from_folder(folder_path):
    files = []

    folder_path = get_absolute_path(folder_path)

    folder_files = os.listdir(folder_path)
    for file_name in folder_files:
        full_file_path = os.path.join(folder_path, file_name)
        template = load_grayscale_image(full_file_path)
        files.append(template)

    return files

def save_image(roi, most_common_label="_", path=r""):
    cv2.imwrite(path + "\\" + str(most_common_label) + "_" + str(general_helpers.generate_random_string(9)) + ".png", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))