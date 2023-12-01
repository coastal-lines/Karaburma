import os
import cv2
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.io import imread as sk_image_reader
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from karaburma.elements.features.basic_element_features import BasicElementFeatures
from karaburma.elements.features.listbox_element_features import ListboxElementFeatures
from karaburma.elements.features.table.table_element_features import TableElementFeatures
from karaburma.utils import files_helper
from karaburma.utils.config_manager import ConfigManager
from karaburma.utils.logging_manager import LoggingManager

#config_reader = ConfigReader()

def train_basic_model(samples_directory, dimension, method=""):
    samples = []
    labels = []

    for folder_name in os.listdir(samples_directory):
        folder_path = os.path.join(samples_directory, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                image = sk_image_reader(os.path.join(folder_path, filename))
                feature = BasicElementFeatures(None).prepare_features_for_basic_elements(image, dimension)
                samples.append(np.array(feature).flatten())
                labels.append(folder_name)

    features = np.array(samples)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

    if (method == "100"):
        X_train = features
        y_train = labels

    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    classifier = svm.SVC(probability=True, class_weight='balanced', kernel="linear", C=5.0)
    classifier.fit(X_train_flattened, y_train)

    y2 = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y2)
    print("Accuracy is", accuracy)
    LoggingManager().log_information(f"Basic model was trained by samples from \n {samples_directory} \n Accuracy is {accuracy}")

    return classifier, X_train, y_train

def train_listbox_model(samples_directory, positive_samples_folder, negative_samples_folder, method = "100"):
    image_paths = []
    labels = []

    for folder_name in os.listdir(samples_directory):
        folder_path = os.path.join(samples_directory, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                image_paths.append(file_path)

                if (folder_name == positive_samples_folder):
                    labels.append(0)
                if (folder_name == negative_samples_folder):
                    labels.append(1)

    features = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        concatenated_features = ListboxElementFeatures(None, None, None, None, None).prepare_features_for_listbox(image)
        features.append(concatenated_features)

    label_vector = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, label_vector, test_size=0.3, random_state=42)

    if (method == "100"):
        X_train = features
        y_train = labels

    svm_model = svm.SVC(probability=True, class_weight='balanced', kernel="linear", C=1.0)
    svm_model.fit(X_train, y_train)

    y2 = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y2)
    LoggingManager().log_information(f"Listbox model was trained by samples from \n {samples_directory} \n Accuracy is {accuracy}")

    return svm_model

def train_table_model(samples_directory):
    image_paths = []
    labels = []

    for folder_name in os.listdir(samples_directory):
        folder_path = os.path.join(samples_directory, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                image_paths.append(file_path)

    corner_features = []

    for image_path in image_paths:
        image = cv2.imread(image_path)

        concatenated_features = TableElementFeatures(None, None, None, None, None).prepare_features_for_table_image(image)
        corner_features.append(concatenated_features)

    # Step 4: Prepare the feature matrix
    feature_matrix = corner_features

    # Step 5: Prepare the label vector
    label_vector = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, label_vector, test_size=0.1, random_state=42)

    svm_model = svm.SVC(probability=True, class_weight='balanced')
    svm_model.fit(X_train, y_train)

    svm_predictions = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print("Accuracy: " + str(svm_accuracy))
    LoggingManager().log_information(f"Table model was trained by samples from \n {samples_directory} \n Accuracy is {svm_accuracy}")

    return svm_model

def train_and_save_model_for_basic_elements(model_path):
    model, _, _ = train_basic_model(files_helper.get_absolute_path(ConfigManager().config.samples_path["basic_samples"]),
                                   ConfigManager().config.elements_parameters.common_element.preprocessing["sample_dimension"],
                                    "100")

    files_helper.save_model(model, model_path)

def train_and_save_model_for_listbox_element(model_path, samples_directory, positive_samples_folder):
    model = train_listbox_model(files_helper.get_absolute_path(ConfigManager().config.samples_path["listbox_samples"]),
                                samples_directory,
                                positive_samples_folder,
                                "100")

    files_helper.save_model(model, model_path)

def train_and_save_model_for_table_element(model_path):
    model = train_table_model(files_helper.get_absolute_path(ConfigManager().config.samples_path["table_samples"]))
    files_helper.save_model(model, model_path)

def debug_visualization():
    global_config = ConfigManager()

    samples = []
    labels = []
    samples_directory = files_helper.get_absolute_path(global_config.config.samples_path["basic_samples"])
    dimension = global_config.config.elements_parameters.common_element.preprocessing["sample_dimension"]
    for folder_name in os.listdir(samples_directory):
        folder_path = os.path.join(samples_directory, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                image = sk_image_reader(os.path.join(folder_path, filename))
                feature = BasicElementFeatures(None).prepare_features_for_basic_elements(image, dimension)
                samples.append(np.array(feature).flatten())
                labels.append(folder_name)

    features = np.array(samples)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)
    svm_model = svm.SVC(probability=True, kernel="linear", C=5.0, decision_function_shape='ovr')
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

#debug_visualization()

