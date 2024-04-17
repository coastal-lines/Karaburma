import os
from typing import List, Tuple
import numpy as np


import os
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage.io import imread as sk_image_reader
from sklearn.svm._base import BaseLibSVM

from karaburma.elements.features.basic_element_features import BasicElementFeatures
from karaburma.elements.features.listbox_element_features import ListboxElementFeatures
from karaburma.elements.features.table.table_element_features import TableElementFeatures


def get_model_prediction(classifier: BaseLibSVM, X_test: np.ndarray):
    return classifier.predict(X_test)

def get_model_accuracy(classifier: BaseLibSVM, X_test: np.ndarray, y_test: np.ndarray) -> float:
    svm_predictions = get_model_prediction(classifier, X_test)
    return accuracy_score(y_test, svm_predictions)

def get_train_and_test_data(features: np.ndarray, labels: np.ndarray, test_size: float, random_state: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)

def get_classifier(train_data: Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray], probability: bool, class_weight :str, kernel: str, C: float) -> BaseLibSVM:
    X_train, X_test, y_train, y_test = train_data

    classifier = svm.SVC(probability=probability, class_weight=class_weight, kernel=kernel, C=C)
    classifier.fit(X_train, y_train)

    return classifier

def get_basic_dataset_features_and_labels(samples_directory: str, dimension: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []

    for folder_name in os.listdir(samples_directory):
        folder_path = os.path.join(samples_directory, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                image = sk_image_reader(os.path.join(folder_path, filename))
                feature = BasicElementFeatures(None).prepare_features_for_basic_elements(image, dimension)
                features.append(np.array(feature).flatten())
                labels.append(folder_name)

    features = np.array(features)
    labels = np.array(labels)

    return features, labels

def get_listbox_dataset_features_and_labels(samples_directory: str) -> Tuple[np.ndarray, np.ndarray]:
    labels = []
    features = []

    for folder_name in os.listdir(samples_directory):
        folder_path = os.path.join(samples_directory, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                image = sk_image_reader(os.path.join(folder_path, filename))
                feature = ListboxElementFeatures(None, None, None, None, None).prepare_features_for_listbox(image)
                features.append(np.array(feature).flatten())
                labels.append(folder_name)

    features = np.array(features)
    labels = np.array(labels)

    return features, labels

def get_table_dataset_features_and_labels(samples_directory: str) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []

    for folder_name in os.listdir(samples_directory):
        folder_path = os.path.join(samples_directory, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                image = sk_image_reader(os.path.join(folder_path, filename))
                feature = TableElementFeatures(None, None, None, None, None).prepare_features_for_table_image(image)
                features.append(np.array(feature).flatten())
                labels.append(folder_name)

    features = np.array(features)
    labels = np.array(labels)

    return features, labels