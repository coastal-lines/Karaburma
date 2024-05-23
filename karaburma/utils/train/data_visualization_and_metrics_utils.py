import os
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score

from karaburma.utils import files_helper
from karaburma.utils.config_manager import ConfigManager
from karaburma.utils.train import train_models_module, train_models_utils

def extract_metrics_for_each_class_from_multiclassed_svm_classifier(labels, precision_per_class, recall_per_class, f1_per_class):
    for class_label, precision, recall, f1 in zip(labels, precision_per_class, recall_per_class, f1_per_class):
        print("---")
        print(f"class: {class_label}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1-score: {f1}")
        print("---")

def train_binary_model_and_get_metrics(model_type: str, samples_directory: str, class_weight="balanced", kernel="rbf", c=1.0, pos_label=None, average=None):
    features, labels = None, None

    match model_type:
        case "table":
            features, labels = train_models_utils.get_table_dataset_features_and_labels(samples_directory)
        case "listbox":
            features, labels = train_models_utils.get_listbox_dataset_features_and_labels(samples_directory)
        case "basic":
            features, labels = train_models_utils.get_basic_dataset_features_and_labels(samples_directory, [79, 24])

    X_train, X_test, y_train, y_test = train_models_utils.get_train_and_test_data(features, labels, test_size=0.1, random_state=42)
    svm_classifier = train_models_utils.get_classifier((X_train, X_test, y_train, y_test), probability=True, class_weight=class_weight, kernel=kernel, C=c)
    y_pred = train_models_utils.get_model_prediction(svm_classifier, X_test)

    accuracy = train_models_utils.get_model_accuracy(svm_classifier, X_test, y_test)
    precision = train_models_utils.get_model_precision_score(y_test, y_pred, pos_label, average)
    recall = train_models_utils.get_model_recall_score(y_test, y_pred, pos_label, average)
    f1 = train_models_utils.get_model_f1_score(y_test, y_pred, pos_label, average)

    print(f"--- \n Model is: {model_type} \n accuracy: {accuracy} \n precision: {precision}, recall: {recall}, f1-score: {f1} \n --- \n ")

    if (model_type == "basic"):
        extract_metrics_for_each_class_from_multiclassed_svm_classifier(labels, accuracy, precision, recall)

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    ConfigManager(config_path)

    basic_samples_dim = ConfigManager().config.elements_parameters.common_element.preprocessing["sample_dimension"]
    basic_samples = files_helper.get_absolute_path(ConfigManager().config.samples_path["basic_samples"])
    listbox_samples = files_helper.get_absolute_path(ConfigManager().config.samples_path["listbox_samples"])
    table_samples = files_helper.get_absolute_path(ConfigManager().config.samples_path["table_samples"])

    #train_binary_model_and_get_metrics("table", table_samples, "balanced", "rbf", 1.0, "table1", None)
    #train_binary_model_and_get_metrics("listbox", listbox_samples, "balanced", "linear", 1.0, "listbox", None)
    train_binary_model_and_get_metrics("basic", basic_samples, "balanced", "linear", 5.0, None, None)

    #debug_basic_features_visualization_pca(basic_samples, basic_samples_dim)
    #debug_table_features_visualization_pca(table_samples)
    #debug_table_learning_curve(table_samples)
    #debug_listbox_learning_curve(listbox_samples)
    #debug_basic_learning_curve(basic_samples, basic_samples_dim)
    #debug_table_accuracy_and_c_validation_curve(table_samples)
    #debug_basic_accuracy_and_c_validation_curve(basic_samples, basic_samples_dim)
    #debug_pca(table_samples)