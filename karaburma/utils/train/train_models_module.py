from typing import List
from sklearn.svm._base import BaseLibSVM

from karaburma.utils import files_helper
from karaburma.utils.config_manager import ConfigManager
from karaburma.utils.logging_manager import LoggingManager
from karaburma.utils.train import train_models_utils


def train_basic_model(samples_directory: str, dimension: List[int], method="") -> BaseLibSVM:
    features, labels = train_models_utils.get_basic_dataset_features_and_labels(samples_directory, dimension)
    X_train, X_test, y_train, y_test = train_models_utils.get_train_and_test_data(features, labels, test_size=0.1, random_state=42)

    if (method == "100"):
        X_train = features
        y_train = labels

    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    classifier = train_models_utils.get_classifier((X_train_flattened, X_test, y_train, y_test), probability=True, class_weight='balanced', kernel="linear", C=5.0)

    accuracy = train_models_utils.get_model_accuracy(classifier, X_test, y_test)
    LoggingManager().log_information(f"Basic model was trained by samples from \n {samples_directory} \n Accuracy is {accuracy}")

    return classifier

def train_listbox_model(samples_directory, method="100"):
    features, labels = train_models_utils.get_listbox_dataset_features_and_labels(samples_directory)
    X_train, X_test, y_train, y_test = train_models_utils.get_train_and_test_data(features, labels, test_size=0.3, random_state=42)

    if (method == "100"):
        X_train = features
        y_train = labels

    classifier = train_models_utils.get_classifier((X_train, X_test, y_train, y_test), probability=True, class_weight='balanced', kernel="linear", C=1.0)

    accuracy = train_models_utils.get_model_accuracy(classifier, X_test, y_test)
    LoggingManager().log_information(f"Listbox model was trained by samples from \n {samples_directory} \n Accuracy is {accuracy}")

    return classifier

def train_table_model(samples_directory):
    features, labels = train_models_utils.get_table_dataset_features_and_labels(samples_directory)
    X_train, X_test, y_train, y_test = train_models_utils.get_train_and_test_data(features, labels, test_size=0.1, random_state=42)

    classifier = train_models_utils.get_classifier((X_train, X_test, y_train, y_test), probability=True, class_weight='balanced', kernel="rbf", C=1.0)

    accuracy = train_models_utils.get_model_accuracy(classifier, X_test, y_test)
    LoggingManager().log_information(f"Table model was trained by samples from \n {samples_directory} \n Accuracy is {accuracy}")

    return classifier

def train_and_save_model_for_basic_elements(model_path):
    model = train_basic_model(files_helper.get_absolute_path(ConfigManager().config.samples_path["basic_samples"]),
                                   ConfigManager().config.elements_parameters.common_element.preprocessing["sample_dimension"], "100")
    files_helper.save_model(model, model_path)

def train_and_save_model_for_listbox_element(model_path):
    model = train_listbox_model(files_helper.get_absolute_path(ConfigManager().config.samples_path["listbox_samples"]), "100")
    files_helper.save_model(model, model_path)

def train_and_save_model_for_table_element(model_path):
    model = train_table_model(files_helper.get_absolute_path(ConfigManager().config.samples_path["table_samples"]))
    files_helper.save_model(model, model_path)
