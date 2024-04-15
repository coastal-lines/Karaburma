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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

from karaburma.elements.features.basic_element_features import BasicElementFeatures
from karaburma.elements.features.listbox_element_features import ListboxElementFeatures
from karaburma.elements.features.table.table_element_features import TableElementFeatures
from karaburma.utils import files_helper
from karaburma.utils.config_manager import ConfigManager
from karaburma.utils.logging_manager import LoggingManager
from karaburma.utils.train import train_models_utils


def train_basic_model(samples_directory, dimension, method=""):
    features, labels = train_models_utils.get_basic_dataset_features_and_labels(samples_directory, dimension)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

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
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

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
                                   ConfigManager().config.elements_parameters.common_element.preprocessing["sample_dimension"],
                                    "100")

    files_helper.save_model(model, model_path)

def train_and_save_model_for_listbox_element(model_path):
    model = train_listbox_model(files_helper.get_absolute_path(ConfigManager().config.samples_path["listbox_samples"]), "100")
    files_helper.save_model(model, model_path)

def train_and_save_model_for_table_element(model_path):
    model = train_table_model(files_helper.get_absolute_path(ConfigManager().config.samples_path["table_samples"]))
    files_helper.save_model(model, model_path)

def debug_basic_model_visualization_confusion_matrix():
    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    ConfigManager(config_path)

    samples = []
    labels = []
    samples_directory = files_helper.get_absolute_path(ConfigManager().config.samples_path["basic_samples"])
    dimension = ConfigManager().config.elements_parameters.common_element.preprocessing["sample_dimension"]
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
    svm_model = svm.SVC(probability=True, kernel="linear", C=5.0)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    accuracy = str(accuracy_score(y_test, y_pred))
    print("accuracy: ", accuracy)

    classifier = train_basic_model()

    cm = confusion_matrix(y_test, y_pred, labels=svm_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = svm_model.classes_)
    disp.plot()
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

def debug_listbox_model_visualization_confusion_matrix():
    image_paths = []
    labels = []

    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    ConfigManager(config_path)

    samples_directory = files_helper.get_absolute_path(ConfigManager().config.samples_path["listbox_samples"])

    for folder_name in os.listdir(samples_directory):
        folder_path = os.path.join(samples_directory, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                image_paths.append(file_path)

                if (folder_name == "listbox"):
                    labels.append(0)
                if (folder_name == "non_listbox"):
                    labels.append(1)

    features = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        concatenated_features = ListboxElementFeatures(None, None, None, None, None).prepare_features_for_listbox(image)
        features.append(concatenated_features)

    label_vector = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, label_vector, test_size=0.3, random_state=42)

    svm_model = svm.SVC(probability=True, class_weight='balanced', kernel="linear", C=1.0)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    accuracy = str(accuracy_score(y_test, y_pred))
    print("accuracy: ", accuracy)

    cm = confusion_matrix(y_test, y_pred, labels=svm_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = svm_model.classes_)
    disp.plot()
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    print("confusion_matrix: ", cm)

def debug_table_model_visualization_confusion_matrix():
    image_paths = []
    labels = []

    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    ConfigManager(config_path)

    samples_directory = files_helper.get_absolute_path(ConfigManager().config.samples_path["table_samples"])

    for folder_name in os.listdir(samples_directory):
        folder_path = os.path.join(samples_directory, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                image_paths.append(file_path)
                labels.append(folder_name)

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

    cm = confusion_matrix(y_test, svm_predictions, labels=svm_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = svm_model.classes_)
    disp.plot()
    plt.show()

    cm = confusion_matrix(y_test, svm_predictions)
    print("confusion_matrix: ", cm)

def debug_table_data_visualization_pca():
    image_paths = []
    labels = []

    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    ConfigManager(config_path)

    samples_directory = files_helper.get_absolute_path(ConfigManager().config.samples_path["table_samples"])

    for folder_name in os.listdir(samples_directory):
        folder_path = os.path.join(samples_directory, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                image_paths.append(file_path)
                labels.append(folder_name)

    corner_features = []

    for image_path in image_paths:
        image = cv2.imread(image_path)

        concatenated_features = TableElementFeatures(None, None, None, None, None).prepare_features_for_table_image(
            image)
        corner_features.append(concatenated_features)

    # Step 4: Prepare the feature matrix
    feature_matrix = corner_features

    # Step 5: Prepare the label vector
    label_vector = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, label_vector, test_size=0.1, random_state=42)

    # Step 6: Apply PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    svm_model = svm.SVC(probability=True, class_weight='balanced')
    svm_model.fit(X_train, y_train)

    # Step 8: Optionally, you can visualize the PCA-transformed data
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for label in np.unique(y_train):
        plt.scatter(X_train_pca[y_train == label, 0], X_train_pca[y_train == label, 1], label=label)
    plt.title('PCA of Table Images')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

def debug_basic_data_visualization_pca():
    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    ConfigManager(config_path)

    samples = []
    labels = []
    samples_directory = files_helper.get_absolute_path(ConfigManager().config.samples_path["basic_samples"])
    dimension = ConfigManager().config.elements_parameters.common_element.preprocessing["sample_dimension"]
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
    svm_model = svm.SVC(probability=True, kernel="linear", C=5.0)
    svm_model.fit(X_train, y_train)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(samples)

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], label=label)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization of Image Features')
    plt.legend()
    plt.show()

def debug_pca_listboxes():
    image_paths = []
    labels = []

    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    ConfigManager(config_path)

    samples_directory = files_helper.get_absolute_path(ConfigManager().config.samples_path["listbox_samples"])

    for folder_name in os.listdir(samples_directory):
        folder_path = os.path.join(samples_directory, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                image_paths.append(file_path)

                if (folder_name == "listbox"):
                    labels.append(0)
                if (folder_name == "non_listbox"):
                    labels.append(1)

    features = []
    samples=[]
    for image_path in image_paths:
        image = cv2.imread(image_path)
        concatenated_features = ListboxElementFeatures(None, None, None, None, None).prepare_features_for_listbox(image)
        features.append(concatenated_features)
        samples.append(np.array(concatenated_features).flatten())

    label_vector = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, label_vector, test_size=0.3, random_state=42)

    svm_model = svm.SVC(probability=True, class_weight='balanced', kernel="linear", C=1.0)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(samples)


    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], label=label)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization of Image Features')
    plt.legend()
    plt.show()

def grid_search_for_model():
    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    ConfigManager(config_path)

    samples = []
    labels = []
    samples_directory = files_helper.get_absolute_path(ConfigManager().config.samples_path["basic_samples"])
    dimension = ConfigManager().config.elements_parameters.common_element.preprocessing["sample_dimension"]
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

    param_grid = {
        'C': [0.1, 1, 5, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': [0.001, 0.01, 0.1, 1]
    }

    # Создайте сетку параметров
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)

    # Примените GridSearch
    grid_search.fit(X_train, y_train)

    # Получите лучшие параметры
    best_params = grid_search.best_params_

    print(best_params)


#debug_basic_model_visualization_confusion_matrix()
#debug_listbox_model_visualization_confusion_matrix()
#debug_table_model_visualization_confusion_matrix()
#debug_basic_data_visualization_pca()
#debug_table_data_visualization_pca()
#debug_pca_listboxes()
#grid_search_for_model()


ConfigManager(os.path.join(files_helper.get_project_root_path(), "config.json"))
samples_directory = files_helper.get_absolute_path(ConfigManager().config.samples_path["basic_samples"])
dimension = ConfigManager().config.elements_parameters.common_element.preprocessing["sample_dimension"]
train_basic_model(samples_directory, dimension)
#train_listbox_model(samples_directory, dimension)
#train_table_model(samples_directory)