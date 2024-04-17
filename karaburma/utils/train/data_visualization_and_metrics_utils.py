from typing import List
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

from karaburma.utils.train import train_models_utils, train_models_module


def debug_basic_model_visualization_confusion_matrix(samples_directory: str, dimension: List[int]):
    features, labels = train_models_utils.get_basic_dataset_features_and_labels(samples_directory, dimension)
    X_train, X_test, y_train, y_test = train_models_utils.get_train_and_test_data(features, labels, test_size=0.1, random_state=42)

    classifier = train_models_module.train_basic_model(samples_directory, dimension)
    prediction = train_models_utils.get_model_prediction(classifier, X_test)

    model_confusion_matrix = confusion_matrix(y_test, prediction, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=model_confusion_matrix, display_labels = classifier.classes_)
    disp.plot()
    plt.show()

def debug_listbox_model_visualization_confusion_matrix(samples_directory: str):
    features, labels = train_models_utils.get_listbox_dataset_features_and_labels(samples_directory)
    X_train, X_test, y_train, y_test = train_models_utils.get_train_and_test_data(features, labels, test_size=0.3, random_state=42)

    classifier = train_models_module.train_listbox_model(samples_directory)
    prediction = train_models_utils.get_model_prediction(classifier, X_test)

    model_confusion_matrix = confusion_matrix(y_test, prediction, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=model_confusion_matrix, display_labels=classifier.classes_)
    disp.plot()
    plt.show()

def debug_table_model_visualization_confusion_matrix(samples_directory: str):
    features, labels = train_models_utils.get_table_dataset_features_and_labels(samples_directory)
    X_train, X_test, y_train, y_test = train_models_utils.get_train_and_test_data(features, labels, test_size=0.1, random_state=42)

    classifier = train_models_module.train_table_model(samples_directory)
    prediction = train_models_utils.get_model_prediction(classifier, X_test)

    model_confusion_matrix = confusion_matrix(y_test, prediction, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=model_confusion_matrix, display_labels = classifier.classes_)
    disp.plot()
    plt.show()

def debug_basic_features_visualization_pca(samples_directory: str, dimension: List[int]):
    features, labels = train_models_utils.get_basic_dataset_features_and_labels(samples_directory, dimension)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)

    label_colors = {'button': 'red', 'checkbox': 'aqua', 'combobox': 'darkviolet', 'h_scroll': 'deepskyblue', "input": "peru", "non": "black", "radiobutton": "yellow", "slider": "hotpink"}
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], label=label, color=label_colors[label], edgecolors='k')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('PCA visualization of Basic Features')
    plt.legend()
    plt.show()

def debug_listbox_features_visualization_pca(samples_directory: str):
    features, labels = train_models_utils.get_listbox_dataset_features_and_labels(samples_directory)


    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)

    label_colors = {"listbox": "yellow", "non_listbox": "deepskyblue"}

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], label=label, color=label_colors[label], edgecolors='k')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('PCA visualization of Listbox Features')
    plt.legend()
    plt.show()

def debug_table_features_visualization_pca(samples_directory: str):
    features, labels = train_models_utils.get_table_dataset_features_and_labels(samples_directory)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)

    label_colors = {"table1": "yellow", "nontable2": "deepskyblue"}

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], label=label, color=label_colors[label], edgecolors='k')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('PCA visualization of Table Features')
    plt.legend()
    plt.show()

def grid_search_for_model(X_train: np.ndarray, y_train: np.ndarray):
    param_grid = {
        'C': [0.1, 1, 5, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': [0.001, 0.01, 0.1, 1]
    }

    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(best_params)