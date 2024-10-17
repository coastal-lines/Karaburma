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

def debug_grid_search_for_model(X_train: np.ndarray, y_train: np.ndarray):
    param_grid = {
        'C': [0.1, 1, 5, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': [0.001, 0.01, 0.1, 1]
    }

    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(best_params)

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

    label_colors = {"listbox": "lime", "non_listbox": "deepskyblue"}

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

    label_colors = {"table1": "lime", "nontable2": "deepskyblue"}

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], label=label, color=label_colors[label], edgecolors='k')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('PCA visualization of Table Features')
    plt.legend()
    plt.show()

def debug_basic_learning_curve(samples_directory: str, dimension: List[int]):
    features, labels = train_models_utils.get_basic_dataset_features_and_labels(samples_directory, dimension)
    X_train, X_test, y_train, y_test = train_models_utils.get_train_and_test_data(features, labels, test_size=0.1, random_state=42)
    classifier = train_models_module.train_basic_model(samples_directory, dimension)

    train_sizes, train_scores, test_scores = learning_curve(estimator=classifier,
                                                            X=X_train,
                                                            y=y_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 5),
                                                            cv=5)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.1, 1.0])
    plt.show()

def debug_listbox_learning_curve(samples_directory: str):
    features, labels = train_models_utils.get_listbox_dataset_features_and_labels(samples_directory)
    X_train, X_test, y_train, y_test = train_models_utils.get_train_and_test_data(features, labels, test_size=0.3, random_state=42)
    classifier = train_models_module.train_listbox_model(samples_directory)

    train_sizes, train_scores, test_scores = learning_curve(estimator=classifier,
                                                            X=X_train,
                                                            y=y_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            cv=5)

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', label='Training accuracy')
    plt.plot(train_sizes, test_mean, color='green', marker='s', linestyle='--', label='Validation accuracy')
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.1, 1.0])
    plt.show()

def debug_table_learning_curve(samples_directory: str):
    features, labels = train_models_utils.get_table_dataset_features_and_labels(samples_directory)
    X_train, X_test, y_train, y_test = train_models_utils.get_train_and_test_data(features, labels, test_size=0.1, random_state=42)
    classifier = train_models_module.train_table_model(samples_directory)

    train_sizes, train_scores, test_scores = learning_curve(estimator=classifier,
                                                            X=X_train,
                                                            y=y_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            cv=5)

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', label='Training accuracy')
    plt.plot(train_sizes, test_mean, color='green', marker='s', linestyle='--', label='Validation accuracy')
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.6, 1.0])
    plt.show()

def debug_table_accuracy_and_c_validation_curve(samples_directory: str):
    features, labels = train_models_utils.get_table_dataset_features_and_labels(samples_directory)
    X_train, X_test, y_train, y_test = train_models_utils.get_train_and_test_data(features, labels, test_size=0.1, random_state=42)

    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    train_acc = []
    test_acc = []

    for C in C_values:
        svm_model = SVC(class_weight = 'balanced', kernel='rbf', C=C, random_state=42)
        scores = cross_val_score(estimator=svm_model, X=X_train, y=y_train, cv=5)

        train_acc.append(np.mean(scores))

        svm_model.fit(X_train, y_train)
        test_acc.append(svm_model.score(X_test, y_test))

    plt.plot(C_values, train_acc, label='Train accuracy')
    plt.plot(C_values, test_acc, label='Test accuracy')
    plt.fill_between(C_values, train_acc, test_acc, color='skyblue', alpha=0.3)
    plt.xscale('log')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs C')
    plt.show()

def debug_basic_accuracy_and_c_validation_curve(samples_directory: str, dimension: List[int]):
    features, labels = train_models_utils.get_basic_dataset_features_and_labels(samples_directory, dimension)
    X_train, X_test, y_train, y_test = train_models_utils.get_train_and_test_data(features, labels, test_size=0.3, random_state=42)

    # Desired values of C parameter
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    # Arrays to store accuracy on the training and test sets
    train_acc = []
    test_acc = []

    for C in C_values:
        # Create SVM model for each C value
        svm_model = SVC(class_weight="balanced", kernel="rbf", decision_function_shape="ovo", C=C, random_state=42)
        scores = cross_val_score(estimator=svm_model, X=X_train, y=y_train, cv=5)

        # Average accuracy on the training set
        train_acc.append(np.mean(scores))

        # Train the model on the entire training set and evaluate its accuracy on the test set
        svm_model.fit(X_train, y_train)
        test_acc.append(svm_model.score(X_test, y_test))

    plt.plot(C_values, train_acc, label='Train accuracy')
    plt.plot(C_values, test_acc, label='Test accuracy')
    plt.fill_between(C_values, train_acc, test_acc, color='skyblue', alpha=0.3)
    plt.xscale('log')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs C')
    plt.show()

def adaboost_comparison_svm_table_model(samples_directory: str):
    features, labels = train_models_utils.get_table_dataset_features_and_labels(samples_directory)
    X_train, X_test, y_train, y_test = train_models_utils.get_train_and_test_data(features, labels, test_size=0.1, random_state=42)
    svm_classifier = train_models_utils.get_classifier((X_train, X_test, y_train, y_test), probability=True, class_weight='balanced', kernel="rbf", C=1.0)
    svm_classifier_accuracy = train_models_utils.get_model_accuracy(svm_classifier, X_test, y_test)

    adaboost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)
    adaboost_classifier.fit(X_train, y_train)
    adaboost_pred = adaboost_classifier.predict(X_test)
    adaboost_classifier_accuracy = accuracy_score(y_test, adaboost_pred)

    plt.bar(['SVM', 'AdaBoost'], [svm_classifier_accuracy, adaboost_classifier_accuracy])
    plt.ylabel('Accuracy')
    plt.title('Comparison of SVM and AdaBoost Accuracies')
    plt.show()

def train_model_and_get_metrics(model_type: str, samples_directory: str, class_weight="balanced", kernel="rbf", c=1.0, pos_label=None, average=None):
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

    if (type(precision) is np.ndarray):
        print(f"--- \n model is: {model_type} \n accuracy: {accuracy}")

        for class_label, precision,  in zip(svm_classifier.classes_, precision):
            print(f"class_label: {class_label}, precision: {precision}")

        for class_label, recall,  in zip(svm_classifier.classes_, recall):
            print(f"class_label: {class_label}, recall: {recall}")

        for class_label, f1, in zip(svm_classifier.classes_, f1):
            print(f"class_label: {class_label}, f1: {f1}")
    else:
        print(f"--- \n model is: {model_type} \n accuracy: {accuracy} \n precision: {precision}, recall: {recall}, f1-score: {f1} \n --- \n ")

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    ConfigManager(config_path)

    basic_samples_dim = ConfigManager().config.elements_parameters.common_element.preprocessing["sample_dimension"]
    basic_samples = files_helper.get_absolute_path(ConfigManager().config.samples_path["basic_samples"])
    listbox_samples = files_helper.get_absolute_path(ConfigManager().config.samples_path["listbox_samples"])
    table_samples = files_helper.get_absolute_path(ConfigManager().config.samples_path["table_samples"])

    #train_model_and_get_metrics("table", table_samples, "balanced", "rbf", 1.0, "table1", None)
    #train_model_and_get_metrics("listbox", listbox_samples, "balanced", "linear", 1.0, "listbox", None)
    #train_model_and_get_metrics("basic", basic_samples, "balanced", "linear", 5.0, None, "weighted")
    train_model_and_get_metrics("basic", basic_samples, "balanced", "linear", 5.0, None, None)


    #debug_basic_features_visualization_pca(basic_samples, basic_samples_dim)
    #debug_table_features_visualization_pca(table_samples)
    #debug_table_learning_curve(table_samples)
    #debug_listbox_learning_curve(listbox_samples)
    #debug_basic_learning_curve(basic_samples, basic_samples_dim)
    #debug_table_accuracy_and_c_validation_curve(table_samples)
    #debug_basic_accuracy_and_c_validation_curve(basic_samples, basic_samples_dim)
    #debug_pca(table_samples)