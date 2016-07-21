from sklearn import preprocessing, cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error, \
    mean_squared_error, median_absolute_error, r2_score
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt


def describe_data(data, info=False, describe=False, value_counts=None, unique=None,
                  univariate_feature_selection=None, description=None):
    # Data diagnostics
    if description is not None:
        print("\n" + description)

    # Info
    if info:
        print("\nInfo:")
        print(data.info())

    # Description
    if describe:
        print("\nDescribe:")
        print(data.describe())

    # Value counts
    if value_counts is not None:
        for feature in value_counts:
            print("\nValue Counts [" + feature + "]")
            print(pd.value_counts(data[feature]))

    # Unique values
    if unique is not None:
        for feature in unique:
            print("\nUnique [" + feature + "]")
            print(data[feature].unique())

    # Univariate feature selection
    if univariate_feature_selection is not None:
        # Extract predictors and target
        predictors = univariate_feature_selection[0]
        target = univariate_feature_selection[1]

        # Perform feature selection
        selector = SelectKBest(f_classif, k="all")
        selector.fit(data[predictors], data[target])

        # Get the raw p-values for each feature, and transform from p-values into scores
        scores = -np.log10(selector.pvalues_)
        print("\nUnivariate Feature Selection:")
        for feature, imp in zip(predictors, scores):
            print(feature, imp)


def clean_data(data, encode_auto=None, encode_man=None, fillna=None, scale_features=None):
    # Automatically encode features to numeric
    if encode_auto is not None:
        for feature in encode_auto:
            data[feature] = preprocessing.LabelEncoder().fit_transform(data[feature])

    # Manually encode features to numeric
    if encode_man is not None:
        for feature, encoding in encode_man.items():
            for cur_value, new_value in encoding.items():
                data.loc[data[feature] == cur_value, feature] = new_value

    # Fill missing values
    if fillna is not None:
        for feature, method in fillna:
            if method == "median":
                data[feature] = data[feature].fillna(data[feature].median())
            if method == "mean":
                data[feature] = data[feature].fillna(data[feature].mean())
            if method == "mode":
                data[feature] = data[feature].fillna(data[feature].mode())

    # Scale values based on min and max
    if scale_features is not None:
        data[scale_features] = MinMaxScaler().fit_transform(data[scale_features])


def metrics(data, predictors, target, algs, alg_names, feature_importances=None, base_score=None, oob_score=None,
            cross_val=None, folds=5, scoring="accuracy", split_accuracy=None, split_classification_report=None,
            split_confusion_matrix=None, plot=True, grid_search_params=None, description="METRICS:"):
    # Feature importances
    def print_feature_importances(alg, name):
        print("Feature Importances [" + name + "]")
        for feature, imp in zip(predictors, alg.feature_importances_):
            print(feature, imp)

    # Base score estimate
    def print_base_score(alg, name):
        score = alg.score(data[predictors], data[target])
        print("Base Score: {} [{}]".format(score, name))

    # Out of bag estimate
    def print_oob_score(alg, name):
        score = alg.oob_score_
        print("OOB Score: {} [{}]".format(score, name))

    # Cross validation
    def print_cross_val(alg, name):
        scores = cross_validation.cross_val_score(alg, data[predictors], data[target], cv=folds, scoring=scoring)
        if scoring == "root_mean_squared_error":
            print("Cross Validation: {:0.2f} (+/- {:0.2f}) [{}]".format(abs(scores.mean())**0.5, scores.std(), name))
        else:
            print("Cross Validation: {:0.2f} (+/- {:0.2f}) [{}]".format(abs(scores.mean()), scores.std(), name))

    # Split accuracy
    def print_split_accuracy(alg, name, split_name, X_train, X_test, y_train, y_test):
        y_pred = alg.fit(X_train, y_train).predict(X_test)
        if scoring == "accuracy":
            print("{}: {:0.2f} [{}]".format(split_name, accuracy_score(y_test, y_pred), name))
        elif scoring == "mean_absolute_error":
            print("{}: {:0.2f} [{}]".format(split_name, mean_absolute_error(y_test, y_pred), name))
        elif scoring == "root_mean_squared_error":
            print("{}: {:0.2f} [{}]".format(split_name, mean_squared_error(y_test, y_pred)**0.5, name))
        elif scoring == "mean_squared_error":
            print("{}: {:0.2f} [{}]".format(split_name, mean_squared_error(y_test, y_pred), name))
        elif scoring == "median_absolute_error":
            print("{}: {:0.2f} [{}]".format(split_name, median_absolute_error(y_test, y_pred), name))
        elif scoring == "r2":
            print("{}: {:0.2f} [{}]".format(split_name, r2_score(y_test, y_pred), name))

    # Split classification report
    def print_split_classification_report(alg, name, X_train, X_test, y_train, y_test):
        print("Classification Report [" + name + "]")
        y_pred = alg.fit(X_train, y_train).predict(X_test)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(classification_report(y_test, y_pred))

    # Split confusion matrix
    def print_split_confusion_matrix(alg, name, X_train, X_test, y_train, y_test, display_plot=True):
        # Print algorithm name
        print("Confusion Matrix [" + name + "]")

        # Create predictions
        y_pred = alg.fit(X_train, y_train).predict(X_test)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        print("Not Normalized:")
        print(cm)

        # Normalize the confusion matrix by row (i.e by the number of samples in each class)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized:")
        print(cm_normalized)

        if display_plot:
            # Configure confusion matrix plot
            def plot_confusion_matrix(cm, title="Confusion matrix", cmap=plt.cm.Blues):
                plt.imshow(cm, interpolation="nearest", cmap=cmap)
                plt.title(title)
                plt.colorbar()
                tick_marks = np.arange(4)
                plt.xticks(tick_marks, [0, 1, 2, 3], rotation=45)
                plt.yticks(tick_marks, [0, 1, 2, 3])
                plt.tight_layout()
                plt.ylabel("True label")
                plt.xlabel("Predicted label")

            # Plot normalized confusion matrix
            plt.figure()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_confusion_matrix(cm_normalized, title="Normalized Confusion Matrix\n[{}]".format(name))

            # Show confusion matrix plots
            plt.show()

    # Grid search
    def print_grid_search(alg, name, params):
        # Print algorithm being grid searched
        print("Grid Search [{}]".format(name))

        # Run grid search
        if scoring == "root_mean_squared_error":
            grid_search = GridSearchCV(estimator=alg, cv=folds, param_grid=params, scoring="mean_squared_error")
        else:
            grid_search = GridSearchCV(estimator=alg, cv=folds, param_grid=params, scoring=scoring)
        grid_search.fit(data[predictors], data[target])

        # Print best parameters and score
        print(grid_search.best_params_)
        if scoring == "root_mean_squared_error":
            print("Cross Validation: {}".format(grid_search.best_score_**0.5))
        else:
            print("Cross Validation: {}".format(grid_search.best_score_))

    # Print description of metrics
    if description is not None:
        print("\n" + description)

    # Fit algorithms /just once/ for base score and oob score (as opposed to redundantly refitting)
    if base_score is not None or oob_score is not None:
        # If none, set lengths to zero
        if base_score is None:
            len_base_score = 0
        else:
            len_base_score = len(base_score)
        if oob_score is None:
            len_oob_score = 0
        else:
            len_oob_score = len(oob_score)

        # Max length
        i = max(len_base_score, len_oob_score)

        # Fit wherever base score or oob score are true
        for i in range(i):
            if len_base_score < i + 1:
                if oob_score[i]:
                    algs[i].fit(data[predictors], data[target])
            elif len_oob_score < i + 1:
                if base_score[i]:
                    algs[i].fit(data[predictors], data[target])
            else:
                if base_score[i] or oob_score[i]:
                    algs[i].fit(data[predictors], data[target])

    # Call respective methods
    if feature_importances is not None:
        print("")
        for i, val in enumerate(feature_importances):
            if val:
                print_feature_importances(algs[i], alg_names[i])
    if base_score is not None:
        print("")
        for i, val in enumerate(base_score):
            if val:
                print_base_score(algs[i], alg_names[i])
    if oob_score is not None:
        print("")
        for i, val in enumerate(oob_score):
            if val:
                print_oob_score(algs[i], alg_names[i])
    if cross_val is not None:
        print("")
        for i, val in enumerate(cross_val):
            if val:
                print_cross_val(algs[i], alg_names[i])

    # If split is needed
    if split_accuracy is not None or split_classification_report is not None or split_confusion_matrix is not None:
        # Split the data into a training set and a test set
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[predictors], data[target],
                                                                             test_size=1.0 / folds)

        # Print ratio of split
        split_name = "{:g}/{:g} Split: ".format(100 - 100 / folds, 100 / folds)

        # Call respective methods
        if split_accuracy:
            print("")
            for i, val in enumerate(split_accuracy):
                if val:
                    print_split_accuracy(algs[i], alg_names[i], split_name, X_train, X_test, y_train, y_test)
        if split_classification_report:
            print("")
            for i, val in enumerate(split_classification_report):
                if val:
                    print_split_classification_report(algs[i], alg_names[i], X_train, X_test, y_train, y_test)
        if split_confusion_matrix:
            print("")
            for i, val in enumerate(split_confusion_matrix):
                if val:
                    print_split_confusion_matrix(algs[i], alg_names[i], X_train, X_test, y_train, y_test, plot)

    # Finish calling respective methods
    if grid_search_params is not None:
        print("")
        for i, val in enumerate(grid_search_params):
            if val is not None:
                print_grid_search(algs[i], alg_names[i], val)


def ensemble(algs, alg_names, ensemble_name=None, in_ensemble=None, weights=None, voting="soft"):
    # Estimators for the ensemble
    estimators = []

    # Construct ensemble name
    if weights is not None:
        name = "Weighted Ensemble of "
    else:
        name = "Ensemble of "

    # Add respective algorithms to estimators and construct name
    for index, alg in enumerate(algs):
        if (in_ensemble is None) or in_ensemble[index]:
            estimators.append((alg_names[index], alg))
            name += alg_names[index] + ", "

    # Remove extra comma
    name = name[:-2]

    # Use provided name if not none
    if ensemble_name is not None:
        # Set name
        name = ensemble_name

    # Create ensemble
    alg = VotingClassifier(estimators=estimators, voting=voting, weights=weights)

    # Return ensemble and name
    return {"alg": alg, "name": name}
