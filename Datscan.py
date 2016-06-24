import warnings
from sklearn import preprocessing, cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def main():
    # Set seed
    np.random.seed(0)

    # Create the training & test sets from files
    train = pd.read_csv("data/all_visits_practice.csv")

    # Preliminary data diagnostics
    # print("Info")
    # print(train.info())
    # print("Describe")
    # print(train.describe())
    # print("Unique")
    # print(train["EVENT_ID"].unique())
    # print("Value counts EVENT_ID")
    # print(pd.value_counts(train["EVENT_ID"]))
    # print("Value counts NP3BRADY")
    # print(pd.value_counts(train["NP3BRADY"]))

    # Encode EVENT_ID to numeric
    train["EVENT_ID"] = preprocessing.LabelEncoder().fit_transform(train["EVENT_ID"])

    # Generate new features
    # new_features(train)

    # The columns we'll use to predict the target
    predictors = ["PATNO", "EVENT_ID", "CAUDATE_R", "CAUDATE_L", "PUTAMEN_R", "PUTAMEN_L"]

    # Prepare predictors
    train_predictors = train[predictors]

    # Prepare target
    train_target = train["NP3BRADY"]

    # Create and train the random forest
    # Multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=350, min_samples_split=50, min_samples_leaf=25, oob_score=True)

    # Fit the algorithm to the data
    rf.fit(train_predictors, train_target)

    # Metrics:
    print("RANDOM FOREST METRICS: \n")

    # Perform feature selection
    selector = SelectKBest(f_classif, k='all')
    selector.fit(train_predictors, train_target)

    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = -np.log10(selector.pvalues_)
    print("Univariate feature selection:")
    for feature, imp in zip(predictors, scores):
        print(feature, imp)

    # Feature importances
    print("\nFeature importances:")
    for feature, imp in zip(predictors, rf.feature_importances_):
        print(feature, imp)

    # Base estimate
    print("\nBase score: ")
    print(rf.score(train_predictors, train_target))

    # Cross validate our RF and output the mean score
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     scores = cross_validation.cross_val_score(rf, train_predictors, train_target, cv=3)
    #     print("Cross validated score: ")
    #     print(scores.mean())

    # Out of bag estimate
    print("OOB score: ")
    print(rf.oob_score_)

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_predictors, train_target)

    # Print a confusion matrix for random forest
    y_pred_rf = rf.fit(X_train, y_train).predict(X_test)
    print("\nConfusion matrix of random forest (rows: true, cols: pred)")
    print(confusion_matrix(y_test, y_pred_rf))

    # Classification report
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("\nClassification report of random forest:")
        print(classification_report(y_test, y_pred_rf))

    print("ENSEMBLE METRICS:\n")

    # Base score of weighted ensemble of RF, LR, and SVM
    ensemble(rf, train_predictors, train_target, train_predictors, train_target, False, True, False, False)

    # Accuracy scores of ensemble of Random Forest, Logistic Regression, and SVM
    ensemble(rf, X_train, y_train, X_test, y_test, True, False, True, True)


# Generate new features
# def new_features(data):


def ensemble(rf, X_train, y_train, X_test, y_test, accuracy, base_score, class_report, conf_mat):
    # Random forest
    y_pred_rf = rf.fit(X_train, y_train).predict(X_test)
    if accuracy:
        print("Accuracy score of random forest: " + str(accuracy_score(y_test, y_pred_rf)))

    # Logistic regression
    lr = LogisticRegression()
    y_pred_lr = lr.fit(X_train, y_train).predict(X_test)
    if accuracy:
        print("Accuracy score of logistic regression: " + str(accuracy_score(y_test, y_pred_lr)))

    # SVM
    svm = SVC()
    y_pred_svm = svm.fit(X_train, y_train).predict(X_test)
    if accuracy:
        print("Accuracy score of SVM: " + str(accuracy_score(y_test, y_pred_svm)))

    # Decision tree
    dt = DecisionTreeClassifier()
    y_pred_dt = dt.fit(X_train, y_train).predict(X_test)
    if accuracy:
        print("Accuracy score of decision tree: " + str(accuracy_score(y_test, y_pred_dt)))

    # Weighted ensemble of RF, LR, and SVM
    y_pred_ensemble = ((y_pred_rf + y_pred_lr + y_pred_svm) / 3).astype(int)
    if accuracy:
        print("Accuracy score of ensemble of RF, LR, and SVM: " + str(accuracy_score(y_test, y_pred_ensemble)))

    # Base score
    if base_score:
        print("Base score of ensemble of RF, LR, and SVM: " +str(accuracy_score(y_train, y_pred_ensemble)))

    # Confusion matrix
    if conf_mat:
        print("\nConfusion matrix of ensemble (rows: true, cols: pred)")
        print(confusion_matrix(y_test, y_pred_ensemble))
        print("")

    # Classification report
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if class_report:
            print("\nClassification report of ensemble of RF, LR, and SVM: ")
            print(classification_report(y_test, y_pred_ensemble))

    return y_pred_ensemble


if __name__ == "__main__":
    main()
