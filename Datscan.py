from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def main():
    # Set seed
    np.random.seed(0)

    # Create the training & test sets from files
    train = pd.read_csv("data/all_visits_practice.csv")

    # Diagnostics
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
    # new_features(test)

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

    # # Predict
    # predictions = rf.predict(test[predictors])
    #
    # # Create submission and output
    # submission = pd.DataFrame({
    #     "PassengerId": test["PassengerId"],
    #     "Survived": predictions
    # })
    # submission.to_csv("data/kaggle.csv", index=False)

    # Metrics:
    print("Random Forest Metrics: \n")

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
    # scores = cross_validation.cross_val_score(rf, train_predictors, train_target, cv=3)
    # print("Cross validated score: ")
    # print(scores.mean())

    # Out of bag estimate
    print("OOB score: ")
    print(rf.oob_score_)

    # Output roc auc score
    # train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
    # traindata, validatedata = train[train['is_train']==True], train[train['is_train']==False]
    # x_traindata = traindata[predictors]
    # y_traindata = traindata["NP3BRADY"]
    # x_validatedata = validatedata[predictors]
    # y_validatedata = validatedata["NP3BRADY"]
    # rf.fit(x_traindata, y_traindata)
    # disbursed = rf.predict_proba(x_validatedata)
    # fpr, tpr, _ = roc_curve(y_validatedata, disbursed[:,1])
    # roc_auc = auc(fpr, tpr)
    # print("Roc_auc score (I don't fully understand this metric): ")
    # print(roc_auc)

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(train_predictors, train_target)

    # Print a confusion matrix for random forest
    y_pred_rf = rf.fit(X_train, y_train).predict(X_test)
    print("\nConfusion matrix (rows: actual, cols: prediction)")
    print(confusion_matrix(y_test, y_pred_rf))

    # Classification report
    # print("\nClassification report:")
    # print(classification_report(y_test, y_pred))

    # Accuracy scores of ensemble of Random Forest, Logistic Regression, and SVM
    ensemble(rf, X_train, y_train, X_test, y_test, True, False)

    # Base score of weighted ensemble of RF, LR, and SVM
    ensemble(rf, train_predictors, train_target, train_predictors, train_target, False, True)


# Generate new features
# def new_features(data):


def ensemble(rf, X_train, y_train, X_test, y_test, accuracy, base_score):
    # Random forest
    y_pred_rf = rf.fit(X_train, y_train).predict(X_test)
    if accuracy: print("\nAccuracy score of random forest: " + str(accuracy_score(y_test, y_pred_rf)))

    # Logistic regression
    lr = LogisticRegression()
    y_pred_lr = lr.fit(X_train, y_train).predict(X_test)
    if accuracy:
        print("Accuracy score of logistic regression: " + str(accuracy_score(y_test, y_pred_lr)))

    # SVM
    svm = SVC()
    y_pred_svm = svm.fit(X_train, y_train).predict(X_test)
    if accuracy: print("Accuracy score of SVM: " + str(accuracy_score(y_test, y_pred_svm)))

    # Decision tree
    dt = DecisionTreeClassifier()
    y_pred_dt = dt.fit(X_train, y_train).predict(X_test)
    if accuracy:
        print("Accuracy score of decision tree: " + str(accuracy_score(y_test, y_pred_dt)))

    # Weighted ensemble of RF, LR, and SVM
    y_pred_ensemble = (y_pred_rf + y_pred_lr + y_pred_svm) / 3
    if accuracy:
        print("Accuracy score of weighted ensemble of RF, LR, and SVM: " + str(accuracy_score(y_test, y_pred_ensemble)))

    # Base score
    if base_score:
        print("Base score of weighted ensemble of RF, LR, and SVM: " +str(accuracy_score(y_train, y_pred_ensemble)))

    return y_pred_ensemble


if __name__ == "__main__":
    main()
