from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix


def main():
    # Create the training & test sets from files
    train = pd.read_csv("data/all_visits_practice.csv")

    # Generate new features
    # new_features(train)
    # new_features(test)

    # The columns we'll use to predict the target
    predictors = ["CAUDATE_R", "CAUDATE_L", "PUTAMEN_R", "PUTAMEN_L"]

    # Prepare predictors
    train_predictors = train[predictors]

    # Prepare target
    train_target = train["NP3BRADY"]

    # Create and train the random forest
    # Multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=25, min_samples_leaf=25, oob_score=True)

    # Fit the algorithm to the data
    y_pred = rf.fit(train_predictors, train_target)

    # # Predict
    # predictions = rf.predict(test[predictors])
    #
    # # Create submission and output
    # submission = pd.DataFrame({
    #     "PassengerId": test["PassengerId"],
    #     "Survived": predictions
    # })
    # submission.to_csv("data/kaggle.csv", index=False)

    # Perform feature selection
    selector = SelectKBest(f_classif, k='all')
    selector.fit(train_predictors, train_target)

    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = -np.log10(selector.pvalues_)
    print("Univariate feature selection:")
    print(predictors)
    print(scores)

    # Feature importances
    print("Feature importances:")
    print(rf.feature_importances_)

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

    # Split the data into a training set and a test set, and print a confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(train_predictors, train_target, random_state=1)
    y_pred = rf.fit(X_train, y_train).predict(X_test)
    print("\nConfusion matrix: ")
    print(confusion_matrix(y_test, y_pred))


# Generate new features
# def new_features(data):


if __name__ == "__main__":
    main()
