from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report
import pandas as pd
import numpy as np
import re
import operator
import warnings
import matplotlib.pyplot as plt


def main():
    # Set seed
    np.random.seed(1)

    # Create the training & test sets from files
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # Prepare train data
    # train["Age"] = train["Age"].fillna(train["Age"].median())
    train.loc[train["Sex"] == "male", "Sex"] = 0
    train.loc[train["Sex"] == "female", "Sex"] = 1
    train["Embarked"] = train["Embarked"].fillna("C")
    train.loc[train["Embarked"] == "S", "Embarked"] = 0
    train.loc[train["Embarked"] == "C", "Embarked"] = 1
    train.loc[train["Embarked"] == "Q", "Embarked"] = 2
    train["Cabin"] = train["Cabin"].fillna("Z")

    # Prepare test data
    # test["Age"] = test["Age"].fillna(test["Age"].median())
    test.loc[test["Sex"] == "male", "Sex"] = 0
    test.loc[test["Sex"] == "female", "Sex"] = 1
    test["Embarked"] = test["Embarked"].fillna("S")
    test.loc[test["Embarked"] == "S", "Embarked"] = 0
    test.loc[test["Embarked"] == "C", "Embarked"] = 1
    test.loc[test["Embarked"] == "Q", "Embarked"] = 2
    # test["Fare"] = test["Fare"].fillna(test["Fare"].median())
    test.Fare = test.Fare.map(lambda x: np.nan if x == 0 else x)
    classmeans = test.pivot_table('Fare', index='Pclass', aggfunc='mean')
    test.Fare = test[['Fare', 'Pclass']].apply(lambda x: classmeans[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1)
    test["Cabin"] = test["Cabin"].fillna("Z")

    # Discretize fare
    # bins_and_binned_fare = pd.qcut(train.Fare, 10, retbins=True)
    # bins = bins_and_binned_fare[1]
    # train.Fare = bins_and_binned_fare[0]
    # test.Fare = pd.cut(test.Fare, bins)

    # Generate new features
    new_features(train)
    new_features(test)

    # Predict and impute missing ages with Random Forest
    age_predictors = ["Pclass", "Sex", "Fare", "SibSp", "Parch", "Embarked", "FamilySize", "Title", "FamilyId", "Deck"]
    age_train = train.loc[train.Age.isnull() == 0, age_predictors]
    age_train.append(test.loc[test.Age.isnull() == 0, age_predictors])
    age_target = train.loc[train.Age.isnull() == 0, "Age"]
    age_target.append(test.loc[test.Age.isnull() == 0, "Age"])
    age_test = train.loc[train.Age.isnull(), age_predictors]
    rf = RandomForestRegressor(n_estimators=150, min_samples_split=4, min_samples_leaf=2, oob_score=True)
    age_predictions = rf.fit(age_train, age_target).predict(age_test)
    train.loc[train.Age.isnull(), "Age"] = age_predictions.astype(float)
    age_predictions = rf.predict(test.loc[test.Age.isnull(), age_predictors])
    test.loc[test.Age.isnull(), "Age"] = age_predictions

    # Add in Child column
    train["Child"] = 0
    train.loc[train["Age"] < 18, "Child"] = 1

    # Add in Mother column
    train["Mother"] = 0
    train.loc[(train["Age"] > 18) & (train["Sex"] == 1) & (train["Parch"] > 0) & (train["Title"] != 2), "Mother"] = 1

    # Add in Child column
    test["Child"] = 0
    test.loc[test["Age"] < 18, "Child"] = 1

    # Add in Mother column
    test["Mother"] = 0
    test.loc[(test["Age"] > 18) & (test["Sex"] == 1) & (test["Parch"] > 0) & (test["Title"] != 2), "Mother"] = 1

    # The columns we'll use to predict the target
    predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked",
                  "FamilySize", "Title", "FamilyId", "Deck"]

    # Prepare predictors
    train_predictors = train[predictors]

    # Prepare target
    train_target = train["Survived"]

    # Create and train the random forest
    # Multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=150, min_samples_split=4, min_samples_leaf=2, oob_score=True)

    # Fit the algorithm to the data
    rf.fit(train_predictors, train_target)

    # Perform feature selection
    selector = SelectKBest(f_classif, k=5)
    selector.fit(train_predictors, train_target)

    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = -np.log10(selector.pvalues_)
    print("Univariate feature selection:")
    for feature, imp in zip(predictors, scores):
        print(feature, imp)

    print("\nRANDOM FOREST METRICS:")

    # Feature importances
    print("\nFeature importances:")
    for feature, imp in zip(predictors, rf.feature_importances_):
        print(feature, imp)

    # Base estimate
    print("\nBase score: ")
    print(rf.score(train_predictors, train_target))

    # Cross validate our RF and output the mean score
    scores = cross_validation.cross_val_score(rf, train_predictors, train_target, cv=4)
    print("Cross validated score: ")
    print(scores.mean())

    # Out of bag estimate
    print("OOB score: ")
    print(rf.oob_score_)

    # Split the data into a training set and a test set, and train the model
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_predictors, train_target, test_size=0.25)
    rf.fit(X_train, y_train)

    # Output roc auc score
    disbursed = rf.predict_proba(X_test)
    print("Roc_auc score:")
    print(roc_auc_score(y_test, disbursed[:, 1]))

    # Print a confusion matrix
    y_pred = rf.predict(X_test)
    print("\nConfusion matrix (rows: actual, cols: prediction)")
    print(confusion_matrix(y_test, y_pred))

    # Ensemble
    ens = ensemble(train_predictors, train_target)

    # Predict
    predictions = ens.fit(train_predictors, train_target).predict(test[predictors])

    # Map predictions to outcomes (only possible outcomes are 1 and 0)
    predictions[predictions > .5] = 1
    predictions[predictions <= .5] = 0

    # Create submission and output
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv("data/kaggle.csv", index=False)


# Generate new features
def new_features(data):
    # Generating a familysize column
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    # The .apply method generates a new series
    data["NameLength"] = data["Name"].apply(lambda x: len(x))

    # Add a new Title feature:

    # A function to get the title from a name.
    def get_title(name):
        # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""

    # Get all the titles
    titles = data["Name"].apply(get_title)

    # Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                     "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Dona": 10, "Sir": 9, "Capt": 7, "Ms": 2}
    for k, v in title_mapping.items():
        titles[titles == k] = v

    # Verify that we converted everything.
    # print(pd.value_counts(titles))

    # Add in the title column.
    data["Title"] = titles

    # Add a new Deck feature

    # Get all the decks (first character of cabin)
    decks = data["Cabin"].str[0]

    # Map each character (A - F) to an int
    deck_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8, "Z": 9}
    for k, v in deck_mapping.items():
        decks[decks == k] = v

    # Verify that we converted everything.
    # print(pd.value_counts(decks))

    # Add in the decks column
    data["Deck"] = decks

    # Add a new Family Id feature:

    # A dictionary mapping family name to id
    family_id_mapping = {}

    # A function to get the id given a row
    def get_family_id(row):
        # Find the last name by splitting on a comma
        last_name = row["Name"].split(",")[0]
        # Create the family id
        family_id = "{0}{1}".format(last_name, row["FamilySize"])
        # Look up the id in the mapping
        if family_id not in family_id_mapping:
            if len(family_id_mapping) == 0:
                current_id = 1
            else:
                # Get the maximum id from the mapping and add one to it if we don't have an id
                current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
            family_id_mapping[family_id] = current_id
        return family_id_mapping[family_id]

    # Get the family ids with the apply method
    family_ids = data.apply(get_family_id, axis=1)

    # There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
    family_ids[data["FamilySize"] < 3] = -1

    # Add in the FamilyId column.
    data["FamilyId"] = family_ids

    # Making FamilySize discrete based on 3 categories: singleton, small, and large families
    data.loc[data["FamilySize"] == 1, "FamilySize"] = 0
    data.loc[(data["FamilySize"].between(2, 4)), "FamilySize"] = 1
    data.loc[data["FamilySize"] > 4, "FamilySize"] = 2


# Create ensemble
def ensemble(X, y):
    # Classifiers
    rf = RandomForestClassifier(n_estimators=150, min_samples_split=4, min_samples_leaf=2, oob_score=True)
    lr = LogisticRegression()
    svm = SVC(probability=True)
    gb = GradientBoostingClassifier(n_estimators=25, max_depth=3)

    # Ensemble
    eclf = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('gb', gb)], voting='soft', weights=[3, 1, 3])

    print("\nENSEMBLE METRICS:\n")

    # Cross validation accuracies
    print("Cross validation:\n")
    for clf, label in zip([rf, lr, svm, gb, eclf], ['Random Forest', 'Logistic Regression', 'SVM', 'Gradient Boosting', 'Ensemble of RF, LR, and Grad Boost']):
        scores = cross_validation.cross_val_score(clf, X, y, cv=4, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)
    print("\n75/25 split: \n")
    y_pred_ensemble = eclf.fit(X_train, y_train).predict(X_test)

    # Accuracies
    for clf, label in zip([rf, lr, svm, gb, eclf], ['Random Forest', 'Logistic Regression', 'SVM', 'Gradient Boosting', 'Ensemble of RF, LR, and Grad Boost']):
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        print("Accuracy: %0.2f [%s]" % (accuracy_score(y_test, y_pred), label))

    # Classification report
    print("\nClassification report:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(classification_report(y_test, y_pred_ensemble))

    # Print a confusion matrix for ensemble
    print("\nConfusion matrix of ensemble (rows: true, cols: pred)")

    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(4)
        plt.xticks(tick_marks, [0, 1, 2, 3], rotation=45)
        plt.yticks(tick_marks, [0, 1, 2, 3])
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_ensemble)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization:')
    print(cm)

    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix:')
    print(cm_normalized)
    plt.figure()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

    plt.show()

    return eclf


if __name__ == "__main__":
    main()
