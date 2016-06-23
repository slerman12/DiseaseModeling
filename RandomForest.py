from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
import re
import operator
from sklearn.metrics import roc_curve, auc


def main():
    # Create the training & test sets from files
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # Prepare train data
    train["Age"] = train["Age"].fillna(train["Age"].median())
    train.loc[train["Sex"] == "male", "Sex"] = 0
    train.loc[train["Sex"] == "female", "Sex"] = 1
    train["Embarked"] = train["Embarked"].fillna("S")
    train.loc[train["Embarked"] == "S", "Embarked"] = 0
    train.loc[train["Embarked"] == "C", "Embarked"] = 1
    train.loc[train["Embarked"] == "Q", "Embarked"] = 2

    # Prepare test data
    test["Age"] = test["Age"].fillna(test["Age"].median())
    test.loc[test["Sex"] == "male", "Sex"] = 0
    test.loc[test["Sex"] == "female", "Sex"] = 1
    test["Embarked"] = test["Embarked"].fillna("S")
    test.loc[test["Embarked"] == "S", "Embarked"] = 0
    test.loc[test["Embarked"] == "C", "Embarked"] = 1
    test.loc[test["Embarked"] == "Q", "Embarked"] = 2
    test["Fare"] = test["Fare"].fillna(test["Fare"].median())

    # Generate new features
    new_features(train)
    new_features(test)

    # Print some preliminary info about the datasets
    train.info()
    print("----------------------------")
    test.info()

    # The columns we'll use to predict the target
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
    # predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    # Prepare predictors
    train_predictors = train[predictors]

    # Prepare target
    train_target = train["Survived"]

    # Create and train the random forest
    # Multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2, oob_score=True)

    # Fit the algorithm to the data
    rf.fit(train_predictors, train_target)

    # Perform feature selection
    selector = SelectKBest(f_classif, k=5)
    selector.fit(train_predictors, train_target)

    # Feature importances
    print("Feature importances:")
    print(rf.feature_importances_)

    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = -np.log10(selector.pvalues_)
    print("Univariate feature selection:")
    print(predictors)
    print(scores)

    # Base estimate
    print("\nBase score: ")
    print(rf.score(train_predictors, train_target))

    # Cross validate our RF and output the mean score
    scores = cross_validation.cross_val_score(rf, train_predictors, train_target, cv=3)
    print("Cross validated score: ")
    print(scores.mean())

    # Out of bag estimate
    print("OOB score: ")
    print(rf.oob_score_)

    # Predict
    predictions = rf.predict(test[predictors])

    # Map predictions to outcomes (only possible outcomes are 1 and 0)
    predictions[predictions > .5] = 1
    predictions[predictions <= .5] = 0

    # Create submission and output
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv("data/kaggle.csv", index=False)

    train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
    traindata, validatedata = train[train['is_train']==True], train[train['is_train']==False]
    x_traindata = traindata[predictors]
    y_traindata = traindata["Survived"]
    x_validatedata = validatedata[predictors]
    y_validatedata = validatedata["Survived"]
    rf.fit(x_traindata, y_traindata)
    disbursed = rf.predict_proba(x_validatedata)
    fpr, tpr, _ = roc_curve(y_validatedata, disbursed[:,1])
    roc_auc = auc(fpr, tpr)
    print ("\nRoc_auc score (I don't fully understand this metric): ")
    print (roc_auc)


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

    # Get all the titles and print how often each one occurs.
    titles = data["Name"].apply(get_title)
    # print(pd.value_counts(titles))

    # Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                     "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Dona": 10, "Sir": 9, "Capt": 7, "Ms": 2}
    for k, v in title_mapping.items():
        titles[titles == k] = v

    # Verify that we converted everything.
    # print(pd.value_counts(titles))

    # Add in the title column.
    data["Title"] = titles

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


if __name__ == "__main__":
    main()
