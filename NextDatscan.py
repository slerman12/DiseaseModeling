from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import math
import MachineLearning as mL


def main():
    # Set seed
    np.random.seed(0)

    # Create the training/test set(s) from file(s)
    train = pd.read_csv("data/all_visits_practice_2.csv")

    # Preliminary data diagnostics
    mL.describe_data(data=train, describe=True, info=True, value_counts=["ONOFF", "NP3BRADY"],
                     description="PRELIMINARY DATA DIAGNOSTICS:")

    # Encode EVENT_ID to numeric
    mL.clean_data(data=train, encode_man={"EVENT_ID": {"SC": 0, "V04": 4, "V06": 6, "V10": 10}})

    # Choose On or Off
    train = train[train["ONOFF"] == 0]

    # Remove the class with only a single sample
    train = train[train.NP3BRADY != 4]

    # Generate new features
    train = generate_features(train)

    # Value counts for EVENT_ID after feature generation
    mL.describe_data(data=train, info=True, describe=True, value_counts=["EVENT_ID", "NP3BRADY_NEXT"],
                     description="AFTER FEATURE GENERATION:")

    # Predictors for the model
    predictors = ["TIME_PASSED", "EVENT_ID", "CAUDATE_R", "CAUDATE_L", "PUTAMEN_R", "PUTAMEN_L", "NP3BRADY_NOW"]

    # Target for the model
    target = "NP3BRADY_NEXT"

    # Univariate feature selection
    mL.describe_data(data=train, univariate_feature_selection=[predictors, target])

    # Algs for model
    algs = [RandomForestClassifier(n_estimators=1000, min_samples_split=50, min_samples_leaf=2, oob_score=True),
            LogisticRegression(),
            SVC(probability=True),
            GaussianNB(),
            MultinomialNB(),
            BernoulliNB(),
            KNeighborsClassifier(n_neighbors=25),
            GradientBoostingClassifier(n_estimators=10, max_depth=3)]

    # Alg names for model
    alg_names = ["Random Forest",
                 "Logistic Regression",
                 "SVM",
                 "Gaussian Naive Bayes",
                 "Multinomial Naive Bayes",
                 "Bernoulli Naive Bayes",
                 "kNN",
                 "Gradient Boosting"]

    grid_search_params = [{"n_estimators": [50, 500, 1000],
                           "min_samples_split": [25, 50, 75],
                           "min_samples_leaf": [2, 15, 25, 50]}]

    # Ensemble
    ens = mL.ensemble(algs=algs, alg_names=alg_names,
                      ensemble_name="Weighted ensemble of RF, LR, SVM, GNB, KNN, and GB",
                      in_ensemble=[True, True, True, True, False, False, True, True], weights=[3, 2, 1, 3, 1, 3],
                      voting="soft")

    # Add ensemble to algs and alg_names
    algs.append(ens["alg"])
    alg_names.append(ens["name"])

    # Display ensemble metrics
    mL.metrics(data=train, predictors=predictors, target=target, algs=algs, alg_names=alg_names,
               feature_importances=[True], base_score=[True], oob_score=[True],
               cross_val=[True, True, True, True, True, True, True, True, True],
               split_accuracy=[True, True, True, True, True, True, True, True, True],
               split_classification_report=[False, False, False, False, False, False, False, False, True],
               split_confusion_matrix=[False, False, False, False, False, False, False, False, True],
               grid_search_params=grid_search_params)


def generate_features(data):
    # Generate NP3BRADY_NOW
    data["NP3BRADY_NOW"] = data["NP3BRADY"]

    # Features in dataframe
    features = ["PATNO", "TIME_PASSED", "EVENT_ID", "CAUDATE_R", "CAUDATE_L", "PUTAMEN_R", "PUTAMEN_L", "NP3BRADY_NOW",
                "NP3BRADY_NEXT"]

    # Create new dataframe
    new_data = pd.DataFrame(columns=features)

    # Function to build new data
    def set(row):
        if row["EVENT_ID"] < 10:
            for i in range(1, 11):
                if any((data["EVENT_ID"] == row["EVENT_ID"] + i) & (data["PATNO"] == row["PATNO"])):
                    row["NP3BRADY_NEXT"] = data.loc[(data["EVENT_ID"] == row["EVENT_ID"] + i) & (
                        data["PATNO"] == row["PATNO"]), "NP3BRADY_NOW"].item()
                    row["TIME_PASSED"] = i
                    if not math.isnan(new_data.index.max()):
                        new_data.loc[new_data.index.max() + 1] = row[features]
                    else:
                        new_data.loc[0] = row[features]
        return row

    # Generate NP3BRADY_NEXT
    data["NP3BRADY_NEXT"] = -1
    data["TIME_PASSED"] = -1
    data.apply(set, axis=1)

    # Return new data
    return new_data


if __name__ == "__main__":
    main()
