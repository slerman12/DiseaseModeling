import math
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import MachineLearning as mL


def main():
    # Set seed
    np.random.seed(0)

    # Create the data frames from files
    all_patients = pd.read_csv("data/all_pats.csv")
    all_visits = pd.read_csv("data/all_visits.csv")
    all_updrs = pd.read_csv("data/all_updrs.csv")

    # Enrolled PD / Control patients
    pd_control_patients = all_patients.loc[
        ((all_patients["DIAGNOSIS"] == "PD") | (all_patients["DIAGNOSIS"] == "Control")) & (
            all_patients["ENROLL_STATUS"] == "Enrolled"), "PATNO"].unique()

    # Data for these patients
    pd_control_data = all_visits[all_visits["PATNO"].isin(pd_control_patients)]

    # Merge with UPDRS scores
    pd_control_data = pd_control_data.merge(all_updrs[["PATNO", "EVENT_ID", "TOTAL"]], on=["PATNO", "EVENT_ID"],
                                            how="left")

    # Get rid of nulls for UPDRS
    pd_control_data = pd_control_data[pd_control_data["TOTAL"].notnull()]

    # Merge with patient info
    pd_control_data = pd_control_data.merge(all_patients, on="PATNO", how="left")

    # TODO: Figure out what do with SC
    # Only include baseline and subsequent visits
    pd_control_data = pd_control_data[
        (pd_control_data["EVENT_ID"] != "ST") & (
            pd_control_data["EVENT_ID"] != "U01") & (pd_control_data["EVENT_ID"] != "PW") & (
            pd_control_data["EVENT_ID"] != "SC")]

    # Encode to numeric
    mL.clean_data(data=pd_control_data, encode_auto=["GENDER.x", "DIAGNOSIS", "HANDED"], encode_man={
        "EVENT_ID": {"BL": 0, "V01": 1, "V02": 2, "V03": 3, "V04": 4, "V05": 5, "V06": 6, "V07": 7, "V08": 8,
                     "V09": 9, "V10": 10, "V11": 11, "V12": 12}})

    # TODO: Optimize flexibility with NAs
    # Eliminate features with more than 20% NAs
    for feature in pd_control_data.keys():
        if len(pd_control_data.loc[pd_control_data[feature].isnull(), feature]) / len(
                pd_control_data[feature]) > 0.2:
            pd_control_data = pd_control_data.drop(feature, 1)

    # TODO: Rethink this
    # Eliminate features with more than 30% NA at Baseline
    for feature in pd_control_data.keys():
        if len(pd_control_data.loc[
                           (pd_control_data["EVENT_ID"] == 0) & (pd_control_data[feature].isnull()), feature]) / len(
            pd_control_data[pd_control_data["EVENT_ID"] == 0]) > 0.3:
            pd_control_data = pd_control_data.drop(feature, 1)

    # TODO: Imputation
    # Drop rows with NAs
    pd_control_data = pd_control_data.dropna()

    # Drop duplicates (keep first, delete others)
    pd_control_data = pd_control_data.drop_duplicates(subset=["PATNO", "EVENT_ID"])

    # Drop patients without BL data
    for patient in pd_control_data["PATNO"].unique():
        if patient not in pd_control_data.loc[pd_control_data["EVENT_ID"] == 0, "PATNO"]:
            pd_control_data = pd_control_data[pd_control_data["PATNO"] != patient]

    # Select all features in the data set
    all_data_features = list(pd_control_data.columns.values)

    # Generate features (and update all features list)
    train = generate_features(data=pd_control_data, features=all_data_features, file="data/PPMI_train.csv",
                              action=True)

    # Data diagnostics after feature generation
    mL.describe_data(data=train, describe=True, description="AFTER FEATURE GENERATION:")

    # Initialize predictors as all features
    predictors = list(train.columns.values)

    # Initialize which features to drop from predictors
    drop_predictors = ["PATNO", "EVENT_ID", "INFODT", "INFODT.x", "ORIG_ENTRY", "LAST_UPDATE", "PAG_UPDRS3", "PRIMDIAG",
                       "COMPLT", "INITMDDT", "INITMDVS", "RECRUITMENT_CAT", "IMAGING_CAT", "ENROLL_DATE", "ENROLL_CAT",
                       "ENROLL_STATUS", "BIRTHDT.x", "GENDER.y", "APPRDX", "GENDER", "CNO", "TIME_FUTURE", "TIME_NOW",
                       "SCORE_FUTURE", "SCORE_SLOPE", "TIME_OF_MILESTONE", "TIME_UNTIL_MILESTONE", "BIRTHDT.y",
                       "MONTHS_SINCE_DIAGNOSIS", "MONTHS_SINCE_FIRST_SYMPTOM", "TOTAL"]

    # List of UPDRS components
    updrs_components = ["NP1COG", "NP1HALL", "NP1DPRS", "NP1ANXS", "NP1APAT", "NP1DDS", "NP1SLPN", "NP1SLPD", "NP1PAIN",
                        "NP1URIN", "NP1CNST", "NP1LTHD", "NP1FATG", "NP2SPCH", "NP2SALV", "NP2SWAL", "NP2EAT",
                        "NP2DRES", "NP2HYGN", "NP2HWRT", "NP2HOBB", "NP2TURN", "NP2TRMR", "NP2RISE", "NP2WALK",
                        "NP2FREZ", "PAG_UPDRS3", "NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU", "NP3RIGLU", "PN3RIGRL",
                        "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL", "NP3PRSPR", "NP3PRSPL", "NP3TTAPR",
                        "NP3TTAPL", "NP3LGAGR", "NP3LGAGL", "NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR",
                        "NP3BRADY", "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL",
                        "NP3RTALL", "NP3RTALJ", "NP3RTCON"]

    # Drop UPDRS components
    # drop_predictors.extend(updrs_components)

    # Drop unwanted features from predictors list
    for feature in drop_predictors:
        if feature in predictors:
            predictors.remove(feature)

    # Target for the model
    target = "SCORE_FUTURE"

    # Univariate feature selection
    mL.describe_data(data=train, univariate_feature_selection=[predictors, target])

    # Algs for model
    # Grid search (futures): n_estimators=50, min_samples_split=75, min_samples_leaf=50
    # Futures: n_estimators=150, min_samples_split=100, min_samples_leaf=25
    # Grid search (slopes): 'min_samples_split': 75, 'n_estimators': 50, 'min_samples_leaf': 25
    algs = [RandomForestRegressor(n_estimators=150, min_samples_split=100, min_samples_leaf=25, oob_score=True),
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

    # TODO: Configure ensemble
    # Ensemble
    ens = mL.ensemble(algs=algs, alg_names=alg_names,
                      ensemble_name="Weighted ensemble of RF, LR, SVM, GNB, KNN, and GB",
                      in_ensemble=[True, True, True, True, False, False, True, True], weights=[3, 2, 1, 3, 1, 3],
                      voting="soft")

    # Add ensemble to algs and alg_names
    # algs.append(ens["alg"])
    # alg_names.append(ens["name"])

    # Parameters for grid search
    grid_search_params = [{"n_estimators": [50, 150, 300, 500, 750, 1000],
                           "min_samples_split": [4, 8, 25, 50, 75, 100],
                           "min_samples_leaf": [2, 8, 15, 25, 50, 75, 100]}]

    # Display ensemble metrics
    mL.metrics(data=train, predictors=predictors, target=target, algs=algs, alg_names=alg_names,
               feature_importances=[True], base_score=[True], oob_score=[True],
               cross_val=[True], scoring="r2", split_accuracy=[True],
               grid_search_params=None)


def generate_features(data, features=None, file="generated_features.csv", action=True):
    # Initialize features if None
    if features is None:
        # Empty list
        features = []

    # Generate features or use pre-generated features
    if action:
        # Generate UPDRS subset sums
        generated_features = generate_updrs_subsets(data=data, features=features)

        # Generate months
        generated_features = generate_months(data=generated_features, features=features, id_name="PATNO",
                                             time_name="EVENT_ID", datetime_name="INFODT", birthday_name="BIRTHDT.x",
                                             diagnosis_date_name="PDDXDT", first_symptom_date_name="SXDT")

        # Generate new data set for predicting future visits
        generated_features = generate_future(data=generated_features, features=features, id_name="PATNO",
                                             score_name="TOTAL", time_name="MONTHS_FROM_BL")

        # Condition(s) for generating milestone
        def milestone_debilitating_tremor(milestone_data):
            return milestone_data["NP2TRMR"] >= 2

        def milestone_rigidity(milestone_data):
            return (milestone_data["NP3RIGN"] >= 2) | (milestone_data["NP3RIGRU"] >= 2) | (
                milestone_data["NP3RIGLU"] >= 2) | (milestone_data["NP3RIGLL"] >= 2)

        # Generate new data set for predicting future milestones
        # generated_features = generate_milestones(data=generated_features, features=features, id_name="PATNO",
        #                                          time_name="MONTHS_FROM_BL", condition=milestone_debilitating_tremor)

        # Generate new data set for predicting future visits
        # generated_features = generate_slopes(data=generated_features, features=features, id_name="PATNO",
        #                                      score_name="TOTAL", time_name="MONTHS_FROM_BL")

        # Save generated features data
        generated_features.to_csv(file, index=False)
    else:
        # Retrieve generated features data
        generated_features = pd.read_csv(file)

    # Return generated features
    return generated_features


def generate_future(data, features, id_name, score_name, time_name):
    # Set features
    features.extend(["SCORE_NOW", "TIME_NOW", "TIME_FUTURE", "TIME_PASSED", "SCORE_FUTURE"])

    # Set max time
    max_time = data[time_name].max()

    # Generate SCORE_NOW and TIME_NOW
    data["SCORE_NOW"] = data[score_name]
    data["TIME_NOW"] = data[time_name]

    # Create new dataframe
    new_data = pd.DataFrame(columns=features)

    # Initialize progress measures
    progress_complete = 0
    progress_total = len(data)

    # Iterate through rows and build new data (generate SCORE_FUTURE, TIME_FUTURE, and TIME_PASSED)
    for index, row in data.iterrows():
        # Update progress
        progress_complete += 1
        sys.stdout.write("\rProgress: {:.2%}".format(progress_complete / progress_total))
        sys.stdout.flush()

        # Check time value(s)
        if row["TIME_NOW"] == 0:
            # TODO: Consider predicting a specific future time instead of any future time
            # For the range of all times after this one
            for i in range(1, max_time + 1):
                # If any future time belongs to the same patient
                if any((data["TIME_NOW"] == row["TIME_NOW"] + i) & (data[id_name] == row[id_name])):
                    # Set next score
                    row["SCORE_FUTURE"] = data.loc[(data["TIME_NOW"] == row["TIME_NOW"] + i) &
                                                   (data[id_name] == row[id_name]), "SCORE_NOW"].item()

                    # Set next time
                    row["TIME_FUTURE"] = data.loc[(data["TIME_NOW"] == row["TIME_NOW"] + i) &
                                                  (data[id_name] == row[id_name]), "TIME_NOW"].item()

                    # Set time passed
                    row["TIME_PASSED"] = i

                    # Add row to new_data
                    if not math.isnan(new_data.index.max()):
                        new_data.loc[new_data.index.max() + 1] = row[features]
                    else:
                        new_data.loc[0] = row[features]

    # Print new line
    print()

    # Return new data
    return new_data


def generate_milestones(data, features, id_name, time_name, condition):
    # Set features
    features.extend(["TIME_NOW", "TIME_OF_MILESTONE", "TIME_UNTIL_MILESTONE"])

    # Create new dataframe
    new_data = pd.DataFrame(columns=features)

    # Initialize progress measures
    progress_complete = 0
    progress_total = len(data)

    # Iterate through rows and build new data (generate TIME_NOW, TIME_OF_MILESTONE, and TIME_UNTIL_MILESTONE)
    for index, row in data.iterrows():
        # Update progress
        progress_complete += 1
        sys.stdout.write("\rProgress: {:.2%}".format(progress_complete / progress_total))
        sys.stdout.flush()

        # Set ID
        data_id = row[id_name]

        # Set time now
        time_now = row[time_name]

        # Check time value(s) and make sure the condition is met for a sample with this ID
        if time_now == 0 and any(data.loc[(data[id_name] == data_id) & (condition(data)), time_name]):
            # Time of milestone
            time_of_milestone = data.loc[(data[id_name] == data_id) & (condition(
                data)), time_name].min()

            # Time until milestone from time now
            time_until_milestone = time_of_milestone - time_now

            # Set features
            row["TIME_NOW"] = time_now
            row["TIME_OF_MILESTONE"] = time_of_milestone
            row["TIME_UNTIL_MILESTONE"] = time_until_milestone

            # Add row to new_data
            if not math.isnan(new_data.index.max()):
                new_data.loc[new_data.index.max() + 1] = row[features]
            else:
                new_data.loc[0] = row[features]

    # Print new line
    print()

    # Return new data
    return new_data


def generate_slopes(data, features, id_name, time_name, score_name):
    # Set features
    features.extend(["SCORE_SLOPE", "SCORE_NOW", "TIME_NOW"])

    # Create new dataframe
    new_data = pd.DataFrame(columns=features)

    # Initialize progress measures
    progress_complete = 0
    progress_total = len(data)

    # Iterate through rows and build new data (generate TIME_NOW, TIME_OF_MILESTONE, and TIME_UNTIL_MILESTONE)
    for index, row in data.iterrows():
        # Update progress
        progress_complete += 1
        sys.stdout.write("\rProgress: {:.2%}".format(progress_complete / progress_total))
        sys.stdout.flush()

        # Set ID
        data_id = row[id_name]

        # Set time now
        time_now = row[time_name]

        # Set score now
        score_now = row[score_name]

        # Check time value(s)
        if time_now == 0:
            # Variables for linear regression
            x = data.loc[data[id_name] == data_id, score_name]
            y = data.loc[data[id_name] == data_id, time_name]

            # Linear regression
            if any(x) and any(y):
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                # Set features
                row["SCORE_SLOPE"] = slope
                row["TIME_NOW"] = time_now
                row["SCORE_NOW"] = score_now

                # Add row to new_data
                if not math.isnan(new_data.index.max()):
                    new_data.loc[new_data.index.max() + 1] = row[features]
                else:
                    new_data.loc[0] = row[features]

    # Print new line
    print()

    # Return new data
    return new_data


def generate_updrs_subsets(data, features):
    # set features
    features.extend(["NP1", "NP2", "NP3"])

    # Sum UPDRS subsets
    data["NP1"] = data.filter(regex="NP1.*").sum(axis=1)
    data["NP2"] = data.filter(regex="NP2.*").sum(axis=1)
    data["NP3"] = data.filter(regex="NP3.*").sum(axis=1)

    # Return new data
    return data


def generate_months(data, features, id_name, time_name, datetime_name, birthday_name, diagnosis_date_name,
                    first_symptom_date_name):
    # Set features
    features.extend(["MONTHS_FROM_BL", "MONTHS_AGE", "MONTHS_SINCE_DIAGNOSIS", "MONTHS_SINCE_FIRST_SYMPTOM"])

    # Initialize columns
    data["MONTHS_FROM_BL"] = -1
    data["MONTHS_AGE"] = -1
    data["MONTHS_SINCE_DIAGNOSIS"] = -1
    data["MONTHS_SINCE_FIRST_SYMPTOM"] = -1

    # Convert dates to date times
    data[datetime_name] = pd.to_datetime(data[datetime_name]).interpolate()
    data[birthday_name] = pd.to_datetime(data[birthday_name])
    # data[diagnosis_date_name] = pd.to_datetime(data[diagnosis_date_name]).interpolate()
    # data[first_symptom_date_name] = pd.to_datetime(data[first_symptom_date_name]).interpolate()

    # Set months from baseline
    for data_id in data[id_name].unique():
        now_date = data.loc[data[id_name] == data_id, datetime_name]
        baseline_date = data.loc[(data[id_name] == data_id) & (data[time_name] == 0), datetime_name].min()
        data.loc[data[id_name] == data_id, "MONTHS_FROM_BL"] = (now_date - baseline_date).apply(
            lambda x: int((x / np.timedelta64(1, 'D')) / 30))

    # Set age, months from diagnosis, and months from first symptom
    data["MONTHS_AGE"] = (data[datetime_name] - data[birthday_name]).apply(lambda x: (x / np.timedelta64(1, 'D')) / 30)
    # data.loc[data["DIAGNOSIS"] == 1, "MONTHS_SINCE_DIAGNOSIS"] = (
    #     data[datetime_name] - data[diagnosis_date_name]).apply(lambda x: (x / np.timedelta64(1, 'D')) / 30)
    # data.loc[data["DIAGNOSIS"] == 1, "MONTHS_SINCE_FIRST_SYMPTOM"] = (
    #     data[datetime_name] - data[first_symptom_date_name]).apply(lambda x: (x / np.timedelta64(1, 'D')) / 30)

    # Return data
    return data

    # # Create new dataframe
    # new_data = pd.DataFrame(columns=features)
    #
    # # Initialize progress measures
    # progress_complete = 0
    # progress_total = len(data)
    #
    # # Iterate through rows and build new data
    # for index, row in data.iterrows():
    #     # Update progress
    #     progress_complete += 1
    #     sys.stdout.write("\rProgress: {:.2%}".format(progress_complete / progress_total))
    #     sys.stdout.flush()
    #
    #     # Set ID
    #     data_id = row[id_name]
    #
    #     # Convert dates to date times
    #     data[datetime_name] = pd.to_datetime(data[datetime_name])
    #     data[birthday_name] = pd.to_datetime(data[birthday_name])
    #     data[datetime_name] = pd.to_datetime(data[datetime_name])
    #
    #     # Set features
    #     row["MONTHS_FROM_BL"] = (
    #         row[datetime_name] - data.loc[(data[id_name] == data_id) & (data[time_name] == 0), datetime_name]).apply(
    #         lambda x: x / np.timedelta64(1, 'M'))
    #     row["MONTHS_AGE"] = (row[datetime_name] - row[birthday_name]).apply(lambda x: x / np.timedelta64(1, 'M'))
    #     if row["DIAGNOSIS"] == 1:
    #         row["MONTHS_SINCE_DIAGNOSIS"] = (row[datetime_name] - row[diagnosis_date_name]).apply(
    #             lambda x: x / np.timedelta64(1, 'M'))
    #         row["MONTHS_SINCE_FIRST_SYMPTOM"] = (row[datetime_name] - row[first_symptom_date_name]).apply(
    #             lambda x: x / np.timedelta64(1, 'M'))
    #     else:
    #         row["MONTHS_SINCE_DIAGNOSIS"] = -1
    #         row["MONTHS_SINCE_FIRST_SYMPTOM"] = -1
    #
    #     # Add row to new_data
    #     if not math.isnan(new_data.index.max()):
    #         new_data.loc[new_data.index.max() + 1] = row[features]
    #     else:
    #         new_data.loc[0] = row[features]
    #
    # # Print new line
    # print()
    #
    # # Return new data
    # return new_data


if __name__ == "__main__":
    main()
