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


def run(target, score_name, gen_filename, gen_action, gen_updrs_subsets, gen_time, gen_future, gen_milestones,
        gen_milestone_features_values, gen_slopes, grid_search, drop_predictors):
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

    # Merge with patient info
    pd_control_data = pd_control_data.merge(all_patients, on="PATNO", how="left")

    # Only include "off" data
    pd_control_data = pd_control_data[pd_control_data["PAG_UPDRS3"] == "NUPDRS3"]

    # # Merge data from rescreens
    # for patient in pd_control_data["PATNO"]:
    #     # Second to last SC infodt
    #     second_to_last = pd_control_data.loc[(pd_control_data["PATNO"] == patient) & (pd_control_data["EVENT_ID"] == "SC"), ]
    #
    #     # Merge second to last SC onto last SC
    #     sc_sc_merge = pd_control_data[pd_control_data["EVENT_ID"] == "SC"].merge(
    #             pd_control_data[pd_control_data["EVENT_ID"] == "SC"], on="PATNO", how="left", suffixes=["", "_SC_ID"])
    #
    #     # Remove SC data that already belongs to BL
    #     pd_control_data.loc[pd_control_data["EVENT_ID"] == "BL"] = sc_bl_merge.drop(
    #             [col for col in sc_bl_merge.columns if col[-6:] == "_SC_ID"], axis=1).values
    #
    #     # Remove SC rows
    #     pd_control_data = pd_control_data[pd_control_data["EVENT_ID"] != "SC"]

    # TODO: Remove after merging rescreens
    # Drop duplicates based on PATNO and EVENT_ID, keep only last
    pd_control_data = pd_control_data.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="last")

    # Merge SC data onto BL data
    sc_bl_merge = pd_control_data[pd_control_data["EVENT_ID"] == "BL"].merge(
        pd_control_data[pd_control_data["EVENT_ID"] == "SC"], on="PATNO", how="left", suffixes=["", "_SC_ID"])

    # Remove SC data that already belongs to BL
    pd_control_data.loc[pd_control_data["EVENT_ID"] == "BL"] = sc_bl_merge.drop(
        [col for col in sc_bl_merge.columns if col[-6:] == "_SC_ID"], axis=1).values

    # Remove SC rows
    pd_control_data = pd_control_data[pd_control_data["EVENT_ID"] != "SC"]

    # # Drop duplicates based on PATNO and EVENT_ID, keep only first
    # pd_control_data = pd_control_data.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")

    # Encode to numeric
    mL.clean_data(data=pd_control_data, encode_auto=["GENDER.x", "DIAGNOSIS", "HANDED", "PAG_UPDRS3"], encode_man={
        "EVENT_ID": {"BL": 0, "V01": 1, "V02": 2, "V03": 3, "V04": 4, "V05": 5, "V06": 6, "V07": 7, "V08": 8,
                     "V09": 9, "V10": 10, "V11": 11, "V12": 12}})

    # Eliminate features with more than 30% NA at Baseline
    for feature in pd_control_data.keys():
        if len(pd_control_data.loc[
                           (pd_control_data["EVENT_ID"] == 0) & (pd_control_data[feature].isnull()), feature]) / len(
            pd_control_data[pd_control_data["EVENT_ID"] == 0]) > 0.3:
            pd_control_data = pd_control_data.drop(feature, 1)

    # TODO: Remove
    pd_control_data[pd_control_data["EVENT_ID"] == 0].to_csv("test1.csv")
    print("Length of data before dropping rows: {}".format(len(pd_control_data[pd_control_data["EVENT_ID"] == 0])))

    # TODO: Imputation
    # Drop baselines with NAs
    pd_control_data = pd_control_data[pd_control_data["PATNO"].isin(
        pd_control_data.loc[(pd_control_data["EVENT_ID"] == 0) & (pd_control_data.notnull().all(axis=1)), "PATNO"])]

    # TODO: Remove
    pd_control_data[pd_control_data["EVENT_ID"] == 0].to_csv("test2.csv")
    print("Length of data after dropping rows: {}".format(len(pd_control_data[pd_control_data["EVENT_ID"] == 0])))

    # Drop rows with NA at score feature
    pd_control_data = pd_control_data[pd_control_data[score_name].notnull()]

    # TODO: Interpolate based on EVENT_ID
    # Drop rows with no INFODT (only one row, I believe, patient 3818)
    pd_control_data = pd_control_data[pd_control_data["INFODT"].notnull()]

    # If gen_future, remove rows with no score
    if gen_future:
        pd_control_data = pd_control_data[pd_control_data[score_name].notnull()]

    # Drop patients without BL data
    for patient in pd_control_data["PATNO"].unique():
        if patient not in pd_control_data.loc[pd_control_data["EVENT_ID"] == 0, "PATNO"].unique():
            pd_control_data = pd_control_data[pd_control_data["PATNO"] != patient]

    # TODO: Remove
    pd_control_data.to_csv("test_delete.csv")

    # Select all features in the data set
    all_data_features = list(pd_control_data.columns.values)

    # Generate features (and update all features list)
    train = generate_features(data=pd_control_data, features=all_data_features, filename=gen_filename,
                              action=gen_action, updrs_subsets=gen_updrs_subsets,
                              time=gen_time, future=gen_future, milestones=gen_milestones, slopes=gen_slopes,
                              score_name=score_name, milestone_features_values=gen_milestone_features_values)

    # Data diagnostics after feature generation
    mL.describe_data(data=train, describe=True, description="AFTER FEATURE GENERATION:")

    # Initialize predictors as all numeric features
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    predictors = list(pd_control_data.select_dtypes(include=numerics).columns.values)

    # Drop unwanted features from predictors list
    for feature in drop_predictors:
        if feature in predictors:
            predictors.remove(feature)

    # TODO: Remove
    train[predictors].to_csv("data/PPMI_train.csv")

    # Univariate feature selection
    mL.describe_data(data=train, univariate_feature_selection=[predictors, target])

    # Algs for model
    # Grid search (futures): n_estimators=50, min_samples_split=75, min_samples_leaf=50
    # Futures: n_estimators=150, min_samples_split=100, min_samples_leaf=25
    # Grid search (slopes): 'min_samples_split': 75, 'n_estimators': 50, 'min_samples_leaf': 25
    algs = [
        RandomForestRegressor(n_estimators=150, min_samples_split=100, min_samples_leaf=25, oob_score=True),
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
                      in_ensemble=[True, True, True, True, False, False, True, True],
                      weights=[3, 2, 1, 3, 1, 3],
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
               feature_importances=[True], base_score=[True], oob_score=[True], cross_val=[True],
               scoring="r2", grid_search_params=grid_search_params if grid_search else None)

    # Display ensemble metrics
    mL.metrics(data=train, predictors=predictors, target=target, algs=algs, alg_names=alg_names,
               cross_val=[True], scoring="mean_absolute_error", description=None)

    # Display ensemble metrics
    mL.metrics(data=train, predictors=predictors, target=target, algs=algs, alg_names=alg_names,
               cross_val=[True], scoring="root_mean_squared_error", description=None)


def generate_features(data, features=None, filename="generated_features.csv", action=True, updrs_subsets=True,
                      time=True, future=True, milestones=False, slopes=False, score_name="TOTAL",
                      milestone_features_values=None):
    # Initialize if None
    if milestone_features_values is None:
        milestone_features_values = []
    if features is None:
        features = []

    # Initialize generated features and time name
    generated_features = []
    time_name = "EVENT_ID"

    # Generate features or use pre-generated features
    if action:
        # Generate UPDRS subset sums
        if updrs_subsets:
            generated_features = generate_updrs_subsets(data=data, features=features)

        # Generate months
        if time:
            generated_features = generate_time(data=generated_features, features=features, id_name="PATNO",
                                               time_name="EVENT_ID", datetime_name="INFODT", birthday_name="BIRTHDT.x",
                                               diagnosis_date_name="PDDXDT", first_symptom_date_name="SXDT")
            time_name = "TIME_FROM_BL"

        # Generate new data set for predicting future visits
        if future:
            generated_features = generate_future(data=generated_features, features=features, id_name="PATNO",
                                                 score_name=score_name, time_name=time_name, time_key_name="EVENT_ID")

        def milestone_condition(milestone_data):
            condition = [milestone_data[pair[0]] > pair[1] for pair in milestone_features_values]
            return np.bitwise_or.reduce(np.array(condition))

        # Generate new data set for predicting future milestones
        if milestones:
            generated_features = generate_milestones(data=generated_features, features=features,
                                                     id_name="PATNO",
                                                     time_name=time_name,
                                                     condition=milestone_condition)

        # Generate new data set for predicting future visits
        if slopes:
            generated_features = generate_slopes(data=generated_features, features=features, id_name="PATNO",
                                                 score_name=score_name, time_name=time_name)

        # Save generated features data
        generated_features.to_csv(filename, index=False)
    else:
        # Retrieve generated features data
        generated_features = pd.read_csv(filename)

    # Return generated features
    return generated_features


def generate_future(data, features, id_name, score_name, time_name, time_key_name):
    # Set features
    new_features = ["SCORE_NOW", "TIME_NOW", "TIME_FUTURE", "TIME_PASSED", "SCORE_FUTURE"]
    for feature in new_features:
        if feature not in features:
            features.append(feature)

    # Initialize new data frame
    new_data = pd.DataFrame(columns=features)

    # Initialize progress measures
    progress_complete = 0
    progress_total = len(data[id_name].unique())

    for group_id in data[id_name].unique():
        # Update progress
        progress_complete += 1
        sys.stdout.write("\rProgress: {:.2%}".format(progress_complete / progress_total))
        sys.stdout.flush()

        # Group's key, times, and scores
        key_time_score = data[data[id_name] == group_id][[id_name, time_key_name, time_name, score_name]]
        key_time_score.rename(columns={time_name: "TIME_FUTURE", score_name: "SCORE_FUTURE"}, inplace=True)

        # Add group's baseline information
        group_data = key_time_score.merge(data[(data[id_name] == group_id) & (data[time_name] == 0)], on=[id_name],
                                          how="left")
        group_data[["SCORE_NOW", "TIME_NOW"]] = group_data[[score_name, time_name]]

        # Calculate time passed
        group_data["TIME_PASSED"] = group_data["TIME_FUTURE"] - group_data["TIME_NOW"]

        # Append group data to new data
        new_data = new_data.append(group_data, ignore_index=True)

    # Print new line
    print()

    # TODO: Ask if necessary. Score goes down a bit without future baseline.
    # TODO: Try predicting specific future rather than just any future
    # Return new data without future baseline
    return new_data[new_data["TIME_FUTURE"] != 0]


def generate_milestones(data, features, id_name, time_name, condition):
    # Set features
    new_features = ["TIME_NOW", "TIME_OF_MILESTONE", "TIME_UNTIL_MILESTONE"]
    for feature in new_features:
        if feature not in features:
            features.append(feature)

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
    new_features = ["SCORE_SLOPE", "SCORE_NOW", "TIME_NOW"]
    for feature in new_features:
        if feature not in features:
            features.append(feature)

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


def generate_time(data, features, id_name, time_name, datetime_name, birthday_name, diagnosis_date_name,
                  first_symptom_date_name):
    # Set features
    new_features = ["TIME_FROM_BL", "AGE", "TIME_SINCE_DIAGNOSIS", "TIME_SINCE_FIRST_SYMPTOM"]
    for feature in new_features:
        if feature not in features:
            features.append(feature)

    # Initialize columns
    data["TIME_FROM_BL"] = -1
    data["AGE"] = -1
    data["TIME_SINCE_DIAGNOSIS"] = -1
    data["TIME_SINCE_FIRST_SYMPTOM"] = -1

    # Convert dates to date times
    data[datetime_name] = pd.to_datetime(data[datetime_name])
    data[birthday_name] = pd.to_datetime(data[birthday_name])
    # data[diagnosis_date_name] = pd.to_datetime(data[diagnosis_date_name])
    # data[first_symptom_date_name] = pd.to_datetime(data[first_symptom_date_name])

    # Set months from baseline
    for data_id in data[id_name].unique():
        now_date = data.loc[data[id_name] == data_id, datetime_name]
        baseline_date = data.loc[(data[id_name] == data_id) & (data[time_name] == 0), datetime_name].min()
        data.loc[data[id_name] == data_id, "TIME_FROM_BL"] = (now_date - baseline_date).apply(
            lambda x: int((x / np.timedelta64(1, 'D')) / 30))

    # Set age, months from diagnosis, and months from first symptom
    data["AGE"] = (data[datetime_name] - data[birthday_name]).apply(lambda x: (x / np.timedelta64(1, 'D')) / 30)
    # data.loc[data["DIAGNOSIS"] == 1, "TIME_SINCE_DIAGNOSIS"] = (
    #     data[datetime_name] - data[diagnosis_date_name]).apply(lambda x: (x / np.timedelta64(1, 'D')) / 30)
    # data.loc[data["DIAGNOSIS"] == 1, "TIME_SINCE_FIRST_SYMPTOM"] = (
    #     data[datetime_name] - data[first_symptom_date_name]).apply(lambda x: (x / np.timedelta64(1, 'D')) / 30)

    # Return data
    return data


def generate_updrs_subsets(data, features):
    # set features
    new_features = ["NP1", "NP2", "NP3"]
    for feature in new_features:
        if feature not in features:
            features.append(feature)

    # Sum UPDRS subsets
    data["NP1"] = data.filter(regex="NP1.*").sum(axis=1)
    data["NP2"] = data.filter(regex="NP2.*").sum(axis=1)
    data["NP3"] = data.filter(regex="NP3.*").sum(axis=1)

    # Return new data
    return data


if __name__ == "__main__":
    # List of UPDRS items
    updrs_items = pd.read_csv("data/itemizedDistributionOfUPDRSMeaning_Use.csv")["colname"].tolist()

    # Configure and run model
    run(target="SCORE_FUTURE",
        score_name="TOTAL",
        gen_filename="data/PPMI_all_features.csv",
        gen_action=False,
        gen_updrs_subsets=True,
        gen_time=True,
        gen_future=True,
        gen_milestones=False,
        gen_milestone_features_values=[("NP2TRMR", 0)],
        gen_slopes=False,
        grid_search=False,
        drop_predictors=["PATNO", "EVENT_ID", "INFODT", "INFODT.x", "ORIG_ENTRY", "LAST_UPDATE", "PRIMDIAG", "COMPLT",
                         "INITMDDT", "INITMDVS", "RECRUITMENT_CAT", "IMAGING_CAT", "ENROLL_DATE", "ENROLL_CAT",
                         "ENROLL_STATUS", "BIRTHDT.x", "GENDER.y", "APPRDX", "GENDER", "CNO", "PAG_UPDRS3",
                         "TIME_FUTURE", "TIME_NOW", "SCORE_FUTURE", "SCORE_SLOPE", "TIME_OF_MILESTONE",
                         "TIME_UNTIL_MILESTONE", "BIRTHDT.y", "TIME_SINCE_DIAGNOSIS", "TIME_SINCE_FIRST_SYMPTOM",
                         "TIME_FROM_BL"] + updrs_items)
