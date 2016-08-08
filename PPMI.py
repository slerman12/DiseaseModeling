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


def run(target, score_name, feature_elimination_n, gen_filename, gen_action, gen_updrs_subsets, gen_time, gen_future,
        gen_milestones, gen_milestone_features_values, gen_slopes, predictors_filename, predictors_action,
        feature_importance_n, grid_search_action, grid_search_results, print_results, results_filename,
        drop_predictors):
    # Set seed
    np.random.seed(0)

    # Data keys
    data_keys = ["PATNO", "EVENT_ID", "INFODT", "PDDXDT", "SXDT", "BIRTHDT.x", "DIAGNOSIS", target]
    if gen_future:
        data_keys.extend(score_name)
    if gen_milestones:
        data_keys.extend([x[0] for x in gen_milestone_features_values])

    # Create the data frames from files
    with np.warnings.catch_warnings():
        np.warnings.simplefilter("ignore")
        all_patients = pd.read_csv("data/all_pats.csv")
        all_visits = pd.read_csv("data/all_visits.csv")
        all_updrs = pd.read_csv("data/all_updrs.csv")
        data_dictionary = pd.read_csv("data/visits_dd_v1.csv")

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
    pd_control_data = pd_control_data.merge(all_patients, on="PATNO", how="left", suffixes=["_x", ""])

    # Only include "off" data
    pd_control_data = pd_control_data[pd_control_data["PAG_UPDRS3"] == "NUPDRS3"]

    # Merge SC data onto BL data
    sc_bl_merge = pd_control_data[pd_control_data["EVENT_ID"] == "BL"].merge(
        pd_control_data[pd_control_data["EVENT_ID"] == "SC"], on="PATNO", how="left", suffixes=["", "_SC_ID"])

    # Remove SC data that already belongs to BL
    pd_control_data.loc[pd_control_data["EVENT_ID"] == "BL"] = sc_bl_merge.drop(
        [col for col in sc_bl_merge.columns if col[-6:] == "_SC_ID"], axis=1).values

    # Remove SC rows
    pd_control_data = pd_control_data[pd_control_data["EVENT_ID"] != "SC"]

    # Drop duplicates based on PATNO and EVENT_ID, keep only first
    pd_control_data = pd_control_data.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")

    # Encode to numeric
    mL.clean_data(data=pd_control_data, encode_auto=["DIAGNOSIS", "HANDED", "PAG_UPDRS3"], encode_man={
        "EVENT_ID": {"BL": 0, "V01": 1, "V02": 2, "V03": 3, "V04": 4, "V05": 5, "V06": 6, "V07": 7, "V08": 8,
                     "V09": 9, "V10": 10, "V11": 11, "V12": 12}})

    # Convert categorical data to binary columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    dummy_features = [item for item in pd_control_data.columns.values if item not in list(
        pd_control_data.select_dtypes(include=numerics).columns.values) + drop_predictors]
    pd_control_data = pd.get_dummies(pd_control_data, columns=dummy_features)

    # Controls have missing PDDXDT and SXDT
    pd_control_data.loc[pd_control_data["DIAGNOSIS"] == 0, "PDDXDT"] = pd.to_datetime("1/1/1800")
    pd_control_data.loc[pd_control_data["DIAGNOSIS"] == 0, "SXDT"] = pd.to_datetime("1/1/1800")

    if predictors_action:
        if print_results:
            print("Optimizing Predictors . . .")

        # Drop unused columns
        for column in pd_control_data.keys():
            if (column in drop_predictors) and (column not in data_keys):
                pd_control_data = pd_control_data.drop(column, 1)
    else:
        # Drop unused columns
        pd_control_data = pd_control_data[list(
            set(pd.read_csv(predictors_filename)["predictors"].values.tolist() + data_keys) & set(
                pd_control_data.columns.values.tolist()))]

        if print_results:
            # Print number patients and features before feature elimination
            print("BEFORE FEATURE ELIMINATION: Patients: {}, Features: {}".format(
                len(pd_control_data[pd_control_data["EVENT_ID"] == 0]),
                len(pd_control_data.keys())))

    # Perform optimal feature elimination
    if feature_elimination_n is None:
        feature_elimination_n = max([x / 1000 for x in range(25, 1000, 25)],
                                    key=lambda n: feature_row_selection(pd_control_data, n, True, True))
        if print_results:
            print("\rFeature Elimination N: {}\n".format(feature_elimination_n))

    # Feature/row elimination
    pd_control_data = feature_row_selection(pd_control_data, feature_elimination_n)

    if (not predictors_action) and print_results:
        # Print number patients and features after feature elimination
        print("AFTER FEATURE ELIMINATION: Patients: {}, Features: {}".format(
            len(pd_control_data[pd_control_data["EVENT_ID"] == 0]),
            len(pd_control_data.keys())))

    # Select all features in the data set
    all_data_features = list(pd_control_data.columns.values)

    # Generate features (and update all features list)
    train = generate_features(data=pd_control_data, features=all_data_features, filename=gen_filename,
                              action=gen_action, updrs_subsets=gen_updrs_subsets,
                              time=gen_time, future=gen_future, milestones=gen_milestones, slopes=gen_slopes,
                              score_name=score_name, milestone_features_values=gen_milestone_features_values,
                              progress=(not predictors_action) and print_results)

    if (not predictors_action) and print_results:
        # Data diagnostics after feature generation
        mL.describe_data(data=train, describe=True, description="AFTER FEATURE GENERATION:")

    # Parameters for grid search
    grid_search_params = [{"n_estimators": [50, 150, 300, 500, 750, 1000],
                           "min_samples_split": [4, 8, 25, 50, 75, 100],
                           "min_samples_leaf": [2, 8, 15, 25, 50, 75, 100]}]

    # Algs for model
    # Grid search (futures): n_estimators=50, min_samples_split=75, min_samples_leaf=50
    # Futures: n_estimators=150, min_samples_split=100, min_samples_leaf=25
    # Grid search (slopes): 'min_samples_split': 75, 'n_estimators': 50, 'min_samples_leaf': 25
    # Futures: 'min_samples_leaf': 100, 'min_samples_split': 25, 'n_estimators': 50
    # Newest Futures: {'n_estimators': 500, 'min_samples_leaf': 2, 'min_samples_split': 4}
    # TRMR: {'n_estimators': 150, 'min_samples_leaf': 2, 'min_samples_split': 8}
    algs = [
        RandomForestRegressor(n_estimators=150, min_samples_split=8, min_samples_leaf=2, oob_score=True),
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

    if predictors_action:
        # Initialize predictors as all numeric features
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        predictors = list(train.select_dtypes(include=numerics).columns.values)

        # Drop unwanted features from predictors list
        for feature in drop_predictors:
            if feature in predictors:
                predictors.remove(feature)

        # If grid search action, use grid search estimator
        if grid_search_action:
            algs[0] = mL.metrics(data=train, predictors=predictors, target=target, algs=algs, alg_names=alg_names,
                                 scoring="r2", grid_search_params=grid_search_params,
                                 output=True)["Grid Search Random Forest"].best_estimator_

        # Get feature importances
        feature_importances = mL.metrics(data=train, predictors=predictors, target=target, algs=algs,
                                         alg_names=alg_names, feature_importances=[True], output=True,
                                         description=None)["Feature Importances Random Forest"]

        # Set important features as predictors
        predictors = [x for x, y in feature_importances if y >= feature_importance_n]

        # Output predictors to file
        pd.DataFrame({"predictors": predictors}).to_csv(predictors_filename, index=False)

        # Run with new predictors
        run(target, score_name, feature_elimination_n, gen_filename, gen_action, gen_updrs_subsets, gen_time,
            gen_future, gen_milestones, gen_milestone_features_values, gen_slopes, predictors_filename,
            False, feature_importance_n, grid_search_action, grid_search_results, print_results, results_filename,
            drop_predictors)
    else:
        # Get predictors from file
        predictors = pd.read_csv(predictors_filename)["predictors"].values.tolist()

        # Create file of training data
        train[predictors].to_csv("data/PPMI_train.csv")

        # Grid search
        if grid_search_action or grid_search_results:
            # Compute grid search
            grid_search = mL.metrics(data=train, predictors=predictors, target=target, algs=algs, alg_names=alg_names,
                                     scoring="r2", grid_search_params=grid_search_params,
                                     output=True)

            # If grid search action, use grid search estimator
            if grid_search_action:
                algs[0] = grid_search["Grid Search Random Forest"].best_estimator_

        # Univariate feature selection
        # mL.describe_data(data=train, univariate_feature_selection=[predictors, target])

        # Display metrics, including r2 score
        metrics = mL.metrics(data=train, predictors=predictors, target=target, algs=algs, alg_names=alg_names,
                             feature_importances=[True], base_score=[True], oob_score=[True], cross_val=[True],
                             scoring="r2", output=not print_results, )
        # feature_dictionary=[data_dictionary, "FEATURE", "DSCR"])

        # Display mean absolute error score
        metrics.update(mL.metrics(data=train, predictors=predictors, target=target, algs=algs, alg_names=alg_names,
                                  cross_val=[True], scoring="mean_absolute_error", description=None,
                                  output=not print_results))

        # Display root mean squared error score
        metrics.update(mL.metrics(data=train, predictors=predictors, target=target, algs=algs, alg_names=alg_names,
                                  cross_val=[True], scoring="root_mean_squared_error", description=None,
                                  output=not print_results))

        # If grid search results, print results
        if grid_search_results:
            print(grid_search["Grid Search String Random Forest"])

        if not print_results:
            # Write results to file
            results = pd.DataFrame(
                columns=["milestone", "description", "base", "oob", "r2", "mas", "rmse", "features", "importances"])
            results.loc[0, "milestone"] = gen_milestone_features_values[0][0]
            results.loc[0, "description"] = gen_milestone_features_values[0][2]
            results.loc[0, "base"] = metrics["Base Score Random Forest"]
            results.loc[0, "oob"] = metrics["OOB Score Random Forest"]
            results.loc[0, "r2"] = metrics["Cross Validation r2 Random Forest"]
            results.loc[0, "mas"] = metrics["Cross Validation mean_absolute_error Random Forest"]
            results.loc[0, "rmse"] = metrics["Cross Validation root_mean_squared_error Random Forest"]
            feature_importances = list(metrics["Feature Importances Random Forest"])
            results.loc[0, "features"] = feature_importances[0][0]
            results.loc[0, "importances"] = feature_importances[0][1]
            for feature, importance in feature_importances[1:]:
                index = results.index.max() + 1
                results.loc[index, "features"] = feature
                results.loc[index, "importances"] = importance
            results.to_csv(results_filename, mode="a", header=False, index=False)


# Automatic feature/row selection
def feature_row_selection(data, n, test=False, progress=False):
    # Make a copy of the data
    data = data.copy()

    # Eliminate features with more than n NA at BL
    for column in data.keys():
        if len(data.loc[(data["EVENT_ID"] == 0) & (data[column].isnull()), column]) / len(
                data[data["EVENT_ID"] == 0]) > n:
            data = data.drop(column, 1)

    # TODO: Imputation
    # Drop patients with NAs
    data = data[data["PATNO"].isin(
        data.loc[(data["EVENT_ID"] == 0) & (data.notnull().all(axis=1)), "PATNO"])]

    # Drop patients without BL data
    for patno in data["PATNO"].unique():
        if patno not in data.loc[data["EVENT_ID"] == 0, "PATNO"].unique():
            data = data[data["PATNO"] != patno]

    # Print progress
    if progress:
        sys.stdout.write("\rFeature Elimination Progress: {:.2%}".format(n + .025))
        sys.stdout.flush()

    if test:
        # Return number of features * patients
        return len(data[data["EVENT_ID"] == 0]) * len(data.keys())
    else:
        # Return data
        return data


def generate_features(data, features=None, filename="generated_features.csv", action=True, updrs_subsets=True,
                      time=True, future=True, milestones=False, slopes=False, score_name="TOTAL",
                      milestone_features_values=None, progress=True):
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
                                               diagnosis_date_name="PDDXDT", first_symptom_date_name="SXDT",
                                               progress=progress)
            time_name = "TIME_FROM_BL"

        # Generate new data set for predicting future visits
        if future:
            generated_features = generate_future(data=generated_features, features=features, id_name="PATNO",
                                                 score_name=score_name, time_name=time_name, time_key_name="EVENT_ID",
                                                 progress=progress)

        def milestone_condition(milestone_data):
            condition = [milestone_data[pair[0]] > pair[1] for pair in milestone_features_values]
            return np.bitwise_or.reduce(np.array(condition))

        # Generate new data set for predicting future milestones
        if milestones:
            generated_features = generate_milestones(data=generated_features, features=features, id_name="PATNO",
                                                     time_name=time_name, condition=milestone_condition,
                                                     progress=progress)

        # Generate new data set for predicting future visits
        if slopes:
            generated_features = generate_slopes(data=generated_features, features=features, id_name="PATNO",
                                                 score_name=score_name, time_name=time_name, progress=progress)

        # Save generated features data
        generated_features.to_csv(filename, index=False)
    else:
        # Retrieve generated features data
        generated_features = pd.read_csv(filename)

    # Return generated features
    return generated_features


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


def generate_time(data, features, id_name, time_name, datetime_name, birthday_name, diagnosis_date_name,
                  first_symptom_date_name, progress):
    # Set features
    new_features = ["TIME_FROM_BL", "AGE", "TIME_SINCE_DIAGNOSIS", "TIME_SINCE_FIRST_SYMPTOM"]
    for feature in new_features:
        if feature not in features:
            features.append(feature)

    # TODO: Interpolate based on time
    # Drop rows with no date time
    data = data[data[datetime_name].notnull()]

    # Initialize columns
    data["TIME_FROM_BL"] = -1
    data["AGE"] = -1
    data["TIME_SINCE_DIAGNOSIS"] = -1
    data["TIME_SINCE_FIRST_SYMPTOM"] = -1

    # Convert dates to date times
    data[datetime_name] = pd.to_datetime(data[datetime_name])
    data[birthday_name] = pd.to_datetime(data[birthday_name])
    data[datetime_name] = pd.to_datetime(data[datetime_name])
    data[diagnosis_date_name] = pd.to_datetime(data[diagnosis_date_name])
    data[first_symptom_date_name] = pd.to_datetime(data[first_symptom_date_name])

    # Initialize progress measures
    progress_complete = 0
    progress_total = len(data[id_name].unique())

    # Set time from baseline
    for data_id in data[id_name].unique():
        # Update progress
        if progress:
            progress_complete += 1
            sys.stdout.write("\rProgress: {:.2%} [Generating Times]".format(progress_complete / progress_total))
            sys.stdout.flush()

        # Set time from BL
        now_date = data.loc[data[id_name] == data_id, datetime_name]
        baseline_date = data.loc[(data[id_name] == data_id) & (data[time_name] == 0), datetime_name].min()
        data.loc[data[id_name] == data_id, "TIME_FROM_BL"] = (now_date - baseline_date).apply(
            lambda x: int((x / np.timedelta64(1, 'D')) / 30))

    # Set age, months from diagnosis, and months from first symptom
    data["AGE"] = (data[datetime_name] - data[birthday_name]).apply(lambda x: (x / np.timedelta64(1, 'D')) / 30)
    data.loc[data["DIAGNOSIS"] == 1, "TIME_SINCE_DIAGNOSIS"] = (
        data.loc[data["DIAGNOSIS"] == 1, datetime_name] - data.loc[data["DIAGNOSIS"] == 1, diagnosis_date_name]).apply(
        lambda x: (x / np.timedelta64(1, 'D')) / 30)
    data.loc[data["DIAGNOSIS"] == 1, "TIME_SINCE_FIRST_SYMPTOM"] = (
        data.loc[data["DIAGNOSIS"] == 1, datetime_name] - data.loc[
            data["DIAGNOSIS"] == 1, first_symptom_date_name]).apply(lambda x: (x / np.timedelta64(1, 'D')) / 30)

    if progress:
        # Print new line
        print()

    # Return data
    return data


def generate_future(data, features, id_name, score_name, time_name, time_key_name, progress):
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

    # Remove rows without score
    data = data[data[score_name].notnull()]

    for group_id in data[id_name].unique():
        if progress:
            # Update progress
            progress_complete += 1
            sys.stdout.write("\rProgress: {:.2%} [Generating Futures]".format(progress_complete / progress_total))
            sys.stdout.flush()

        # Group's key, times, and scores
        key_time_score = data[data[id_name] == group_id][[id_name, time_key_name, time_name, score_name]]
        key_time_score.rename(columns={time_name: "TIME_FUTURE", score_name: "SCORE_FUTURE"}, inplace=True)

        # Add group's baseline information
        group_data = key_time_score.merge(data[(data[id_name] == group_id) & (data[time_name] == 0)], on=[id_name],
                                          how="left")
        group_data["SCORE_NOW"] = group_data[score_name]
        group_data["TIME_NOW"] = group_data[time_name]

        # Calculate time passed
        group_data["TIME_PASSED"] = group_data["TIME_FUTURE"] - group_data["TIME_NOW"]

        # Append group data to new data
        new_data = new_data.append(group_data, ignore_index=True)

    if progress:
        # Print new line
        print()

    # Return new data without future baseline
    return new_data[(new_data["TIME_FUTURE"] > 0) & (new_data["TIME_FUTURE"] < 250)]


def generate_milestones(data, features, id_name, time_name, condition, progress):
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
        if progress:
            # Update progress
            progress_complete += 1
            sys.stdout.write("\rProgress: {:.2%} [Generating Milestones]".format(progress_complete / progress_total))
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

    if progress:
        # Print new line
        print()

    # Return new data
    return new_data


def generate_slopes(data, features, id_name, time_name, score_name, progress):
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
        if progress:
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

    if progress:
        # Print new line
        print()

    # Return new data
    return new_data


if __name__ == "__main__":
    # Print results
    print_results = False

    # Milestone info
    milestone_info = pd.read_csv("data/itemizedDistributionOfUPDRSMeaning_Use.csv")

    # List of UPDRS items
    updrs_items = milestone_info["colname"].tolist()

    # Results filename
    results_filename = "data/PPMI_Milestones_Over_0.csv"

    # If not print results, output results to file
    if not print_results:
        pd.DataFrame(
            columns=["milestone", "description", "base", "oob", "r2", "mas", "rmse", "features", "importances"]).to_csv(
            results_filename, index=False)

    # Predict symptomatic milestones over 0
    for milestone in milestone_info.loc[milestone_info["use0"] == 1, "colname"].tolist():
        # Description of milestone
        milestone_description = milestone_info.loc[milestone_info["colname"] == milestone, "mean0"].item()

        # Configure and run model
        run(target="TIME_UNTIL_MILESTONE",
            score_name="TOTAL",
            feature_elimination_n=.025,
            gen_filename="data/PPMI_all_features.csv",
            gen_action=True,
            gen_updrs_subsets=True,
            gen_time=True,
            gen_future=False,
            gen_milestones=True,
            gen_milestone_features_values=[(milestone, 0, milestone_description)],
            gen_slopes=False,
            predictors_filename="data/predictors.csv",
            predictors_action=True,
            feature_importance_n=.001,
            grid_search_action=True,
            grid_search_results=False,
            print_results=print_results,
            results_filename=results_filename,
            drop_predictors=["PATNO", "EVENT_ID", "INFODT", "INFODT.x", "ORIG_ENTRY", "LAST_UPDATE", "PRIMDIAG",
                             "COMPLT",
                             "INITMDDT", "INITMDVS", "RECRUITMENT_CAT", "IMAGING_CAT", "ENROLL_DATE", "ENROLL_CAT",
                             "ENROLL_STATUS", "BIRTHDT.x", "GENDER.x", "APPRDX", "GENDER", "CNO", "PAG_UPDRS3",
                             "TIME_NOW",
                             "SCORE_FUTURE", "SCORE_SLOPE", "TIME_OF_MILESTONE", "TIME_FUTURE", "TIME_UNTIL_MILESTONE",
                             "BIRTHDT.y", "TIME_FROM_BL", "WDDT", "WDRSN", "SXDT", "PDDXDT", "SXDT_x",
                             "PDDXDT_x", "TIME_SINCE_DIAGNOSIS"])

    print("Done [Over 0]")

    # Results filename
    results_filename = "data/PPMI_Milestones_Over_1.csv"

    # If not print results, output results to file
    if not print_results:
        pd.DataFrame(
            columns=["milestone", "description", "base", "oob", "r2", "mas", "rmse", "features", "importances"]).to_csv(
            results_filename, index=False)

    # Predict symptomatic milestones over 0
    for milestone in milestone_info.loc[milestone_info["use1"] == 1, "colname"].tolist():
        # Description of milestone
        milestone_description = milestone_info.loc[milestone_info["colname"] == milestone, "mean1"].item()

        # Configure and run model
        run(target="TIME_UNTIL_MILESTONE",
            score_name="TOTAL",
            feature_elimination_n=.025,
            gen_filename="data/PPMI_all_features.csv",
            gen_action=True,
            gen_updrs_subsets=True,
            gen_time=True,
            gen_future=False,
            gen_milestones=True,
            gen_milestone_features_values=[(milestone, 1, milestone_description)],
            gen_slopes=False,
            predictors_filename="data/predictors.csv",
            predictors_action=True,
            feature_importance_n=.001,
            grid_search_action=True,
            grid_search_results=False,
            print_results=print_results,
            results_filename=results_filename,
            drop_predictors=["PATNO", "EVENT_ID", "INFODT", "INFODT.x", "ORIG_ENTRY", "LAST_UPDATE", "PRIMDIAG",
                             "COMPLT",
                             "INITMDDT", "INITMDVS", "RECRUITMENT_CAT", "IMAGING_CAT", "ENROLL_DATE", "ENROLL_CAT",
                             "ENROLL_STATUS", "BIRTHDT.x", "GENDER.x", "APPRDX", "GENDER", "CNO", "PAG_UPDRS3",
                             "TIME_NOW",
                             "SCORE_FUTURE", "SCORE_SLOPE", "TIME_OF_MILESTONE", "TIME_FUTURE", "TIME_UNTIL_MILESTONE",
                             "BIRTHDT.y", "TIME_FROM_BL", "WDDT", "WDRSN", "SXDT", "PDDXDT", "SXDT_x",
                             "PDDXDT_x", "TIME_SINCE_DIAGNOSIS"])

    print("Done [Over 1]")
