import math
import sys
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, make_scorer
from sklearn.exceptions import UndefinedMetricWarning
import MachineLearning as mL
import warnings


# Display progress in console
class Progress:
    # Initialize progress measures
    progress_complete = 0.00
    progress_total = 0.00
    name = ""
    show = True

    def __init__(self, pc, pt, name, show):
        self.progress_complete = pc
        self.progress_total = pt
        self.name = name
        self.show = show
        if self.show:
            sys.stdout.write("\rProgress: {:.2%} [{}]".format(0, name))
            sys.stdout.flush()

    def update_progress(self):
        # Update progress
        self.progress_complete += 1.00
        if self.show:
            sys.stdout.write("\rProgress: {:.2%} [{}]".format(self.progress_complete / self.progress_total, self.name))
            sys.stdout.flush()
        if (self.progress_complete == self.progress_total) and self.show:
            print("")


# Helper method for retrieving a pre-organized data set
def retrieve_data(filename, keys):
    # Retrieve preprocessed data from file
    data = pd.read_csv(filename)

    # Convert to correct dtypes
    data[keys] = data[keys].apply(pd.to_numeric, errors="coerce")

    # Return data
    return data


# TODO: Consider which categorical features can have NAs eliminated through binary dummies
# Data specific operations (Merge into one file, generate time from baseline in months, standardize feature name/values)
def preprocess_data(base_target, cohorts=None, on_off_dose="off", treated_untreated="treated_and_untreated",
                    print_results=False, data_merged_sc_into_bl_file_path=None, data_filename="preprocessed_data.csv"):
    # Import the data frames from files
    with np.warnings.catch_warnings():
        np.warnings.simplefilter("ignore")
        all_patients = pd.read_csv("data/raw_data/all_pats.csv")
        all_visits = pd.read_csv("data/raw_data/all_visits.csv")
        all_updrs = pd.read_csv("data/raw_data/all_updrs.csv")
        updrs_part_iii = pd.read_csv("data/raw_data/MDS_UPDRS_Part_III__Post_Dose_.csv")
        blood_chemistry_hematology = pd.read_csv("data/raw_data/Blood_Chemistry___Hematology.csv")

    # Include on/off data in the UPDRS dataframe to finish building all_updrs
    all_updrs = all_updrs.merge(updrs_part_iii, how="left",
                                on=["PATNO", "EVENT_ID", "NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU",
                                    "NP3RIGLU", "PN3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR",
                                    "NP3HMOVL", "NP3PRSPR", "NP3PRSPL", "NP3TTAPR", "NP3TTAPL", "NP3LGAGR",
                                    "NP3LGAGL", "NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR",
                                    "NP3BRADY", "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU",
                                    "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ",
                                    "NP3RTCON"])[["PATNO", "EVENT_ID", "TOTAL", "ANNUAL_TIME_BTW_DOSE_NUPDRS",
                                                  "ON_OFF_DOSE", "PD_MED_USE"]]

    # Merge data
    data_merged = all_visits.merge(all_updrs, on=["PATNO", "EVENT_ID", "ON_OFF_DOSE"], how="left").merge(
        all_patients, on="PATNO", how="left", suffixes=["_x", ""])

    data_merged.to_csv("test.csv", index=False)

    # Remove SC rows after combining them with BL rows
    if data_merged_sc_into_bl_file_path is None:
        # Initiate progress
        prog = Progress(0, len(data_merged["PATNO"].unique()), "Merging Screening Into Baseline", print_results)

        # Use SC data where BL is null
        for subject in data_merged["PATNO"].unique():
            if not data_merged[(data_merged["PATNO"] == subject) & (data_merged["EVENT_ID"] == "SC")].empty:
                for column in data_merged.keys():
                    if (data_merged.loc[(data_merged["PATNO"] == subject) & (
                                data_merged["EVENT_ID"] == "BL"), column].isnull().values.all()) and (
                            data_merged.loc[(data_merged["PATNO"] == subject) & (
                                        data_merged["EVENT_ID"] == "SC"), column].notnull().values.any()):
                        data_merged.loc[
                            (data_merged["PATNO"] == subject) & (data_merged["EVENT_ID"] == "BL"), column] = \
                            data_merged.loc[
                                (data_merged["PATNO"] == subject) & (
                                    data_merged["EVENT_ID"] == "SC"), column].tolist()[-1]
            # Update progress
            prog.update_progress()

        # Remove SC rows
        data_merged_sc_into_bl = data_merged[data_merged["EVENT_ID"] != "SC"]

        # Initiate progress
        prog = Progress(0, len(blood_chemistry_hematology["PATNO"].unique()), "Merging 'Blood Chemistry Hematology'",
                        print_results)

        # Add blood chemistry hematology data
        for patient in blood_chemistry_hematology["PATNO"].unique():
            for key in blood_chemistry_hematology.loc[blood_chemistry_hematology["PATNO"] == patient, "LTSTNAME"]:
                data_merged_sc_into_bl.loc[
                    (data_merged_sc_into_bl["PATNO"] == patient) & (data_merged_sc_into_bl["EVENT_ID"] == "BL"), key] = \
                    blood_chemistry_hematology.loc[(blood_chemistry_hematology["PATNO"] == patient) & (
                        blood_chemistry_hematology["LTSTNAME"] == key), "LSIRES"].values[0]

            # Update progress
            prog.update_progress()

        # Create csv of all of these datasets merged after SC rows have been combined with BL and removed
        data_merged_sc_into_bl.to_csv("data/raw_data/data_merged_SC_into_BL.csv", index=False)
    else:
        # Import pre-existing merged data with no SCs
        data_merged_sc_into_bl = pd.read_csv(data_merged_sc_into_bl_file_path)

    # List of patients only enrolled in selected cohorts
    patients_from_selected_cohorts = all_patients.loc[
        (np.bitwise_or.reduce(np.array([(all_patients["APPRDX"] == cohort) for cohort in cohorts]))) & (
            all_patients["ENROLL_STATUS"] == "Enrolled"), "PATNO"].unique()

    # Data for these patients
    data = data_merged_sc_into_bl[data_merged_sc_into_bl["PATNO"].isin(patients_from_selected_cohorts)]

    # Add binary feature for whether patient is receiving symptomatic therapy
    data.loc[:, "IS_TREATED"] = 0
    for patient in data["PATNO"].unique():
        if not data[(data["PATNO"] == patient) & (data["EVENT_ID"] == "ST")].empty:
            st_date = pd.to_datetime(
                data.loc[(data["PATNO"] == patient) & (data["EVENT_ID"] == "ST"), "INFODT"].values[0])
            data.loc[(data["PATNO"] == patient) & (pd.to_datetime(data["INFODT"]) >= st_date), "IS_TREATED"] = 1

    # TODO: Check differentiation between on/off. Look at PD_MED_USE (untreated: PD_MED_USE = 0 and ON_OFF_DOSE is null)
    # Treated vs untreated periods
    if treated_untreated == "treated_and_untreated" or treated_untreated == "treated":
        # On vs. Off dose
        if on_off_dose == "off":
            # Only include "off" data (Exclude NUPDRS3A measurements and <6hrs since last PD med dose intake)
            data = data[data["PAG_UPDRS3"] == "NUPDRS3"]

            # if treated_untreated == "treated_and_untreated":
            #     data = data[data["ON_OFF_DOSE"] != 2]
            # else:
            #     data = data[data["ON_OFF_DOSE"] == 1]

            # Drop on dose
            data = data[data["ON_OFF_DOSE"] != 2]
        elif on_off_dose == "on":
            # Only include "on" data (>6hrs since last PD med dose intake measurements or untreated)
            data = data[data["PAG_UPDRS3"] == "NUPDRS3A"]

            # if treated_untreated == "treated_and_untreated":
            #     data = data[(data["ON_OFF_DOSE"] == 2) | data["ON_OFF_DOSE"].isnull()]
            # else:
            #     data = data[(data["ON_OFF_DOSE"] == 2)]

            # Only use on dose for treated periods
            data = data[(data["ON_OFF_DOSE"] == 2) | data["ON_OFF_DOSE"].isnull()]

        # Only include treated patients
        if treated_untreated == "treated":
            # TODO: Adjust baselines for treated periods! ST should be the new baseline
            data = data[data["IS_TREATED"] == 1]
        elif treated_untreated == "treated_and_untreated":
            # Print number of treated vs untreated patients
            print("{} untreated {}-dose patients\n{} treated {}-dose patients\n".format(len(data.loc[data["IS_TREATED"]
                                                                                                 == 0, "PATNO"].unique()),
                                                                                        on_off_dose,
                                                                                        len(data.loc[data["IS_TREATED"]
                                                                                                 == 1, "PATNO"].unique()),
                                                                                        on_off_dose))
    elif treated_untreated == "untreated":
        # Untreated
        # data = data[data["PD_MED_USE"] == 0]
        # data = data[data["ON_OFF_DOSE"].isnull()]

        # Only use untreated period
        data = data[data["IS_TREATED"] == 0]

    # Only include patients when they're not receiving symptomatic treatment
    # data = data[data["ON_OFF_DOSE"].isnull()]

    # Add binary feature for whether patient is receiving symptomatic therapy
    # data.loc[:, "IS_TREATED"] = 1
    # data.loc[data["ON_OFF_DOSE"].isnull(), "IS_TREATED"] = 0

    # Make ON_OFF_DOSE binary dummies - why?
    # data = pd.get_dummies(data, columns=["ON_OFF_DOSE"])

    # TODO: Why are there duplicates? Are there actually?
    # Drop duplicates based on PATNO and EVENT_ID, keep only first
    data = data.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")

    # TODO: Figure out which other categorical data can be numerically encoded
    # Encode to numeric
    mL.clean_data(data=data, encode_auto=["HANDED", "PAG_UPDRS3"], encode_man={
        "EVENT_ID": {"BL": 0, "V01": 1, "V02": 2, "V03": 3, "V04": 4, "V05": 5, "V06": 6, "V07": 7, "V08": 8,
                     "V09": 9, "V10": 10, "V11": 11, "V12": 12, "V13": 13, "ST": -1}})

    # Create HAS_PD column
    data.loc[:, "HAS_PD"] = 0
    data.loc[(data["APPRDX"] == "PD") | (data["APPRDX"] == "GRPD") | (
        data["APPRDX"] == "GCPD"), "HAS_PD"] = 1

    # TODO: Check if other cohorts like swedds have those dates
    # Controls have missing PDDXDT and SXDT, set to arbitrary date
    data.loc[data["HAS_PD"] == 0, "PDDXDT"] = pd.to_datetime("1/1/1800")
    data.loc[data["HAS_PD"] == 0, "SXDT"] = pd.to_datetime("1/1/1800")

    # List of features
    features = list(data.columns.values)

    # Generate features (UPDRS subsets)
    data = generate_updrs_subsets(data=data, features=features)

    # Set feature keys
    feature_keys = ["PATNO", "EVENT_ID", "INFODT", "PDDXDT", "SXDT", "BIRTHDT.x", "HAS_PD", base_target]

    # Drop patients with baseline NA at feature keys
    data = data[
        data["PATNO"].isin(data.loc[(data["EVENT_ID"] == 0) & (data[feature_keys].notnull().all(axis=1)), "PATNO"])]

    # Print size of each cohort
    for cohort in cohorts:
        print("Patients in {} cohort: {}\n".format(cohort, len(data.loc[data["APPRDX"] == cohort, "PATNO"].unique())))

    # Generate features (times)
    data = generate_time(data=data, features=features, id_name="PATNO",
                         time_name="EVENT_ID", datetime_name="INFODT", birthday_name="BIRTHDT.x",
                         diagnosis_date_name="PDDXDT", first_symptom_date_name="SXDT",
                         progress=print_results)

    # Create csv
    data.to_csv(data_filename, index=False)

    # Return pd control data
    return data.copy()


# Drop patients w/o BL, drop rows w/ NA at key features, generate outcome measure
def process_data(data, model_type, patient_key, time_key, base_target, outcome_measure, drop_predictors=None,
                 print_results=False, output_file=False, data_filename="processed_data.csv",
                 symptom_features_values=None, cutoff=None, post_lme_data=None):
    # Model type booleans
    future_score = model_type == "future_severity"
    rate_of_progression = model_type == "rate_of_progression"
    time_until_symptom_onset = model_type == "time_until_symptom_onset"

    # Target features
    data_targets = [base_target] if future_score or rate_of_progression else [
        x[0] for x in symptom_features_values] if time_until_symptom_onset else []

    # Drop any observations before baseline
    data = data[data[time_key] >= 0]

    # Drop patients without BL data
    for patno in data[patient_key].unique():
        if patno not in data.loc[data[time_key] == 0, patient_key].unique():
            data = data[data[patient_key] != patno]

    # Drop rows with NA at target keys
    for key in data_targets:
        data = data[data[key].notnull()]

    # List of features
    features = list(data.columns.values)

    # Helper function to check if a symptom has presented itself
    def symptom_onset_criteria(symptom_data):
        condition = [symptom_data[pair[0]] > pair[1] for pair in symptom_features_values]
        return np.bitwise_or.reduce(np.array(condition))

    # Feature generation
    if future_score:
        # Generate new data set for predicting future visits
        data = generate_future_score(data=data, features=features, id_name=patient_key, score_name=base_target,
                                     time_name=time_key, progress=print_results)
    elif time_until_symptom_onset:
        # Generate new data set for predicting future milestones
        data = generate_time_until_symptom_onset(data=data, features=features, id_name=patient_key, time_name=time_key,
                                                 condition=symptom_onset_criteria, progress=print_results)
    elif rate_of_progression:
        # Generate new data set for predicting rate of progression
        data = generate_rate_of_progression(data=data, id_name=patient_key, score_name=base_target, time_name=time_key,
                                            target=outcome_measure, progress=print_results, cutoff=cutoff,
                                            post_lme_data=post_lme_data)

    # Drop unused columns
    for column in data.keys():
        if column in drop_predictors:
            if column != patient_key and column != time_key and column != outcome_measure:
                data = data.drop(column, 1)

    # Save generated features data
    if output_file:
        data.to_csv(data_filename, index=False)

    # Return data
    return data.copy()


# TODO: Truly maximize by searching space of all feature/patient NA-less combinations
# TODO: Make binary dummies for categorical data!
# Automatic feature and row elimination (automatically get rid of NAs and maximize data)
def eliminate_nulls_maximally(data, patient_key, time_key, outcome_measure, drop_predictors=None, add_predictors=None,
                              na_elimination_n=None, print_results=False, dummy_features=None, final_features=None,
                              balance_classes=False, data_filename="disease_modeling_data.csv"):
    # Initiate empty list(s) when no drop/add predictors
    if drop_predictors is None:
        drop_predictors = []
    if add_predictors is None:
        add_predictors = []

    # Account for any dummy features generated from data set
    if dummy_features is not None:
        # Convert categorical data to binary dummy columns (one hot encoding)
        data = pd.get_dummies(data, columns=dummy_features)

    # Use manually selected features
    if final_features is not None:
        data = data[final_features]

    # Drop unused columns
    for column in data.keys():
        if column in drop_predictors and column not in add_predictors:
            if column != patient_key and column != time_key and column != outcome_measure:
                data = data.drop(column, 1)

    # Drop variables (columns) with more than N% of patients having NA at baseline and then drop patients with NAs at BL
    def feature_row_elimination(n, test=False):
        # Make a copy of the data
        d = data.copy()

        # Get original dimensions
        if balance_classes:
            original_observation_count, original_variable_count = min(len(d[d[outcome_measure] == 0].index),
                                                                      len(d[d[outcome_measure] == 1].index)), \
                                                                  len(d.keys())
        else:
            original_observation_count, original_variable_count = len(d.index), len(d.keys())

        # Eliminate features with more than n (%) NA at BL
        for col in d.keys():
            if col not in add_predictors + [outcome_measure]:
                # Processed data produces baseline-only data so checking for baseline might be redundant
                if time_key is not None:
                    if d.loc[d[time_key] == 0, col].isnull().values.sum().astype(float) / len(
                            d[d[time_key] == 0].index) > n:
                        d = d.drop(col, 1)
                else:
                    if d[col].isnull().values.sum().astype(float) / len(d.index) > n:
                        d = d.drop(col, 1)

        # Drop observations with NAs at BL
        if time_key is not None:
            # d = d[d[patient_key].isin(d.loc[(d[time_key] == 0) & (d.notnull().all(axis=1)), patient_key])]
            # TODO: this should only drop baseline observation
            d = d.dropna(axis=0, how='any')
        else:
            # d = d[d[patient_key].isin(d.loc[d.notnull().all(axis=1), patient_key])]
            d = d.dropna(axis=0, how='any')

        # Display progress
        prog.update_progress()

        # Return dimensions/d
        if test:
            # TODO: Weigh w.r.t. original proportions
            # Return # of features * observations. If classes balanced, then features * (observations in smaller class)
            # Weighed proportionally to original dimensions
            if balance_classes:
                return len(d.keys()) * (min(len(d[d[outcome_measure] == 0].index),
                                            len(d[d[outcome_measure] == 1].index)) *
                                        (original_variable_count / original_observation_count))
            else:
                return len(d.keys()) * (len(d.index) * (original_variable_count / original_observation_count))
        else:
            # Return d
            return d

    # Print "before" dimensions
    if print_results:
        # Print number patients and features before feature elimination
        print("BEFORE NA ELIMINATION: Observations: {}, Patients: {}, Features: {}".format(
            len(data.index),
            len(data[patient_key].unique()),
            len(data.keys())))

    # Initiate progress
    prog = Progress(0, 41, "NA Elimination", print_results and na_elimination_n is None)

    # Find optimal feature elimination n
    if na_elimination_n is None:
        na_elimination_n = max([x / 1000 for x in range(0, 1000, 25)],
                               key=lambda n: feature_row_elimination(n, True))

        # Print optimal feature elimination n
        if print_results:
            print("\rNA Elimination N: {:.0%}".format(na_elimination_n))

    # Perform automatic NA elimination
    data = feature_row_elimination(na_elimination_n)

    # Print "after" dimensions
    if print_results:
        # Print number patients and features after feature elimination
        print("AFTER NA ELIMINATION: Observations: {}, Patients: {}, Features: {}".format(
            len(data.index),
            len(data[patient_key].unique()),
            len(data.keys())))

    if balance_classes:
        # Print cutoff proportions
        print("INCLUSION/EXCLUSION INTERVAL SIZE: {}:{} [# of Patients]".format(
            data.loc[data[outcome_measure] == 0, outcome_measure].size,
            data.loc[data[outcome_measure] == 1, outcome_measure].size))

        # Use smaller class size value
        if data.loc[data[outcome_measure] == 0, outcome_measure].size < data.loc[
                    data[outcome_measure] == 1, outcome_measure].size:
            # Use a subset of the larger class that is the same size as the smaller class
            drop_size = data.loc[data[outcome_measure] == 1, outcome_measure].size - data.loc[
                data[outcome_measure] == 0, outcome_measure].size
            drop_indices = np.random.choice(data[data[outcome_measure] == 1].index, drop_size, replace=False)
        else:
            drop_size = data.loc[data[outcome_measure] == 0, outcome_measure].size - data.loc[
                data[outcome_measure] == 1, outcome_measure].size
            drop_indices = np.random.choice(data[data[outcome_measure] == 0].index, drop_size, replace=False)

        # Drop subset of larger class so both classes are equal in size
        data = data.drop(drop_indices)

        # Print cutoff proportions
        print("INCLUSION/EXCLUSION INTERVAL SIZE AFTER CLASSES BALANCED: {}:{} [# of Patients]".format(
            data.loc[data[outcome_measure] == 0, outcome_measure].size,
            data.loc[data[outcome_measure] == 1, outcome_measure].size))

    # Create csv
    data.to_csv(data_filename, index=False)

    # Return data
    return data.copy()


# Train and optimize a model with grid search
def model(data, model_type, outcome_measure, is_regressor=True, drop_predictors=None, add_predictors=None,
          do_grid_search=False, feature_importance_min=0.1, print_results=True, output_results=True, n_jobs=-1,
          optimize_precision=False, results_filename="results.csv"):
    # Initiate empty list(s) when no drop/add predictors
    if drop_predictors is None:
        drop_predictors = []
    if add_predictors is None:
        add_predictors = []

    # Drop unused columns
    for column in data.keys():
        if column in drop_predictors and column not in add_predictors and column != outcome_measure:
            data = data.drop(column, 1)

    # Convert categorical data to binary dummy columns (one hot encoding)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    dummy_features = [item for item in data.columns.values if item not in list(
        data.select_dtypes(include=numerics).columns.values) + drop_predictors]
    data = pd.get_dummies(data, columns=dummy_features)

    # Initialize training data
    training_data = data.copy()

    # Drop unused columns
    for column in training_data.keys():
        if column in drop_predictors and column not in add_predictors:
            training_data = training_data.drop(column, 1)

    # Print data diagnostics
    if print_results:
        # Data diagnostics after feature generation
        mL.describe_data(data=data, value_counts=None if is_regressor else [outcome_measure], describe=True,
                         description="DATA SUMMARY:")

    # Initialize output
    model_results = {}

    # List of predictors
    predictors = list(training_data.columns.values)

    # Univariate feature selection
    # mL.describe_data(data=data, univariate_feature_selection=[predictors, outcome_measure])

    # Parameters for grid search
    grid_search_params = [{"n_estimators": [50, 150, 300, 500, 750, 1000],
                           "min_samples_split": [4, 8, 25, 50, 75, 100],
                           "min_samples_leaf": [2, 8, 15, 25, 50, 75, 100]}]

    # Algorithms for model
    algs = [
        RandomForestRegressor(n_estimators=500, min_samples_split=4, min_samples_leaf=2, n_jobs=n_jobs,
                              oob_score=True) if is_regressor else RandomForestClassifier(n_estimators=500,
                                                                                          min_samples_split=25,
                                                                                          min_samples_leaf=2,
                                                                                          oob_score=True),
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

    # Ensemble
    ens = mL.ensemble(algs=algs, alg_names=alg_names,
                      ensemble_name="Weighted ensemble of RF, LR, SVM, GNB, KNN, and GB",
                      in_ensemble=[True, True, True, True, False, False, True, True],
                      weights=[3, 2, 1, 3, 1, 3],
                      voting="soft")

    # Add ensemble to algs and alg_names
    # algs.append(ens["alg"])
    # alg_names.append(ens["name"])

    # If grid search needs to be run
    if do_grid_search:
        # Precision scorer
        precision_scorer = make_scorer(precision_score, pos_label=0, average="binary")

        # Run grid search
        grid_search = mL.metrics(data=data, predictors=predictors, target=outcome_measure, algs=algs,
                                 alg_names=alg_names, n_jobs=n_jobs,
                                 scoring="r2" if is_regressor else precision_scorer if optimize_precision else "accuracy",
                                 grid_search_params=grid_search_params,
                                 print_results=True)

        # Get best estimator
        grid_search_estimator = grid_search["Grid Search Random Forest"].best_estimator_

        # Set algorithm to grid search estimator
        algs[0] = grid_search_estimator

    # # Ability to eliminate linear dependencies
    # # Feature importance dictionary
    # fid = {}
    # for x, y in feature_importances:
    #     fid[x] = y
    #
    # # Linear dependant features
    # lin_dependencies = [["UPDRS_I", "UPDRS_II", "UPDRS_III", "TOTAL", "SCORE_NOW" if base_target == "TOTAL"],
    #                     ["GENDER.y_M", "GENDER.y_FNC", "GENDER.y_FC"],
    #                     ["UPDRS_I", "NP1COG", "NP1HALL", "NP1DPRS", "NP2ANXS", "NP1APAT", "NP1DDS"]]
    #
    # # Eliminate lowest ranking linearly dependant feature
    # for dep in lin_dependencies:
    #     if set(dep) < set(top_predictors):
    #         top_predictors.remove(min(dep, key=lambda n: fid[n]))

    # Display feature importances and other metrics
    metrics = mL.metrics(data=data, predictors=predictors, target=outcome_measure, algs=algs,
                         alg_names=alg_names, feature_importances=[True], base_score=[print_results or output_results],
                         oob_score=[print_results or output_results], print_results=print_results, description=None)

    # Precision scorer
    precision_scorer = make_scorer(precision_score, pos_label=0, average="binary")

    # Display metrics, including r2 score
    metrics.update(mL.metrics(data=data, predictors=predictors, target=outcome_measure, algs=algs,
                              alg_names=alg_names, cross_val=[print_results or output_results],
                              scoring="r2", print_results=print_results, description=None))

    if optimize_precision:
        # Display precision score
        metrics.update(mL.metrics(data=data, predictors=predictors, target=outcome_measure, algs=algs,
                                  alg_names=alg_names, cross_val=[print_results or output_results],
                                  scoring=precision_scorer, description=None, print_results=print_results))

    # Display mean absolute error score
    metrics.update(mL.metrics(data=data, predictors=predictors, target=outcome_measure, algs=algs,
                              alg_names=alg_names, cross_val=[print_results or output_results],
                              scoring="neg_mean_absolute_error", description=None, print_results=print_results))

    # Display root mean squared error score
    metrics.update(mL.metrics(data=data, predictors=predictors, target=outcome_measure, algs=algs,
                              alg_names=alg_names, cross_val=[print_results or output_results],
                              scoring="root_mean_squared_error", description=None, print_results=print_results))

    # Initialize accuracy metric
    metrics["Cross Validation accuracy Random Forest"] = None
    metrics["Cross Validation make_scorer(precision_score, average=binary, pos_label=0) Random Forest"] = None

    # If classifier
    if not is_regressor:
        # Display classification accuracy
        metrics.update(mL.metrics(data=data, predictors=predictors, target=outcome_measure, algs=algs,
                                  alg_names=alg_names, cross_val=[True], scoring="accuracy", description=None,
                                  print_results=print_results))

        # Display classification report
        mL.metrics(data=data, predictors=predictors, target=outcome_measure, algs=algs, alg_names=alg_names,
                   split_classification_report=[True], description=None, print_results=print_results)

        # Display confusion matrix
        mL.metrics(data=data[predictors + [outcome_measure]], predictors=predictors, target=outcome_measure,
                   algs=algs, alg_names=alg_names, split_confusion_matrix=[True], description=None,
                   print_results=print_results)

    # Get feature importances
    feature_importances = metrics["Feature Importances Random Forest"]

    # Set important features as top predictors
    top_predictors = [x for x, y in feature_importances if y > feature_importance_min]

    # Output results file
    if output_results:
        # Create blank results csv
        pd.DataFrame(columns=["model type", "target", "base", "oob", "r2", "mae", "rmse", "accuracy", "precision",
                              "features", "importances"]).to_csv(results_filename, index=False)

        # Create results data frame
        results = pd.DataFrame(
            columns=["model type", "target", "base", "oob", "r2", "mae", "rmse", "accuracy", "precision",
                     "features", "importances"])
        results.loc[0, "model type"] = model_type
        results.loc[0, "target"] = outcome_measure
        results.loc[0, "base"] = metrics["Base Score Random Forest"]
        results.loc[0, "oob"] = metrics["OOB Score Random Forest"]
        results.loc[0, "r2"] = metrics["Cross Validation r2 Random Forest"]
        results.loc[0, "mae"] = metrics["Cross Validation neg_mean_absolute_error Random Forest"]
        results.loc[0, "rmse"] = metrics["Cross Validation root_mean_squared_error Random Forest"]
        results.loc[0, "accuracy"] = metrics["Cross Validation accuracy Random Forest"]
        results.loc[0, "precision"] = metrics[
            "Cross Validation make_scorer(precision_score, average=binary, pos_label=0) Random Forest"]
        feature_importances = list(metrics["Feature Importances Random Forest"])
        results.loc[0, "features"] = feature_importances[0][0]
        results.loc[0, "importances"] = feature_importances[0][1]
        for feature, importance in feature_importances[1:]:
            index = results.index.max() + 1
            results.loc[index, "features"] = feature
            results.loc[index, "importances"] = importance

        # Write results to file
        results.to_csv(results_filename, mode="a", header=False, index=False)

    # Set results
    model_results["Top Predictors"] = top_predictors
    model_results["Model"] = algs[0]
    model_results["Dummy Features"] = dummy_features

    # Return model_results
    return model_results


# Generate UPDRS_I, UPDRS_II, and UPDRS_III
def generate_updrs_subsets(data, features):
    # set features
    new_features = ["UPDRS_I", "UPDRS_II", "UPDRS_III", "UPDRS_II_AND_III"]
    for feature in new_features:
        if feature not in features:
            features.append(feature)

    # Sum UPDRS subsets
    data.loc[:, "UPDRS_I"] = data.filter(regex="NP1.*").sum(axis=1)
    data.loc[:, "UPDRS_II"] = data.filter(regex="NP2.*").sum(axis=1)
    data.loc[:, "UPDRS_III"] = data.filter(regex="NP3.*").sum(axis=1)
    data.loc[:, "UPDRS_II_AND_III"] = data["UPDRS_II"] + data["UPDRS_III"]

    # Return new data
    return data


# Generate time-related features
def generate_time(data, features, id_name, time_name, datetime_name, birthday_name, diagnosis_date_name,
                  first_symptom_date_name, progress):
    # Set features
    new_features = ["TIME_FROM_BL", "AGE", "TIME_SINCE_DIAGNOSIS", "TIME_SINCE_FIRST_SYMPTOM"]
    for feature in new_features:
        if feature not in features:
            features.append(feature)

    # Drop rows with no date time
    data = data[data[datetime_name].notnull()]

    # Initialize columns
    data.loc[:, "TIME_FROM_BL"] = -1
    data.loc[:, "AGE"] = -1
    data.loc[:, "TIME_SINCE_DIAGNOSIS"] = -1
    data.loc[:, "TIME_SINCE_FIRST_SYMPTOM"] = -1

    # Convert dates to date times
    data.loc[:, datetime_name] = pd.to_datetime(data[datetime_name])
    data.loc[:, birthday_name] = pd.to_datetime(data[birthday_name])
    data.loc[:, datetime_name] = pd.to_datetime(data[datetime_name])
    data.loc[:, diagnosis_date_name] = pd.to_datetime(data[diagnosis_date_name])
    data.loc[:, first_symptom_date_name] = pd.to_datetime(data[first_symptom_date_name])

    # Initialize progress measures
    prog = Progress(0, len(data[id_name].unique()), "Generating Times", progress)

    # Set time from baseline
    for data_id in data[id_name].unique():
        # Set time from BL
        now_date = data.loc[data[id_name] == data_id, datetime_name]
        baseline_date = data.loc[(data[id_name] == data_id) & (data[time_name] == 0), datetime_name].min()
        data.loc[data[id_name] == data_id, "TIME_FROM_BL"] = (now_date - baseline_date).apply(
            lambda x: (x / np.timedelta64(1, 'D')) / 365)

        # Update progress
        prog.update_progress()

    # Set age in years, years from diagnosis, and years from first symptom
    data.loc[:, "AGE"] = (data[datetime_name] - data[birthday_name]).apply(lambda x: (x / np.timedelta64(1, 'D')) / 365)
    data.loc[data["HAS_PD"] == 1, "TIME_SINCE_DIAGNOSIS"] = (
        data.loc[data["HAS_PD"] == 1, datetime_name] - data.loc[data["HAS_PD"] == 1, diagnosis_date_name]).apply(
        lambda x: (x / np.timedelta64(1, 'D')) / 365)
    data.loc[data["HAS_PD"] == 1, "TIME_SINCE_FIRST_SYMPTOM"] = (
        data.loc[data["HAS_PD"] == 1, datetime_name] - data.loc[data["HAS_PD"] == 1, first_symptom_date_name]).apply(
        lambda x: (x / np.timedelta64(1, 'D')) / 365)

    # Return data
    return data


# Generate future scores
def generate_future_score(data, features, id_name, score_name, time_name, progress):
    # Set features
    new_features = ["SCORE_NOW", "TIME_NOW", "TIME_FUTURE", "TIME_PASSED", "SCORE_FUTURE"]
    for feature in new_features:
        if feature not in features:
            features.append(feature)

    # Initialize new data frame
    new_data = pd.DataFrame(columns=features)

    # Remove rows without score
    data = data[data[score_name].notnull()]

    # Initialize progress measures
    prog = Progress(0, len(data[id_name].unique()), "Generating Futures", progress)

    for group_id in data[id_name].unique():
        # Group's key, times, and scores
        key_time_score = data[data[id_name] == group_id][[id_name, time_name, score_name]]
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

        # Update progress
        prog.update_progress()

    # TODO: customizable time range
    # Return new data with future baseline
    return new_data[(new_data["TIME_FUTURE"] >= 0) & (new_data["TIME_FUTURE"] <= 24)]


# Generate time until symptom onsets
def generate_time_until_symptom_onset(data, features, id_name, time_name, condition, progress):
    # Set features
    new_features = ["TIME_NOW", "TIME_OF_MILESTONE", "TIME_UNTIL_MILESTONE"]
    for feature in new_features:
        if feature not in features:
            features.append(feature)

    # Create new dataframe
    new_data = pd.DataFrame(columns=features)

    # Initialize progress measures
    prog = Progress(0, len(data), "Generating Milestones", progress)

    # Iterate through rows and build new data (generate TIME_NOW, TIME_OF_MILESTONE, and TIME_UNTIL_MILESTONE)
    for index, row in data.iterrows():
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

        # Update progress
        prog.update_progress()

    # Return new data
    return new_data


# Generate rates of progression
def generate_rate_of_progression(data, id_name, time_name, score_name, target, progress, min_duration=None,
                                 max_duration=None, min_observations=3, cutoff=None, post_lme_data=None):
    # Only include patients with at least two years of data
    if min_duration is not None:
        data = data[data[id_name].isin(data.loc[data[time_name] >= min_duration, id_name].unique())]

    # Only include up to a certain duration of data
    if max_duration is not None:
        data = data[data[time_name] <= max_duration]

    # Only include patients with at least a certain number of observations
    for subject in data[id_name].unique():
        if len(data[data[id_name] == subject]) < min_observations:
            data = data[data[id_name] != subject]

    # If linear mixed effects model
    if target == "RATE_LME_CONTINUOUS" \
            or target == "RATE_LME_INCLUSION_EXCLUSION_SLOW" \
            or target == "RATE_LME_INCLUSION_EXCLUSION_FAST" \
            or target == "RATE_LME_DISCRETE" \
            or target == "RATE_LME_TX_VS_NO_TX_CONTINUOUS" \
            or target == "RATE_LME_INCLUSION_EXCLUSION_MAN":
        # Output pre-LME data
        data.to_csv("data/output/pre_LME_data.csv", index=False)

        # If LME data not already provided
        if post_lme_data is None:
            # Linear mixed-effects model w/ random slopes/random intercepts
            lme = sm.MixedLM.from_formula("{} ~ {}".format(score_name, time_name), data, re_formula="~" + time_name,
                                          groups=data[id_name])

            # Fit lme model
            lme_fit = lme.fit()

            # Lme results
            lme_result = pd.DataFrame.from_dict(lme_fit.random_effects, "index").drop("Intercept", 1)
        else:
            lme_result = post_lme_data[[id_name, time_name]]

        # Rename lme rate
        lme_result.rename(columns={time_name: "RATE_LME_CONTINUOUS"}, inplace=True)

        # Get tertiles
        lme_tertile_1 = np.percentile(lme_result["RATE_LME_CONTINUOUS"], 33 + 1 / 3)
        lme_tertile_2 = np.percentile(lme_result["RATE_LME_CONTINUOUS"], 66 + 2 / 3)

        # Print classification cutoffs
        if progress:
            print("SLOW/MODERATE CUTOFF: {} [Tertile]".format(lme_tertile_1))
            print("MODERATE/FAST CUTOFF: {} [Tertile]".format(lme_tertile_2))
            if cutoff is not None:
                print("SELECTED CUTOFF: {} [Manually]".format(cutoff))

        # Label slow, moderate, and fast progression
        lme_result.loc[lme_result["RATE_LME_CONTINUOUS"] < lme_tertile_1, "RATE_LME_DISCRETE"] = 0
        lme_result.loc[
            (lme_result["RATE_LME_CONTINUOUS"] >= lme_tertile_1) & (
                lme_result["RATE_LME_CONTINUOUS"] < lme_tertile_2), "RATE_LME_DISCRETE"] = 1
        lme_result.loc[lme_result["RATE_LME_CONTINUOUS"] >= lme_tertile_2, "RATE_LME_DISCRETE"] = 2

        # Label fast and not fast progression
        lme_result.loc[lme_result["RATE_LME_CONTINUOUS"] < lme_tertile_1, "RATE_LME_INCLUSION_EXCLUSION_FAST"] = 0
        lme_result.loc[
            (lme_result["RATE_LME_CONTINUOUS"] >= lme_tertile_1) & (
                lme_result["RATE_LME_CONTINUOUS"] < lme_tertile_2), "RATE_LME_INCLUSION_EXCLUSION_FAST"] = 0
        lme_result.loc[lme_result["RATE_LME_CONTINUOUS"] >= lme_tertile_2, "RATE_LME_INCLUSION_EXCLUSION_FAST"] = 1

        # Label slow and not slow progression
        lme_result.loc[lme_result["RATE_LME_CONTINUOUS"] < lme_tertile_1, "RATE_LME_INCLUSION_EXCLUSION_SLOW"] = 0
        lme_result.loc[
            (lme_result["RATE_LME_CONTINUOUS"] >= lme_tertile_1) & (
                lme_result["RATE_LME_CONTINUOUS"] < lme_tertile_2), "RATE_LME_INCLUSION_EXCLUSION_SLOW"] = 1
        lme_result.loc[lme_result["RATE_LME_CONTINUOUS"] >= lme_tertile_2, "RATE_LME_INCLUSION_EXCLUSION_SLOW"] = 1

        if cutoff is not None:
            # Label manually selected cutoff progression
            lme_result.loc[lme_result["RATE_LME_CONTINUOUS"] < cutoff, "RATE_LME_INCLUSION_EXCLUSION_MAN"] = 0
            lme_result.loc[lme_result["RATE_LME_CONTINUOUS"] >= cutoff, "RATE_LME_INCLUSION_EXCLUSION_MAN"] = 1

            # Print cutoff proportions
            print("INCLUSION/EXCLUSION INTERVAL SIZE: {}:{} [# of Patients]".format(
                lme_result.loc[lme_result["RATE_LME_CONTINUOUS"] < cutoff, "RATE_LME_CONTINUOUS"].size,
                lme_result.loc[lme_result["RATE_LME_CONTINUOUS"] >= cutoff, "RATE_LME_CONTINUOUS"].size))

        # Merge baseline data w/ lme results
        data = data[data[time_name] == 0].merge(lme_result, how="left", left_on=[id_name], right_index=True)

        # TODO: Do class balancing after automatic NA elimination so balance not broken after patient elimination!
        # If cutoff and balance classes
        # if cutoff is not None and balance_classes:
        #     # Use smaller class size value
        #     if data.loc[data["RATE_LME_CONTINUOUS"] < cutoff, "RATE_LME_CONTINUOUS"].size < data.loc[
        #                 data["RATE_LME_CONTINUOUS"] >= cutoff, "RATE_LME_CONTINUOUS"].size:
        #         # Use a subset of the larger class that is the same size as the smaller class
        #         drop_size = data.loc[data["RATE_LME_CONTINUOUS"] >= cutoff, "RATE_LME_CONTINUOUS"].size - data.loc[
        #             data["RATE_LME_CONTINUOUS"] < cutoff, "RATE_LME_CONTINUOUS"].size
        #         drop_indices = np.random.choice(data[data["RATE_LME_CONTINUOUS"] >= cutoff].index, drop_size,
        #                                         replace=False)
        #     else:
        #         drop_size = data.loc[data["RATE_LME_CONTINUOUS"] < cutoff, "RATE_LME_CONTINUOUS"].size - data.loc[
        #             data["RATE_LME_CONTINUOUS"] >= cutoff, "RATE_LME_CONTINUOUS"].size
        #         drop_indices = np.random.choice(data[data["RATE_LME_CONTINUOUS"] < cutoff].index, drop_size,
        #                                         replace=False)
        #     data = data.drop(drop_indices)
        #
        #     # Print cutoff proportions
        #     print("INCLUSION/EXCLUSION INTERVAL SIZE AFTER CLASSES BALANCED: {}:{} [# of Patients]".format(
        #         data.loc[data["RATE_LME_CONTINUOUS"] < cutoff, "RATE_LME_CONTINUOUS"].size,
        #         data.loc[data["RATE_LME_CONTINUOUS"] >= cutoff, "RATE_LME_CONTINUOUS"].size))

        # Return data
        return data
    # If linear regression model
    elif target == "RATE_LR_DISCRETE" \
            or target == "RATE_LR_INCLUSION_EXCLUSION_SLOW" \
            or target == "RATE_LR_INCLUSION_EXCLUSION_FAST" \
            or target == "RATE_LR_CONTINUOUS":
        # Initialize progress measures
        prog = Progress(0, len(data[id_name].unique()), "Rate Linear Regression", progress)

        # Iterate through patients
        for data_id in data[id_name].unique():
            # Variables for linear regression
            x_var = data.loc[data[id_name] == data_id, time_name]
            y_var = data.loc[data[id_name] == data_id, score_name]

            # Linear regression
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_var, y_var)

            # Set features
            data.loc[(data[id_name] == data_id) & (data[time_name] == 0), "RATE_LR_CONTINUOUS"] = slope
            data.loc[(data[id_name] == data_id) & (data[time_name] == 0), "TIME_NOW"] = 0
            data.loc[(data[id_name] == data_id) & (data[time_name] == 0), "SCORE_NOW"] = data.loc[
                (data[id_name] == data_id) & (data[time_name] == 0), score_name]

            # Update progress
            prog.update_progress()

        # Only use baseline data
        data = data[data[time_name] == 0]

        # Get tertiles
        tertile_1 = np.percentile(data["RATE_LR_CONTINUOUS"], 33 + 1 / 3)
        tertile_2 = np.percentile(data["RATE_LR_CONTINUOUS"], 66 + 2 / 3)

        # Print classification cutoffs
        if progress:
            print("SLOW/MODERATE CUTOFF: {}".format(tertile_1))
            print("MODERATE/FAST CUTOFF: {}".format(tertile_2))

        # Label slow, medium, and fast progression
        data.loc[data["RATE_LR_CONTINUOUS"] < tertile_1, "RATE_LR_DISCRETE"] = 0
        data.loc[
            (data["RATE_LR_CONTINUOUS"] >= tertile_1) & (
                data["RATE_LR_CONTINUOUS"] < tertile_2), "RATE_LR_DISCRETE"] = 1
        data.loc[data["RATE_LR_CONTINUOUS"] >= tertile_2, "RATE_LR_DISCRETE"] = 2

        # Label slow, medium, and fast progression
        data.loc[data["RATE_LR_CONTINUOUS"] < tertile_1, "RATE_LR_INCLUSION_EXCLUSION_FAST"] = 0
        data.loc[
            (data["RATE_LR_CONTINUOUS"] >= tertile_1) & (
                data["RATE_LR_CONTINUOUS"] < tertile_2), "RATE_LR_INCLUSION_EXCLUSION_FAST"] = 0
        data.loc[data["RATE_LR_CONTINUOUS"] >= tertile_2, "RATE_LR_INCLUSION_EXCLUSION_FAST"] = 1

        # Label slow, medium, and fast progression
        data.loc[data["RATE_LR_CONTINUOUS"] < tertile_1, "RATE_LR_INCLUSION_EXCLUSION_SLOW"] = 0
        data.loc[
            (data["RATE_LR_CONTINUOUS"] >= tertile_1) & (
                data["RATE_LR_CONTINUOUS"] < tertile_2), "RATE_LR_INCLUSION_EXCLUSION_SLOW"] = 1
        data.loc[data["RATE_LR_CONTINUOUS"] >= tertile_2, "RATE_LR_INCLUSION_EXCLUSION_SLOW"] = 1

        if cutoff is not None:
            # Label manually selected cutoff progression
            data.loc[data["RATE_LR_CONTINUOUS"] < cutoff, "RATE_LR_INCLUSION_EXCLUSION_MAN"] = 0
            data.loc[data["RATE_LR_CONTINUOUS"] >= cutoff, "RATE_LR_INCLUSION_EXCLUSION_MAN"] = 1

        # Return new data
        return data


# Compute stats
def stats(histogram=None, bar=None, show=True):
    # Histogram
    if histogram is not None:
        # Plot info for each histogram
        for plot_info in histogram["info"]:
            # Plot histogram
            plt.hist(plot_info["data"], bins="auto" if "bins" not in plot_info else plot_info["bins"],
                     alpha=0.5 if "alpha" not in plot_info else plot_info["alpha"],
                     label=None if "label" not in plot_info else plot_info["label"])

        # Legend
        if histogram["legend"]:
            plt.legend(loc="upper right")

        # Title
        if "title" in histogram:
            plt.title(histogram["title"])

        # X label
        if "xlabel" in histogram:
            plt.xlabel(histogram["xlabel"])

        # Y label
        if "ylabel" in histogram:
            plt.ylabel(histogram["ylabel"])

        # Show plot
        if show:
            plt.show()

    if bar is not None:
        # x Positions
        xpos = np.arange(len(bar["labels"]))

        # Create bar graph
        plt.bar(xpos, bar["yvalues"], align='center', alpha=0.5)
        plt.xticks(xpos, bar["labels"])

        # Title
        if "title" in bar:
            plt.title(bar["title"])

        # X label
        if "xlabel" in bar:
            plt.xlabel(bar["xlabel"])

        # Y label
        if "ylabel" in bar:
            plt.ylabel(bar["ylabel"])

        if show:
            plt.show()


# Histograms of rate frequencies for LME using data of all patients vs of PD patients, compared with LR
def histograms_rate_lme_types_lr(patient_key, time_key, base_target, drop_predictors=None):
    # Initiate empty list when no drop predictors
    if drop_predictors is None:
        drop_predictors = []

    # Select rate of progression model
    model_type = "rate_of_progression"

    # Suffix of file output names
    filename_suffix = "pd_data"

    # Retrieve pre-organized data
    # preprocessed_data_pd = retrieve_data("data/output/preprocessed_{}.csv".format(filename_suffix),
    #                                      [patient_key, time_key, base_target])

    # Data specific operations (ex. cohorts=["PD", "GRPD", "GCPD"] )
    preprocessed_data_pd = preprocess_data(base_target, cohorts=["PD", "GRPD", "GCPD"], print_results=True,
                                           data_merged_sc_into_bl_file_path="data/raw_data/data_merged_SC_into_BL.csv",
                                           data_filename="data/output/preprocessed_{}.csv".format(filename_suffix))

    # Start outcome measure as linear mixed effects
    outcome_measure = "RATE_LME_CONTINUOUS"

    # Prepare data and generate outcome measure
    processed_data_pd_rate_lme = process_data(preprocessed_data_pd, model_type, patient_key, time_key, base_target,
                                              outcome_measure, drop_predictors, print_results=True, output_file=True,
                                              data_filename="data/output/processed_{}_{}.csv".format(
                                                  filename_suffix, outcome_measure))

    # Suffix of file output names
    filename_suffix = "pd_control_data"

    # Data specific operations (ex. cohorts=["PD", "GRPD", "GCPD"] )
    preprocessed_data_pd_control = \
        preprocess_data(base_target, cohorts=["PD", "GRPD", "GCPD", "CONTROL"], print_results=True,
                        data_merged_sc_into_bl_file_path="data/raw_data/data_merged_SC_into_BL.csv",
                        data_filename="data/output/preprocessed_{}.csv".format(
                            filename_suffix))

    # Prepare data and generate outcome measure
    processed_data_pd_control_rate_lme = process_data(preprocessed_data_pd_control, model_type, patient_key, time_key,
                                                      base_target, outcome_measure, drop_predictors, print_results=True,
                                                      output_file=True,
                                                      data_filename="data/output/processed_{}_{}.csv".format(
                                                          filename_suffix, outcome_measure))

    # Change outcome measure to linear regression
    outcome_measure = "RATE_LR_CONTINUOUS"

    # Suffix of file output names
    filename_suffix = "pd_data"

    # Prepare data
    processed_data_pd_rate_lr = process_data(preprocessed_data_pd, model_type, patient_key, time_key, base_target,
                                             outcome_measure, drop_predictors, print_results=True, output_file=True,
                                             data_filename="data/output/processed_{}_{}.csv".format(
                                                 filename_suffix, outcome_measure))

    # Stats on rates
    stats(histogram={"info": [
        {"data": processed_data_pd_rate_lme["RATE_LME_CONTINUOUS"], "label": "LME on PD patients", "alpha": .33},
        {"data": processed_data_pd_control_rate_lme.loc[
            processed_data_pd_control_rate_lme["APPRDX"] != "CONTROL", "RATE_LME_CONTINUOUS"],
         "label": "LME on PD and control patients", "alpha": .33},
        {"data": processed_data_pd_rate_lr["RATE_LR_CONTINUOUS"], "label": "LR on PD patients", "alpha": .33}],
        "title": "Frequency of Rates for PD Patients", "xlabel": "Rate (UPDRS / Time)",
        "ylabel": "Frequency", "legend": True})

    stats(histogram={"info": [
        {"data": processed_data_pd_rate_lme["RATE_LME_CONTINUOUS"], "label": "LME on PD patients", "alpha": .33}],
        "title": "Frequency of Rates for PD Patients", "xlabel": "Rate (UPDRS / Time)",
        "ylabel": "Frequency", "legend": True})

    stats(histogram={"info": [
        {"data": processed_data_pd_control_rate_lme.loc[
            processed_data_pd_control_rate_lme["APPRDX"] != "CONTROL", "RATE_LME_CONTINUOUS"],
         "label": "LME on PD and control patients", "alpha": .33}],
        "title": "Frequency of Rates for PD Patients", "xlabel": "Rate (UPDRS / Time)",
        "ylabel": "Frequency", "legend": True})

    stats(histogram={"info": [
        {"data": processed_data_pd_rate_lr["RATE_LR_CONTINUOUS"], "label": "LR on PD patients", "alpha": .33}],
        "title": "Frequency of Rates for PD Patients", "xlabel": "Rate (UPDRS / Time)",
        "ylabel": "Frequency", "legend": True})

    # Retrieve pre-organized data
    processed_data_pd_rate_lr = retrieve_data("data/output/processed_pd_data_RATE_LR_CONTINUOUS.csv",
                                              [patient_key, time_key])
    processed_data_pd_rate_lme = retrieve_data("data/output/processed_pd_data_RATE_LME_CONTINUOUS.csv",
                                               [patient_key, time_key])
    processed_data_pd_control_rate_lme = retrieve_data("data/output/processed_pd_control_data_RATE_LME_CONTINUOUS.csv",
                                                       [patient_key, time_key])

    # Get slow patients from different models
    slow_patients_lr = processed_data_pd_rate_lr.loc[processed_data_pd_rate_lr["RATE_LR_DISCRETE"] == 0, "PATNO"]
    slow_patients_lme_pd = processed_data_pd_rate_lme.loc[processed_data_pd_rate_lme["RATE_LME_DISCRETE"] == 0, "PATNO"]
    slow_patients_lme_pd_control = processed_data_pd_control_rate_lme.loc[
        (processed_data_pd_control_rate_lme["RATE_LME_DISCRETE"] == 0) & (
            processed_data_pd_control_rate_lme["APPRDX"] != "CONTROL"), "PATNO"]

    # Print descriptions of slow patient groups by model
    print("LIN REG")
    print(slow_patients_lr.describe())
    print("\nLME WITH PD")
    print(slow_patients_lme_pd.describe())
    print("\nLME WITH PD AND Control")
    print(slow_patients_lme_pd_control.describe())
    print("\nTOTAL # OF PD PATIENTS")
    print(processed_data_pd_rate_lr["PATNO"].size)

    # Stats on non overlapping PD patients between different model types
    stats(bar={
        "labels": ("LR vs. LME w/ PD", "LR vs. LME w/ PD & Controls", "LME w/ PD vs. LME w/ PD & Controls"),
        "yvalues": [pd.Series(list(set(slow_patients_lr) | set(slow_patients_lme_pd))).size - pd.Series(
            list(set(slow_patients_lr) & set(slow_patients_lme_pd))).size,
                    pd.Series(list(set(slow_patients_lr) | set(slow_patients_lme_pd_control))).size - pd.Series(
                        list(set(slow_patients_lr) & set(slow_patients_lme_pd_control))).size,
                    pd.Series(list(set(slow_patients_lme_pd) | set(slow_patients_lme_pd_control))).size - pd.Series(
                        list(set(slow_patients_lme_pd) & set(slow_patients_lme_pd_control))).size],
        "title": "Differences Between Models in Classifying PD Patients as 'Slow'", "xlabel": "Model Pair",
        "ylabel": "# of Non-Overlapping PD Patients Classified as 'Slow'"})


# Patient key: Numeric IDs uniquely representing patients (example: "PATNO")
# Time key: Numeric unit of time from baseline where time at baseline = 0 (example: "TIME_FROM_BL")
# Model type: Select "future_severity", "rate_of_progression", or "time_until_symptom_onset")
# Is regression: True for modeling regression, False for classification
# Base target: Feature from raw data to be used to create outcome measure (example: "TOTAL" or combined part II an III)
# Outcome measure: Final target (examples: "SCORE_FUTURE", "TIME_UNTIL_MILESTONE", "RATE_LME_INCLUSION_EXCLUSION_FAST")
# Add predictors: Explicit features to add as predictors, regardless of ranking of importance
# Drop predictors: Explicit features not to use as predictors, regardless of ranking of importance
# Filename suffix: Suffix of file output names
def run(patient_key, time_key, model_type, is_regressor, base_target, outcome_measure, add_predictors=None,
        drop_predictors=None, on_off_dose="off", treated_untreated="treated_and_untreated", cutoff=None,
        balance_classes=False, data_merged_sc_into_bl_file_path=None, do_grid_search=False, no_nulls_data=None,
        processed_data=None, preprocessed_data=None, cohorts=None,
        post_lme_data=None, na_elimination_n=None, optimize_precision=False):
    # Print run details
    print("\nRUN DETAILS\n")
    print("Model type: {}\n"
          "Classifier? {}\n"
          "Base target: {}\n"
          "Outcome measure: {}\n"
          "On or off dose? {}\n"
          "Treated or untreated? {}\n"
          "Cohorts: {}\n"
          "Manually selected cutoff? {}\n"
          "Classes balanced: {}\n"
          "LME with R framework? {}\n"
          "Grid search? {}\n"
          "Precision? {}".format(model_type, not is_regressor, base_target, outcome_measure, on_off_dose,
                                 treated_untreated, ', '.join(cohorts), cutoff, balance_classes,
                                 post_lme_data is not None, do_grid_search, optimize_precision))

    # Initiate empty list(s) when no drop/add predictors or cohorts
    if add_predictors is None:
        add_predictors = []
    if drop_predictors is None:
        drop_predictors = []
    if cohorts is None:
        cohorts = ["PD"]

    # Filename suffixes
    filename_suffix = "{}_{}_{}_{}_{}_{}{}".format(model_type, treated_untreated, on_off_dose, outcome_measure,
                                                   base_target, '_'.join(cohorts),
                                                   "_cutoff_{}".format(cutoff) if cutoff is not None else "")

    # If fully processed and numeric data is not provided
    if no_nulls_data is None:
        # If processed data not provided
        if processed_data is None:
            # If preprocessed data not provided
            if preprocessed_data is None:
                # Data specific operations (cohorts=["PD", "GRPD", "GCPD"] )
                preprocessed_data = preprocess_data(base_target, cohorts=cohorts, print_results=True,
                                                    data_merged_sc_into_bl_file_path=data_merged_sc_into_bl_file_path,
                                                    on_off_dose=on_off_dose, treated_untreated=treated_untreated,
                                                    data_filename="data/output/preprocessed_data_{}_{}_{}.csv".format(
                                                        treated_untreated, on_off_dose, '_'.join(cohorts)))

            # Prepare data and generate outcome measure
            processed_data = process_data(preprocessed_data, model_type, patient_key, time_key, base_target,
                                          outcome_measure, drop_predictors, print_results=True, output_file=True,
                                          data_filename="data/output/processed_data_{}.csv".format(filename_suffix),
                                          cutoff=cutoff, post_lme_data=post_lme_data)

        # Maximize data dimensions w/o NAs
        no_nulls_data = eliminate_nulls_maximally(processed_data, patient_key, time_key, outcome_measure,
                                                  drop_predictors, add_predictors,
                                                  na_elimination_n=na_elimination_n, print_results=True,
                                                  balance_classes=balance_classes,
                                                  data_filename="data/output/no_NAs_data_{}.csv".format(
                                                      filename_suffix))

    # Primary run of model
    primary_estimator = model(no_nulls_data, model_type, outcome_measure, is_regressor, drop_predictors,
                              do_grid_search=do_grid_search, feature_importance_min=0.01, print_results=False,
                              output_results=False, optimize_precision=optimize_precision)

    # Final list of features: top predictors + keys + target
    # final_features = list(
    #         set(primary_estimator["Top Predictors"]).union(add_predictors).union(
    #                 [patient_key, time_key, outcome_measure]))

    # TODO: use above code to include added predictors (this was just for "not dropping")
    final_features = list(
        set(primary_estimator["Top Predictors"]).union([patient_key, time_key, outcome_measure]))

    # Print top ranking variables
    print("\n{} TOP RANKING VARIABLES: {}\n".format(len(final_features), final_features))

    # Eliminate nulls maximally from processed data without unused features
    final_data = eliminate_nulls_maximally(processed_data, patient_key, time_key,
                                           outcome_measure, drop_predictors, na_elimination_n=1,
                                           print_results=True, dummy_features=primary_estimator["Dummy Features"],
                                           final_features=final_features,
                                           balance_classes=balance_classes,
                                           data_filename="data/output/final_data_{}.csv".format(filename_suffix))

    # Run model using top predictors
    estimator = model(final_data, model_type, outcome_measure, is_regressor, drop_predictors,
                      do_grid_search=do_grid_search, print_results=True, output_results=True,
                      optimize_precision=optimize_precision,
                      results_filename="data/output/results_{}.csv".format(filename_suffix))["Model"]

    # # Run model to predict on data
    # predictions = estimator.predict("final_data")
    #
    # # Create predictions file
    # pd.concat([pd.DataFrame(predictions, columns=['predictions']), final_data[outcome_measure]],
    #           axis=1).to_csv("data/output/predictions_{}_{}_{}_{}_{}.csv".format(treated_untreated, on_off_dose,
    #                                                                              filename_suffix, outcome_measure,
    #                                                                              base_target))

    # Return final model
    return estimator


# Main method
if __name__ == "__main__":
    # Suppress grid search undefined metric warning for unused models that have precision 0 denominator
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # Set seed
    np.random.seed(0)

    # TODO: Consider features that have nulls but should be included with binary dummies
    # Features to add as predictors regardless of importance ranking
    # add = []

    add = []

    # Features not to use as predictors
    drop = ["PATNO", "EVENT_ID", "INFODT", "INFODT.x", "INFODT_2", "DIAGNOSIS", "ORIG_ENTRY", "LAST_UPDATE", "PRIMDIAG",
            "COMPLT", "INITMDDT", "INITMDVS", "RECRUITMENT_CAT", "IMAGING_CAT", "ENROLL_DATE", "ENROLL_CAT",
            "ENROLL_STATUS", "BIRTHDT.x", "GENDER.x", "GENDER", "CNO", "PAG_UPDRS3", "TIME_NOW", "SCORE_FUTURE",
            "TIME_OF_MILESTONE", "TIME_FUTURE", "TIME_UNTIL_MILESTONE", "BIRTHDT.y", "TIME_FROM_BL", "WDDT", "WDRSN",
            "SXDT", "PDDXDT", "SXDT_x", "PDDXDT_x", "RATE_LR_CONTINUOUS", "DVT_SFTANIM", "DVT_SDM",
            "DVT_RECOG_DISC_INDEX", "DVT_RETENTION", "DVT_DELAYED_RECALL", "RATE_LME_CONTINUOUS",
            "RATE_LME_INCLUSION_EXCLUSION_FAST", "RATE_LME_INCLUSION_EXCLUSION_SLOW",
            "RATE_LR_INCLUSION_EXCLUSION_SLOW", "RATE_LR_INCLUSION_EXCLUSION_FAST", "RATE_LME_INCLUSION_EXCLUSION_MAN",
            "RATE_LR_DISCRETE", "total_st_unadj", "total_adj", "total_unadj", "total_st_adj", "total_st_unadj",
            "pt3_adj", "pt3_unadj", "pt3_st_adj", "pt3_st_unadj", "pt3_nst_adj", "pt3_nst_unadj", "total_adj0",
            "total_adj1", "total_adj2", "pt3_adj0", "pt3_adj1", "pt3_adj2", "total_nst_adj", "total_nst_unadj",
            "RATE_LME_DISCRETE", "RATE_LR_INCLUSION_EXCLUSION_MAN", "(Intercept)", "RATE_LME_CONTINUOUS_JIHOON",
            "TIME_SINCE_FIRST_SYMPTOM",
            "NP1COG", "NP1HALL", "NP1DPRS", "NP1ANXS", "NP1APAT", "NP1DDS", "NP1SLPN", "NP1SLPD", "NP1PAIN", "NP1URIN",
            "NP1CNST", "NP1LTHD", "NP1FATG", "NP2SPCH", "NP2SALV", "NP2SWAL", "NP2EAT", "NP2DRES", "NP2HYGN", "NP2HWRT",
            "NP2HOBB", "NP2TURN", "NP2TRMR", "NP2RISE", "NP2WALK", "NP2FREZ", "NP3SPCH", "NP3FACXP", "NP3RIGN",
            "NP3RIGRU", "NP3RIGLU", "PN3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL", "NP3PRSPR",
            "NP3PRSPL", "NP3TTAPR", "NP3TTAPL", "NP3LGAGR", "NP3LGAGL", "NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL",
            "NP3POSTR", "NP3BRADY", "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL",
            "NP3RTALL", "NP3RTALJ", "NP3RTCON", "GENDER.y_M_1", "GENDER.y_FNC_0", "GENDER.y_FNS_1", "JLO_TOTRAW",
            "HVLTRT1 à DVT_TOTAL_RECALL", "LNS1A à LNS7C", "TMDISMED", "CNTRLDSM", "HTCM", "TEMPC", "BPARM", "BIOMOM",
            "BIODAD", "FULSIB", "HAFSIB", "MAGPAR", "PAGPAR", "MATAU", "PATAU", "KIDSNUM", "JLO_TOTRAW", "HVLTRT1",
            "HVLTRT2", "HVLTRT3", "HVLTDLY", "HVLTREC", "HVLTFPRL", "HVLTFPUN", "HVLTVRSN", "DVT_TOTAL_RECALL",
            "LNS1A", "LNS1B", "LNS1C", "LNS2A", "LNS2B", "LNS2C", "LNS3A", "LNS3B", "LNS3C", "LNS4A", "LNS4B", "LNS4C",
            "LNS5A", "LNS5B", "LNS5C", "LNS6A", "LNS6B", "LNS6C", "LNS7A", "LNS7B", "LNS7C", "CNTRLDSM", "UPSITBK1",
            "UPSITBK2", "UPSITBK3", "UPSITBK4", "HTCM", "TEMPC", "BPARM", "CSFHemoglobin", "CAUDATE_R", "CAUDATE_L",
            "PUTAMEN_R", "PUTAMEN_L", "Totaltau", "Abeta42", "Total Protein", "Basophils", "Eosinophils",
            "Neutrophils", "Monocytes", "Lymphocytes", "Totaltau (%)", "Abeta42 (%)", "Total Protein (%)",
            "Basophils (%)", "Eosinophils (%)", "Neutrophils (%)", "Monocytes (%)", "Lymphocytes (%)"]

    # Retrieve Jihoon's LME data for "post_lme_data"
    # lme_data = retrieve_data('data/output/updrs_lme.csv', keys=["PATNO"])

    print("\n\n--------------------------------------------------------------------------------------------------------"
          "-------------------------------------------------------")
    print("Rate of progression - UPDRS III - LME - inclusion/exclusion -2 cutoff - PD cohort - untreated")
    print("------------------------------------------------------------------------------------------------------------"
          "---------------------------------------------------\n")

    run(patient_key="PATNO", time_key="TIME_FROM_BL", model_type="rate_of_progression", is_regressor=False,
        base_target="UPDRS_III", outcome_measure="RATE_LME_INCLUSION_EXCLUSION_MAN", add_predictors=add,
        drop_predictors=drop, do_grid_search=True, treated_untreated="untreated", cutoff=-2,
        balance_classes=True, na_elimination_n=None, optimize_precision=True, cohorts=["PD"], post_lme_data=None,
        data_merged_sc_into_bl_file_path="data/raw_data/data_merged_SC_into_BL.csv")

    print("\n\n--------------------------------------------------------------------------------------------------------"
          "-------------------------------------------------------")
    print("Rate of progression - UPDRS III - LME - inclusion/exclusion -1 cutoff - PD cohort - untreated")
    print("------------------------------------------------------------------------------------------------------------"
          "---------------------------------------------------\n")

    run(patient_key="PATNO", time_key="TIME_FROM_BL", model_type="rate_of_progression", is_regressor=False,
        base_target="UPDRS_III", outcome_measure="RATE_LME_INCLUSION_EXCLUSION_SLOW", add_predictors=add,
        drop_predictors=drop, do_grid_search=True, treated_untreated="untreated", cutoff=-1,
        balance_classes=True, na_elimination_n=None, optimize_precision=True, cohorts=["PD"], post_lme_data=None,
        data_merged_sc_into_bl_file_path="data/raw_data/data_merged_SC_into_BL.csv")

    print("\n\n--------------------------------------------------------------------------------------------------------"
          "-------------------------------------------------------")
    print("Rate of progression - UPDRS III - LME - inclusion/exclusion 'slow' tertile - PD cohort - untreated")
    print("------------------------------------------------------------------------------------------------------------"
          "---------------------------------------------------\n")

    run(patient_key="PATNO", time_key="TIME_FROM_BL", model_type="rate_of_progression", is_regressor=False,
        base_target="UPDRS_III", outcome_measure="RATE_LME_INCLUSION_EXCLUSION_SLOW", add_predictors=add,
        drop_predictors=drop, do_grid_search=True, treated_untreated="untreated", on_off_dose="off",
        balance_classes=True, na_elimination_n=None, optimize_precision=True, cohorts=["PD"], post_lme_data=None,
        data_merged_sc_into_bl_file_path="data/raw_data/data_merged_SC_into_BL.csv")

    print("\n\n--------------------------------------------------------------------------------------------------------"
          "-------------------------------------------------------")
    print("Rate of progression - UPDRS III - LME - inclusion/exclusion 'slow' tertile - PD, GRPD, GCPD, CONTROL cohorts"
          " - untreated")
    print("------------------------------------------------------------------------------------------------------------"
          "---------------------------------------------------\n")

    run(patient_key="PATNO", time_key="TIME_FROM_BL", model_type="rate_of_progression", is_regressor=False,
        base_target="UPDRS_III", outcome_measure="RATE_LME_INCLUSION_EXCLUSION_SLOW", add_predictors=add,
        drop_predictors=drop, do_grid_search=True, treated_untreated="untreated", on_off_dose="off",
        balance_classes=True, na_elimination_n=None, optimize_precision=True, cohorts=["PD", "GRPD", "GCPD", "CONTROL"],
        post_lme_data=None, data_merged_sc_into_bl_file_path="data/raw_data/data_merged_SC_into_BL.csv")

    print("\n\n--------------------------------------------------------------------------------------------------------"
          "-------------------------------------------------------")
    print("Rate of progression - UPDRS III - LME - inclusion/exclusion 'slow' tertile - PD, CONTROL cohorts"
          " - untreated")
    print("------------------------------------------------------------------------------------------------------------"
          "---------------------------------------------------\n")

    run(patient_key="PATNO", time_key="TIME_FROM_BL", model_type="rate_of_progression", is_regressor=False,
        base_target="UPDRS_III", outcome_measure="RATE_LME_INCLUSION_EXCLUSION_SLOW", add_predictors=add,
        drop_predictors=drop, do_grid_search=True, treated_untreated="untreated", on_off_dose="off",
        balance_classes=True, na_elimination_n=None, optimize_precision=True, cohorts=["PD", "CONTROL"],
        post_lme_data=None, data_merged_sc_into_bl_file_path="data/raw_data/data_merged_SC_into_BL.csv")

    print("\n\n--------------------------------------------------------------------------------------------------------"
          "-------------------------------------------------------")
    print("Rate of progression - UPDRS III - LME - inclusion/exclusion 'slow' tertile - PD cohort - treated  and "
          "untreated'off' dose")
    print("------------------------------------------------------------------------------------------------------------"
          "---------------------------------------------------\n")

    run(patient_key="PATNO", time_key="TIME_FROM_BL", model_type="rate_of_progression", is_regressor=False,
        base_target="UPDRS_III", outcome_measure="RATE_LME_INCLUSION_EXCLUSION_SLOW", add_predictors=add,
        drop_predictors=drop, do_grid_search=True, treated_untreated="treated_and_untreated", on_off_dose="off",
        balance_classes=True, na_elimination_n=None, optimize_precision=True, cohorts=["PD"], post_lme_data=None,
        data_merged_sc_into_bl_file_path="data/raw_data/data_merged_SC_into_BL.csv")

    print("\n\n--------------------------------------------------------------------------------------------------------"
          "-------------------------------------------------------")
    print("Rate of progression - UPDRS III - LME - inclusion/exclusion -2 cutoff - PD cohort - treated  and "
          "untreated'off' dose")
    print("------------------------------------------------------------------------------------------------------------"
          "---------------------------------------------------\n")

    run(patient_key="PATNO", time_key="TIME_FROM_BL", model_type="rate_of_progression", is_regressor=False,
        base_target="UPDRS_III", outcome_measure="RATE_LME_INCLUSION_EXCLUSION_MAN", cutoff=-2, add_predictors=add,
        drop_predictors=drop, do_grid_search=True, treated_untreated="treated_and_untreated", on_off_dose="off",
        balance_classes=True, na_elimination_n=None, optimize_precision=True, cohorts=["PD"], post_lme_data=None,
        data_merged_sc_into_bl_file_path="data/raw_data/data_merged_SC_into_BL.csv")

    print("\n\n--------------------------------------------------------------------------------------------------------"
          "-------------------------------------------------------")
    print("Future severity - UPDRS III - PD cohort - untreated")
    print("------------------------------------------------------------------------------------------------------------"
          "---------------------------------------------------\n")

    drop += ["UPDRS_III"]
    run(patient_key="PATNO", time_key="TIME_FROM_BL", model_type="future_severity", is_regressor=True,
        base_target="UPDRS_III", outcome_measure="SCORE_FUTURE", add_predictors=add,
        drop_predictors=drop, do_grid_search=True, treated_untreated="untreated", on_off_dose="off",
        na_elimination_n=None, cohorts=["PD"],
        data_merged_sc_into_bl_file_path="data/raw_data/data_merged_SC_into_BL.csv")

    print("\n\n--------------------------------------------------------------------------------------------------------"
          "-------------------------------------------------------")
    print("Future severity - UPDRS III - PD cohort - treated and untreated")
    print("------------------------------------------------------------------------------------------------------------"
          "---------------------------------------------------\n")

    drop += ["UPDRS_III"]
    run(patient_key="PATNO", time_key="TIME_FROM_BL", model_type="future_severity", is_regressor=True,
        base_target="UPDRS_III", outcome_measure="SCORE_FUTURE", add_predictors=add,
        drop_predictors=drop, do_grid_search=True, treated_untreated="treated_and_untreated", on_off_dose="off",
        na_elimination_n=None, cohorts=["PD"],
        data_merged_sc_into_bl_file_path="data/raw_data/data_merged_SC_into_BL.csv")

    print("\n\n--------------------------------------------------------------------------------------------------------"
          "-------------------------------------------------------")
    print("Future severity - UPDRS III - PD, GRPD, GCPD, CONTROL cohorts - untreated")
    print("------------------------------------------------------------------------------------------------------------"
          "---------------------------------------------------\n")

    run(patient_key="PATNO", time_key="TIME_FROM_BL", model_type="future_severity", is_regressor=True,
        base_target="UPDRS_III", outcome_measure="SCORE_FUTURE", add_predictors=add,
        drop_predictors=drop, do_grid_search=True, treated_untreated="untreated", on_off_dose="off",
        na_elimination_n=None, cohorts=["PD", "GRPD", "GCPD", "CONTROL"],
        data_merged_sc_into_bl_file_path="data/raw_data/data_merged_SC_into_BL.csv")
