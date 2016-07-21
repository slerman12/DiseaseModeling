import math
import pandas as pd
import sys
import MachineLearning as mL
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


def main():
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

    # Create csv of pd/control patient data
    pd_control_data.to_csv("data/pd_control_data_before_merge.csv", index=False)

    # Merge with updrs scores
    pd_control_data = pd_control_data.merge(all_updrs, on=["PATNO", "EVENT_ID"], how="left")

    # Create csv of pd/control patient data
    # pd_control_data.to_csv("data/pd_control_data_with_nulls.csv", index=False)

    # Get rid of nulls
    pd_control_updrs_data = pd_control_data[pd_control_data["TOTAL"].notnull()]

    # Merge with patient info
    pd_control_updrs_data = pd_control_updrs_data.merge(all_patients, on="PATNO", how="left")

    # Eliminate features with more than 20% NAs
    for feature in pd_control_updrs_data.keys():
        if len(pd_control_updrs_data.loc[pd_control_updrs_data[feature].isnull(), feature]) / len(
                pd_control_updrs_data[feature]) > 0.2:
            pd_control_updrs_data = pd_control_updrs_data.drop(feature, 1)

    # Drop rows with NAs
    pd_control_updrs_data = pd_control_updrs_data.dropna()

    # Only include baseline and subsequent visits
    pd_control_updrs_data = pd_control_updrs_data[
        (pd_control_updrs_data["EVENT_ID"] != "SC") & (pd_control_updrs_data["EVENT_ID"] != "ST") & (
            pd_control_updrs_data["EVENT_ID"] != "U01") & (pd_control_updrs_data["EVENT_ID"] != "PW")]

    # Encode EVENT_ID to numeric
    mL.clean_data(data=pd_control_updrs_data, encode_auto=["GENDER.x", "DIAGNOSIS", "HANDED"], encode_man={
        "EVENT_ID": {"BL": 0, "V01": 1, "V02": 2, "V03": 3, "V04": 4, "V05": 5, "V06": 6, "V07": 7, "V08": 8, "V09": 9,
                     "V10": 10, "V11": 11, "V12": 12}})

    # Drop duplicates
    pd_control_updrs_data = pd_control_updrs_data.drop_duplicates(subset=["PATNO", "EVENT_ID"])

    # Predictors for the model
    predictors = ["VISIT_NEXT", "SCORE_NOW", "TEMPC", "BPARM", "SYSSUP", "DIASUP", "HRSUP", "SYSSTND", "DIASTND",
                  "HRSTND", "AE", "GENDER.x", "DIAGNOSIS", "CNO", "OTHCOND", "BIOMOM", "BIOMOMPD", "BIODAD", "BIODADPD",
                  "FULSIB", "FULSIBPD", "HAFSIB", "HAFSIBPD", "MAGPAR", "MAGPARPD", "PAGPAR", "PAGPARPD", "MATAU",
                  "MATAUPD", "PATAU", "PATAUPD", "KIDSNUM", "KIDSPD", "INITMD", "BIRTHDT.y", "HISPLAT", "RAINDALS",
                  "RAASIAN", "RABLACK", "RAHAWOPI", "RAWHITE", "RANOS", "EDUCYRS", "HANDED"]

    # Target for the model
    target = "SCORE_NEXT"

    # Generate new features
    # train = generate_features(data=pd_control_updrs_data, predictors=predictors, target=target, id_name="PATNO",
    #                           score_name="TOTAL",
    #                           visit_name="EVENT_ID")

    # Save generated features data
    # train.to_csv("data/PPMI_train.csv", index=False)

    # Retrieve generated features data
    train = pd.read_csv("data/PPMI_train.csv")

    # Value counts for EVENT_ID after feature generation
    mL.describe_data(data=train, describe=True, description="AFTER FEATURE GENERATION:")

    # Univariate feature selection
    mL.describe_data(data=train, univariate_feature_selection=[predictors, target])

    # Algs for model
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

    # Parameters for grid search
    grid_search_params = [{"n_estimators": [50, 150, 300, 500, 750, 1000],
                           "min_samples_split": [4, 8, 25, 50, 75, 100],
                           "min_samples_leaf": [2, 8, 15, 25, 50, 75, 100]}]

    # Ensemble
    ens = mL.ensemble(algs=algs, alg_names=alg_names,
                      ensemble_name="Weighted ensemble of RF, LR, SVM, GNB, KNN, and GB",
                      in_ensemble=[True, True, True, True, False, False, True, True], weights=[3, 2, 1, 3, 1, 3],
                      voting="soft")

    # Add ensemble to algs and alg_names
    # algs.append(ens["alg"])
    # alg_names.append(ens["name"])

    # Display ensemble metrics
    mL.metrics(data=train, predictors=predictors, target=target, algs=algs, alg_names=alg_names,
               feature_importances=[True], base_score=[True], oob_score=[True],
               cross_val=[True], scoring="mean_absolute_error", split_accuracy=[True],
               grid_search_params=None)


def generate_features(data, predictors, target, id_name, score_name, visit_name):
    # Set features
    features = predictors + [target]

    # Set max visit
    max_visit = data[visit_name].max()

    # Generate SCORE_NOW and VISIT_NOW
    data["SCORE_NOW"] = data[score_name]
    data["VISIT_NOW"] = data[visit_name]

    # Create new dataframe
    new_data = pd.DataFrame(columns=features)

    # Initialize progress measures
    progress_complete = 0
    progress_total = len(data)

    # Build new data (generate SCORE_NEXT, VISIT_NEXT, and TIME_PASSED)
    for index, row in data.iterrows():
        # Update progress
        progress_complete += 1
        sys.stdout.write("\rProgress: {:.2%}".format(progress_complete / progress_total))
        sys.stdout.flush()

        # If now visit isn't the max
        if row["VISIT_NOW"] == 0:
            # For the range of all visits after this one
            for i in range(1, max_visit + 1):
                # If any future visit belongs to the same patient
                if any((data["VISIT_NOW"] == row["VISIT_NOW"] + i) & (data[id_name] == row[id_name])):
                    # Set next score
                    row["SCORE_NEXT"] = data.loc[(data["VISIT_NOW"] == row["VISIT_NOW"] + i) &
                                                 (data[id_name] == row[id_name]), "SCORE_NOW"].item()

                    # Set next visit
                    row["VISIT_NEXT"] = data.loc[(data["VISIT_NOW"] == row["VISIT_NOW"] + i) &
                                                 (data[id_name] == row[id_name]), "VISIT_NOW"].item()

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


if __name__ == "__main__":
    main()
