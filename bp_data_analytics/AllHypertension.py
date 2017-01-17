from __future__ import division
import math
import pandas as pd
import sys
from pandas.tseries.offsets import Day
import numpy as np
import matplotlib.pyplot as plt

# Set columns
columns = ["ID", "DAY", "DATE_TIME_CENTRAL_SIT", "DATE_TIME_CENTRAL_STAND", "DATE_TIME_LOCAL_SIT",
           "DATE_TIME_LOCAL_STAND", "TIME_DIFF", "MORNINGNIGHT", "TIMEFRAME", "COMPLIANCE"]


def main():
    # Create the data frame from file
    data = pd.read_csv("data/all_bp.csv")
    visits_data = pd.read_csv("data/STEADY3_VISITS.csv")

    # Convert date-times to pandas date-times
    data["date_time_local"] = pd.to_datetime(data["date_time_local"])

    # Find previous dawn before a time, and next dawn after another time
    def find_first_last_dawn(first_date_time, last_date_time):
        # First day's dawn and last day's dawn
        first_dawn = pd.Timestamp(first_date_time.date() + pd.DateOffset(hours=4, minutes=24))
        last_dawn = pd.Timestamp(last_date_time.date() + pd.DateOffset(hours=4, minutes=24))

        # Previous dawn
        if first_date_time < first_dawn:
            first_dawn = first_dawn - Day(1)

        # Next dawn
        if last_date_time > last_dawn:
            last_dawn = last_dawn + Day(1)

        # Return first and last dawn
        return first_dawn, last_dawn

    # Data frame for storing final result
    result = pd.DataFrame(columns=columns)

    # Function to set features and values to a row and append row to result
    def set_add_row(row_observations, row):
        # Find local and central times of first sit observation for row
        first_sit_date_time_local = row_observations.loc[
            row_observations["state"] == "sit", "date_time_local"].min()
        first_sit_date_time_central = row_observations.loc[
            row_observations["state"] == "sit", "date_time"].min()

        # If first sit exists
        if first_sit_date_time_local is not None and pd.notnull(first_sit_date_time_local):
            # Index of first sit
            sit_index = row_observations[(row_observations["state"] == "sit") &
                                         (row_observations["date_time_local"] == first_sit_date_time_local)].index.min()

            # Stand observation recorded at the same time (to the nearest minute) but has higher index
            equal_time_stand = row_observations[(row_observations["state"] == "stand") &
                                                (row_observations["date_time_local"] == first_sit_date_time_local) &
                                                (row_observations.index.max() > sit_index)]

            # Account for sit/stand pairings recorded with equal time by comparing indices
            if not equal_time_stand.empty:
                # Choose stand
                next_stand_date_time_local = equal_time_stand["date_time_local"].min()
                next_stand_date_time_central = equal_time_stand["date_time"].min()
                prev_sit_date_time_local = first_sit_date_time_local
                prev_sit_date_time_central = first_sit_date_time_central

                # Fill sit data
                row["DATE_TIME_LOCAL_SIT"] = prev_sit_date_time_local
                row["DATE_TIME_CENTRAL_SIT"] = prev_sit_date_time_central

                # Fill stand data
                row["DATE_TIME_LOCAL_STAND"] = next_stand_date_time_local
                row["DATE_TIME_CENTRAL_STAND"] = next_stand_date_time_central

                # Set time difference
                row["TIME_DIFF"] = next_stand_date_time_local - first_sit_date_time_local

                # Complying
                row["COMPLIANCE"] = 1

            else:
                # Find local and central times of next stand observation for row
                next_stand_date_time_local = row_observations.loc[
                    (row_observations["state"] == "stand") &
                    (row_observations["date_time_local"] > first_sit_date_time_local), "date_time_local"].min()
                next_stand_date_time_central = row_observations.loc[
                    (row_observations["state"] == "stand") &
                    (row_observations["date_time"] > first_sit_date_time_central), "date_time"].min()

                # If next stand exists
                if next_stand_date_time_local is not None and pd.notnull(next_stand_date_time_local):
                    # Did both sit and stand

                    # Fill stand data
                    row["DATE_TIME_LOCAL_STAND"] = next_stand_date_time_local
                    row["DATE_TIME_CENTRAL_STAND"] = next_stand_date_time_central

                    # Find preceding sit
                    prev_sit_date_time_local = row_observations.loc[
                        (row_observations["state"] == "sit") &
                        (row_observations["date_time_local"] < next_stand_date_time_local), "date_time_local"].max()
                    prev_sit_date_time_central = row_observations.loc[
                        (row_observations["state"] == "sit") &
                        (row_observations[
                             "date_time"] < next_stand_date_time_central), "date_time"].max()

                    # Fill sit data
                    row["DATE_TIME_LOCAL_SIT"] = prev_sit_date_time_local
                    row["DATE_TIME_CENTRAL_SIT"] = prev_sit_date_time_central

                    # Set time difference
                    row["TIME_DIFF"] = next_stand_date_time_local - first_sit_date_time_local

                    # Complying
                    row["COMPLIANCE"] = 1
                else:
                    # Skipped stand

                    # Fill sit data
                    row["DATE_TIME_LOCAL_SIT"] = first_sit_date_time_local
                    row["DATE_TIME_CENTRAL_SIT"] = first_sit_date_time_central

                    # Fill stand as NA
                    row["DATE_TIME_LOCAL_STAND"] = None
                    row["DATE_TIME_CENTRAL_STAND"] = None

                    # No time diff
                    row["TIME_DIFF"] = None

                    # Noncomplying
                    row["COMPLIANCE"] = 0
        else:
            # Skipped sit

            # Fill sit as NA
            row["DATE_TIME_LOCAL_SIT"] = None
            row["DATE_TIME_CENTRAL_SIT"] = None

            # No time diff
            row["TIME_DIFF"] = None

            # Noncomplying
            row["COMPLIANCE"] = 0

            # Find local and central times of first stand
            first_stand_date_time_local = row_observations.loc[
                row_observations["state"] == "stand", "date_time_local"].min()
            first_stand_date_time_central = row_observations.loc[
                row_observations["state"] == "stand", "date_time"].min()

            if first_sit_date_time_local is not None and pd.notnull(first_sit_date_time_local):
                # Fill first stand data
                row["DATE_TIME_LOCAL_STAND"] = first_stand_date_time_local
                row["DATE_TIME_CENTRAL_STAND"] = first_stand_date_time_central
            else:
                # Skipped both sit and stand

                # Fill stand as NA
                row["DATE_TIME_LOCAL_STAND"] = None
                row["DATE_TIME_CENTRAL_STAND"] = None

        # Add rows to new_data
        max_index = result.index.max()
        for key in row.keys():
            if not math.isnan(max_index):
                result.loc[max_index + 1, key] = row[key]
            else:
                result.loc[0, key] = row[key]

    # Total patient count
    patients_total = len(visits_data.loc[visits_data["Subject"].isin(data["id"]), "Subject"].unique())

    # Iterate through each patient
    for index, patient in enumerate(visits_data.loc[visits_data["Subject"].isin(data["id"]), "Subject"].unique()):
        # Get patient's timeframe dates
        if not visits_data.loc[visits_data["Subject"] == patient, "RS2"].any():
            if not visits_data.loc[visits_data["Subject"] == patient, "RS1"].any():
                sc = pd.Timestamp(visits_data.loc[visits_data["Subject"] == patient, "SC"].min())
            else:
                sc = pd.Timestamp(visits_data.loc[visits_data["Subject"] == patient, "RS1"].min())
        else:
            sc = pd.Timestamp(visits_data.loc[visits_data["Subject"] == patient, "RS2"].min())
        if visits_data.loc[visits_data["Subject"] == patient, "BL"].any():
            bl = pd.Timestamp(visits_data.loc[visits_data["Subject"] == patient, "BL"].min())
            if visits_data.loc[visits_data["Subject"] == patient, "V01"].any():
                v01 = pd.Timestamp(visits_data.loc[visits_data["Subject"] == patient, "V01"].min())
                if visits_data.loc[visits_data["Subject"] == patient, "V02"].any():
                    v02 = pd.Timestamp(visits_data.loc[visits_data["Subject"] == patient, "V02"].min())
                else:
                    v02 = None
            else:
                v01 = None
                v02 = None
        else:
            bl = None
            v01 = None
            v02 = None

        # Initialize times as first dawn before earliest observation, and last dawn after last observation
        time, last_time = find_first_last_dawn(data.loc[data["id"] == patient, "date_time_local"].min(),
                                               data.loc[data["id"] == patient, "date_time_local"].max())

        # Initialize day count
        day_count = 0

        # Iterate by 24 hour periods from dawn to dawn
        while time != last_time:
            # Increment day count
            day_count += 1

            # Initialize morning and night rows to be appended to result
            morning_row = {}
            night_row = {}

            # Observations of this patient on this time interval
            observations = data[(data["id"] == patient) & (data["date_time_local"].between(time, time + Day(1)))]

            # Divide the observations into morning and night
            morning_observations = observations[observations["ampm"] == "M"]
            night_observations = observations[observations["ampm"] == "N"]

            # Set row IDs, day, and morning/night
            morning_row["ID"] = patient
            morning_row["DAY"] = day_count
            morning_row["MORNINGNIGHT"] = "M"
            night_row["ID"] = patient
            night_row["DAY"] = day_count
            night_row["MORNINGNIGHT"] = "N"

            # Set timeframe
            if time < sc:
                morning_row["TIMEFRAME"] = "Before SC"
                night_row["TIMEFRAME"] = "Before SC"
            else:
                if bl is not None:
                    if sc <= time < bl:
                        morning_row["TIMEFRAME"] = "SC to BL"
                        night_row["TIMEFRAME"] = "SC to BL"
                    else:
                        if v01 is not None:
                            if bl <= time < v01:
                                morning_row["TIMEFRAME"] = "BL to V01"
                                night_row["TIMEFRAME"] = "BL to V01"
                            else:
                                if v02 is not None:
                                    if v01 <= time < v02:
                                        morning_row["TIMEFRAME"] = "V01 to V02"
                                        night_row["TIMEFRAME"] = "V01 to V02"
                                    else:
                                        morning_row["TIMEFRAME"] = "After V02"
                                        night_row["TIMEFRAME"] = "After V02"
                                else:
                                    morning_row["TIMEFRAME"] = "After V01"
                                    night_row["TIMEFRAME"] = "After V01"
                        else:
                            morning_row["TIMEFRAME"] = "After BL"
                            night_row["TIMEFRAME"] = "After BL"
                else:
                    morning_row["TIMEFRAME"] = "After SC"
                    night_row["TIMEFRAME"] = "After SC"

            # Append morning and night rows
            set_add_row(morning_observations, morning_row)
            set_add_row(night_observations, night_row)

            # Iterate by a day
            time = time + Day(1)

        # Update progress
        sys.stdout.write("\rProgress: {:.2%}".format(index / patients_total))
        sys.stdout.flush()

    # Output result with time diffs as minutes
    result["TIME_DIFF"] = pd.to_timedelta(result["TIME_DIFF"]).dt.seconds / 60

    # Output results to csv
    result.to_csv("data/All_Hypertension_Results_With_Timeframe.csv", index=False)


def time_frame_compliance():
    # Retrieve results
    result = pd.read_csv("data/All_Hypertension_Results_With_Timeframe.csv")

    # Group by ID, TIMEFRAME, and DAY, and aggregate compliances
    top_7_compliance_per_time_frame = result[["ID", "DAY", "COMPLIANCE", "TIME_DIFF", "TIMEFRAME"]].groupby(
            ["ID", "TIMEFRAME", "DAY"]).agg({"COMPLIANCE": np.mean,
                                             "TIME_DIFF": lambda x: 1 if (2 <= np.mean(x.fillna(0)) <= 15) else (
                                                 0.5 if (
                                                     2 <= x.fillna(0).min() <= 15 or 2 <= x.fillna(0).max() <= 15) else
                                                 0)})

    # Select the 7 best compliance days per time frame for each patient
    top_7_compliance_per_time_frame[["COMPLIANCE", "TIME_DIFF"]] = top_7_compliance_per_time_frame[
        ["COMPLIANCE", "TIME_DIFF"]].groupby(level=["ID", "TIMEFRAME"], group_keys=False).apply(
            lambda x: x.sort_values(by=["COMPLIANCE", "TIME_DIFF"], ascending=False).head(7))

    # Remove other days
    top_7_compliance_per_time_frame = top_7_compliance_per_time_frame[
        pd.notnull(top_7_compliance_per_time_frame["COMPLIANCE"])]

    # Rename compliances
    top_7_compliance_per_time_frame["SIT_STAND_COMPLIANCE"] = top_7_compliance_per_time_frame["COMPLIANCE"]
    top_7_compliance_per_time_frame["TIME_DIFF_COMPLIANCE"] = top_7_compliance_per_time_frame["TIME_DIFF"]
    top_7_compliance_per_time_frame = top_7_compliance_per_time_frame[["TIME_DIFF_COMPLIANCE", "SIT_STAND_COMPLIANCE"]]

    # Output to csv
    top_7_compliance_per_time_frame.to_csv("data/Top_7_Compliances_Per_Time_Frame.csv")

    # Create time frame compliances dataframe
    timeframe_compliances = top_7_compliance_per_time_frame.groupby(level=["ID", "TIMEFRAME"]).agg(
            lambda x: x.sum() / 7).query(
            'TIMEFRAME == "BL to V01" or TIMEFRAME == "SC to BL" or TIMEFRAME == "V01 to V02"')

    # Output to csv
    timeframe_compliances.to_csv("data/Time_Frame_Compliances_Per_Patient.csv")

    # Grouped patient, time frame, and their mean compliance during that time frame
    compliance_means = timeframe_compliances.reset_index().groupby(["ID", "TIMEFRAME"]).mean().reset_index()

    # Initiated dataframe
    timeframe_compliance_means = pd.DataFrame(
            columns=["SC_TO_BL_TIME_DIFF_COMPLIANCE", "SC_TO_BL_SIT_STAND_COMPLIANCE", "BL_TO_V01_TIME_DIFF_COMPLIANCE",
                     "BL_TO_V01_SIT_STAND_COMPLIANCE", "V01_TO_V02_TIME_DIFF_COMPLIANCE",
                     "V01_TO_V02_SIT_STAND_COMPLIANCE"])

    # Set sit/stands
    timeframe_compliance_means[
        ["SC_TO_BL_SIT_STAND_COMPLIANCE", "BL_TO_V01_SIT_STAND_COMPLIANCE", "V01_TO_V02_SIT_STAND_COMPLIANCE"]] = \
        compliance_means.pivot(index="ID", columns="TIMEFRAME", values="SIT_STAND_COMPLIANCE")[
            ["SC to BL", "BL to V01", "V01 to V02"]]

    # Set time diffs
    timeframe_compliance_means[
        ["SC_TO_BL_TIME_DIFF_COMPLIANCE", "BL_TO_V01_TIME_DIFF_COMPLIANCE", "V01_TO_V02_TIME_DIFF_COMPLIANCE"]] = \
        compliance_means.pivot(index="ID", columns="TIMEFRAME", values="TIME_DIFF_COMPLIANCE")[
            ["SC to BL", "BL to V01", "V01 to V02"]]

    # Output to csv
    timeframe_compliance_means.to_csv("data/Time_Frame_Compliances_Per_Patient_As_Features.csv")


def stats():
    # Retrieve results
    result = pd.read_csv("data/All_Hypertension_Results_With_Timeframe.csv")
    top_7 = pd.read_csv("data/Top_7_Compliances_Per_Time_Frame.csv")
    timeframe_compliances = pd.read_csv("data/Time_Frame_Compliances_Per_Patient_As_Features.csv")

    # Timeframe value counts
    print("\nTimeframe Value Counts:")
    print(pd.value_counts(result["TIMEFRAME"]))

    # Print stats
    print("\nMean time diff (Minutes): {}".format(result["TIME_DIFF"].mean()))
    print("Max time diff (Minutes): {}".format(result["TIME_DIFF"].max()))
    print("Min time diff (Minutes): {}\n".format(result["TIME_DIFF"].min()))
    print("Mean time diff compliance from SC to BL: {:.2%} [{} Total Patients]\n"
          "Mean time diff compliance from BL to V01: {:.2%} [{} Total Patients]\n"
          "Mean time diff compliance from V01 to V02: {:.2%} [{} Total Patients]\n\n"
          "Mean sit/stand compliance from SC to BL: {:.2%} [{} Total Patients]\n"
          "Mean sit/stand compliance from BL to V01: {:.2%} [{} Total Patients]\n"
          "Mean sit/stand compliance from V01 to V02: {:.2%} [{} Total Patients]\n".format(
            timeframe_compliances["SC_TO_BL_TIME_DIFF_COMPLIANCE"].mean(),
            timeframe_compliances["SC_TO_BL_TIME_DIFF_COMPLIANCE"].count(),
            timeframe_compliances["BL_TO_V01_TIME_DIFF_COMPLIANCE"].mean(),
            timeframe_compliances["BL_TO_V01_TIME_DIFF_COMPLIANCE"].count(),
            timeframe_compliances["V01_TO_V02_TIME_DIFF_COMPLIANCE"].mean(),
            timeframe_compliances["V01_TO_V02_TIME_DIFF_COMPLIANCE"].count(),
            timeframe_compliances["SC_TO_BL_SIT_STAND_COMPLIANCE"].mean(),
            timeframe_compliances["SC_TO_BL_SIT_STAND_COMPLIANCE"].count(),
            timeframe_compliances["BL_TO_V01_SIT_STAND_COMPLIANCE"].mean(),
            timeframe_compliances["BL_TO_V01_SIT_STAND_COMPLIANCE"].count(),
            timeframe_compliances["V01_TO_V02_SIT_STAND_COMPLIANCE"].mean(),
            timeframe_compliances["V01_TO_V02_SIT_STAND_COMPLIANCE"].count()))
    patient_means = top_7[(top_7["TIMEFRAME"] == "SC to BL") | (top_7["TIMEFRAME"] == "BL to V01") | (
    top_7["TIMEFRAME"] == "V01 to V02")].groupby("ID").agg(lambda x: np.sum(x) / 21)
    print("Mean patient time diff compliance: {}\n"
          "Mean patient sit/stand compliance: {}\n".format(patient_means["TIME_DIFF_COMPLIANCE"].mean(),
                                                           patient_means["SIT_STAND_COMPLIANCE"].mean()))

    # Plot histogram
    result["TIME_DIFF"].plot(kind="hist", bins=range(0, 15, 1), facecolor="pink")
    plt.axis([0, 15, 0, 15000])
    plt.xlabel("Time Difference (Minutes)")
    plt.ylabel("Number of Observations")
    plt.show()

    # Plot histograms
    plt.xlabel("SC to BL Time Diff Compliance (%)")
    plt.ylabel("Number of Patients")
    (timeframe_compliances["SC_TO_BL_TIME_DIFF_COMPLIANCE"] * 100).plot(kind="hist", facecolor="pink")
    plt.show()

    # Plot histograms
    plt.xlabel("BL to V01 Time Diff Compliance (%)")
    plt.ylabel("Number of Patients")
    (timeframe_compliances["BL_TO_V01_TIME_DIFF_COMPLIANCE"] * 100).plot(kind="hist", facecolor="pink")
    plt.show()

    # Plot histograms
    plt.xlabel("V01 to V02 Time Diff Compliance (%)")
    plt.ylabel("Number of Patients")
    (timeframe_compliances["V01_TO_V02_TIME_DIFF_COMPLIANCE"] * 100).plot(kind="hist", facecolor="pink")
    plt.show()


if __name__ == "__main__":
    main()
