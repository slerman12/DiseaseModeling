from __future__ import division
import math
import pandas as pd
from pandas.tseries.offsets import Day
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Set seed
    np.random.seed(0)

    # Create the dataframe from file
    data = pd.read_csv("data/all_bp.csv")

    # Set columns
    columns = ["ID", "DAY", "DATE_TIME_CENTRAL_SIT", "DATE_TIME_CENTRAL_STAND", "DATE_TIME_LOCAL_SIT",
               "DATE_TIME_LOCAL_STAND", "TIME_DIFF", "MORNINGNIGHT", "COMPLIANCE"]

    # Convert datetimes to pandas datetimes
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

    # Dataframe for storing final result
    result = pd.DataFrame(columns=columns)

    # Function to set features and values to a row and append row to result
    def set_add_row(row_observations, row):
        # Find local and central times of first sit observation for row row
        first_sit_date_time_local = row_observations.loc[
            row_observations["state"] == "sit", "date_time_local"].min()
        first_sit_date_time_central = row_observations.loc[
            row_observations["state"] == "sit", "timeOfDay_central"].min()

        # If first sit exists
        if first_sit_date_time_local is not None and pd.notnull(first_sit_date_time_local):
            # Find local and central times of next stand observation for row
            next_stand_date_time_local = row_observations.loc[
                (row_observations["state"] == "stand") &
                (row_observations["date_time_local"] > first_sit_date_time_local), "date_time_local"].min()
            next_stand_date_time_central = row_observations.loc[
                (row_observations["state"] == "stand") &
                (row_observations["timeOfDay_central"] > first_sit_date_time_central), "timeOfDay_central"].min()

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
                    (row_observations["timeOfDay_central"] < next_stand_date_time_central), "timeOfDay_central"].max()

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
                row_observations["state"] == "stand", "timeOfDay_central"].min()

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

    # Iterate through each patient
    for patient in data["id"].unique():
        # Initialize time as first dawn before earliest observation
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

            # Set row IDs and morning/night
            morning_row["ID"] = patient
            morning_row["DAY"] = day_count
            morning_row["MORNINGNIGHT"] = "M"
            night_row["ID"] = patient
            night_row["DAY"] = day_count
            night_row["MORNINGNIGHT"] = "N"

            # Append morning and night rows
            set_add_row(morning_observations, morning_row)
            set_add_row(night_observations, night_row)

            # Iterate by a day
            time = time + Day(1)

    # Output result with time diffs as minutes
    result["TIME_DIFF"] = pd.to_timedelta(result["TIME_DIFF"]).dt.seconds / 60

    # Output results to csv
    result.to_csv("data/All_Hypertension_Results.csv", index=False)


def stats():
    result = pd.read_csv("data/All_Hypertension_Results.csv")

    # Print stats
    print("Total noncompliance: {}/{} [{:.2%}]".format(len(result[result["COMPLIANCE"] == 0].index),
                                                       len(result.index),
                                                       len(result[result["COMPLIANCE"] == 0].index) /
                                                       len(result.index)))
    print("Mean time diff (Minutes): {}".format(result["TIME_DIFF"].mean()))
    print("Max time diff (Minutes): {}".format(result["TIME_DIFF"].max()))
    print("Min time diff (Minutes): {}".format(result["TIME_DIFF"].min()))

    # Plot histogram
    result["TIME_DIFF"].plot(kind="hist", bins=range(0, 15, 1), facecolor="pink")
    plt.axis([0, 15, 0, 15000])
    plt.xlabel("Time Difference (Minutes)")
    plt.ylabel("Number of Observations")
    plt.show()


if __name__ == "__main__":
    stats()
