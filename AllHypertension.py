import pandas as pd
from pandas.tseries.offsets import Day
import numpy as np


def main():
    # Set seed
    np.random.seed(0)

    # Create the dataframe from file
    data = pd.read_csv("data/all_bp.csv")

    # Set dawn and dusk constants
    dawn = pd.Timestamp("04:24:00")
    dusk = pd.Timestamp("15:40:00")

    # Convert datetimes to pandas datetimes
    data["date_time_local"] = pd.to_datetime(data["date_time_local"])

    # Find previous dawn before a time
    def find_previous_dawn(first_date_time, last_date_time):
        # First day's dawn and last day's dawn
        first_dawn = pd.Timestamp(first_date_time.date + " " + dawn)
        last_dawn = pd.Timestamp(last_date_time.date + " " + dawn)

        # Previous dawn
        if first_date_time < first_dawn:
            first_dawn = first_dawn - Day(1)

        # Next dawn
        if last_date_time > last_dawn:
            last_dawn = last_dawn + Day(1)

        # Return first and last dawn
        return first_dawn, last_dawn

    # Dataframe for storing final result
    results = pd.DataFrame(columns=[])

    # Iterate through each patient
    for patient in data["id"].unique():
        # Initialize time as first dawn before earliest observation
        time, last_time = find_previous_dawn(data.loc[data["id"] == patient, "date_time_local"].min())

        # Iterate by 24 hour periods from dawn to dawn
        while time != last_time:
            # Observations of this patient on this time interval
            observations = data[data["id"] == patient & data["date_time_local"].between(time, time + Day(1))]

            # Iterate by a day
            time = time + Day(1)


if __name__ == "__main__":
    main()
