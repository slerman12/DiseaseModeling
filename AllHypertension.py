import pandas as pd
from pandas.tseries.offsets import Day
import numpy as np


def main():
    # Set seed
    np.random.seed(0)

    # Create the training/test set(s) from file(s)
    data = pd.read_csv("data/all_bp.csv")

    # Set dawn and dusk constants
    dawn = pd.Timestamp("04:24:00")
    dusk = pd.Timestamp("15:40:00")

    # Convert datetimes to pandas datetimes
    data["date_time_local"] = pd.to_datetime(data["date_time_local"])

    # Find first previous dawn before a time
    def find_previous_dawn(date_time):
        # Today's dawn
        today_dawn = pd.Timestamp(date_time.date + " " + dawn)

        # Return first previous dawn
        if date_time >= today_dawn:
            return today_dawn
        else:
            return today_dawn - Day(1)

    # Iterate through each patient
    for patient in data["id"].unique():
        # Initialize time as first dawn before earliest observation
        time = find_previous_dawn(data.loc[data["id"] == patient, "date_time_local"].min())


if __name__ == "__main__":
    main()
