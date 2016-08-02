from __future__ import division
import pandas as pd


def main():
    # Create the data frame from file
    # data = pd.read_csv("data/BP_data_source.csv")
    # data[["id", "sit_stand"]] = data["cno-id-sit/stand"].str.split(pat="-").apply(pd.Series)[[1, 2]]
    # data["date_time"] = pd.to_datetime(data["date_time"])
    # data = data.drop("cno-id-sit/stand", 1)
    # data = data.drop("Unnamed: 4", 1)
    # data = data[data["id"].apply(lambda x: str(x).isdigit())]
    # data = data.dropna()
    # data["id"] = data["id"].astype("int64")
    # data.to_csv("data/BP_Data_Final.csv")

    # Retrieve files
    bp_data = pd.read_csv("data/BP_Data_Final.csv")
    time_data = pd.read_csv("data/All_Hypertension_Results_With_Timeframe.csv")

    # Merge files
    sit_stand = bp_data[(bp_data["measurement"] == "BP Systolic") & (bp_data["sit_stand"] == "sit")].drop(
        "measurement", 1).drop("sit_stand", 1).drop("Unnamed: 0", 1)
    merge = pd.merge(left=time_data, right=sit_stand, how="left", left_on=["ID", "DATE_TIME_CENTRAL_SIT"],
                             right_on=["id", "date_time"]).drop(["date_time", "id"], 1)
    merge.rename(columns={"value": "SIT_SYS"}, inplace=True)

    sit_stand = bp_data[(bp_data["measurement"] == "BP Diastolic") & (bp_data["sit_stand"] == "sit")].drop(
        "measurement", 1).drop("sit_stand", 1).drop("Unnamed: 0", 1)
    merge = pd.merge(left=merge, right=sit_stand, how="left", left_on=["ID", "DATE_TIME_CENTRAL_SIT"],
                             right_on=["id", "date_time"]).drop(["date_time", "id"], 1)
    merge.rename(columns={"value": "SIT_DIA"}, inplace=True)

    sit_stand = bp_data[(bp_data["measurement"] == "BP Systolic") & (bp_data["sit_stand"] == "stand")].drop(
        "measurement", 1).drop("sit_stand", 1).drop("Unnamed: 0", 1)
    merge = pd.merge(left=merge, right=sit_stand, how="left", left_on=["ID", "DATE_TIME_CENTRAL_STAND"],
                     right_on=["id", "date_time"]).drop(["date_time", "id"], 1)
    merge.rename(columns={"value": "STD_SYS"}, inplace=True)

    sit_stand = bp_data[(bp_data["measurement"] == "BP Diastolic") & (bp_data["sit_stand"] == "stand")].drop(
        "measurement", 1).drop("sit_stand", 1).drop("Unnamed: 0", 1)
    merge = pd.merge(left=merge, right=sit_stand, how="left", left_on=["ID", "DATE_TIME_CENTRAL_STAND"],
                     right_on=["id", "date_time"]).drop(["date_time", "id"], 1)
    merge.rename(columns={"value": "STD_DIA"}, inplace=True)

    sit_stand = bp_data[(bp_data["measurement"] == "BP Heartrate") & (bp_data["sit_stand"] == "sit")].drop(
        "measurement", 1).drop("sit_stand", 1).drop("Unnamed: 0", 1)
    merge = pd.merge(left=merge, right=sit_stand, how="left", left_on=["ID", "DATE_TIME_CENTRAL_SIT"],
                     right_on=["id", "date_time"]).drop(["date_time", "id"], 1)
    merge.rename(columns={"value": "SIT_HRT_RATE"}, inplace=True)

    sit_stand = bp_data[(bp_data["measurement"] == "BP Heartrate") & (bp_data["sit_stand"] == "stand")].drop(
        "measurement", 1).drop("sit_stand", 1).drop("Unnamed: 0", 1)
    merge = pd.merge(left=merge, right=sit_stand, how="left", left_on=["ID", "DATE_TIME_CENTRAL_STAND"],
                     right_on=["id", "date_time"]).drop(["date_time", "id"], 1)
    merge.rename(columns={"value": "STD_HRT_RATE"}, inplace=True)

    merge["BP_SYS_DIFF"] = merge["SIT_SYS"] - merge["STD_SYS"]

    merge.to_csv("data/All_BP_Results.csv")


if __name__ == "__main__":
    main()
