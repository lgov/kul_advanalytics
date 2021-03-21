import pandas as pd
import numpy as np


# cleaning functions
def text_to_binary(col_name, bin_1, bin_0, df):
    return df[col_name].replace({bin_0:0,bin_1:1}, inplace=True)


def update_dataset_features(df):
    # convert binary text variables into binary: {"Y":1, "N":0}
    for i in ["fraud", "claim_liable", "claim_police", "driver_injured"]:
        text_to_binary(i, "Y", "N", df)
    # {"P":1, "N":0}
    text_to_binary("claim_alcohol", "P", "N", df)
    # {"car":1, "van":0}
    text_to_binary("claim_vehicle_type", "car", "van", df)
    # {"M":1, "F":0}
    text_to_binary("policy_holder_form", "M", "F", df)
    # {"B":1, "N":0}
    text_to_binary("policy_holder_country", "B", "N", df)
    # make claim_lang binary (currently 1:Dutch, 2:Fr) -> 0: Dutch and 1: French
    df["claim_language"] = df["claim_language"] - 1

    # replace "," with "." and convert to numeric
    df["claim_amount"] = pd.to_numeric(df["claim_amount"].str.replace(",", "."))

    # get dummies for cat vars
    df = pd.get_dummies(df, dummy_na=True, columns=["claim_cause"])

    # format date
    YYYYMMDD_date_columns = ["claim_date_registered",
                             "claim_date_occured"]
    for i in YYYYMMDD_date_columns:
        df[i] = pd.to_datetime(df[i], format="%Y%m%d")

    # remove extreme value
    df["claim_vehicle_date_inuse"].replace(to_replace=270505.0, value=np.nan,
                                           inplace=True)

    YYYYMM_columns = ["claim_vehicle_date_inuse",
                      "policy_date_start",
                      "policy_date_next_expiry",
                      "policy_date_last_renewed"]
    for i in YYYYMM_columns:
        df[i] = pd.to_datetime(df[i], format="%Y%m")

    # Add new features first, then drop all unneeded columns
    # New feature: add a new column with the number of claims for this vehicle (claim_vehicle_id)
    df['claim_vehicle_id_count'] = df.groupby(by='claim_vehicle_id')['fraud'].transform('count')
    df['policy_holder_id_count'] = df.groupby(by='policy_holder_id')['fraud'].transform('count')
    df['driver_id_count'] = df.groupby(by='driver_id')['fraud'].transform('count')
    df['driver_vehicle_id_count'] = df.groupby(by='driver_vehicle_id')['fraud'].transform('count')
    df['third_party_1_id_count'] = df.groupby(by='third_party_1_id')['fraud'].transform('count')
    df['third_party_1_vehicle_id_count'] = df.groupby(by='third_party_1_vehicle_id')['fraud'].transform('count')

    # dropped for now but can be added for futher improvement
    drop_temp = [
        "claim_postal_code",  # rural area
        "claim_vehicle_brand",  # premium or not
        "claim_time_occured",
        # could derive morning, noon, afternoon, evening, night or day/night
        "claim_vehicle_cyl",
        "claim_vehicle_load",
        "claim_vehicle_fuel_type",
        "claim_vehicle_power",  # buckets?
        "policy_holder_postal_code",  # rural area
        "policy_holder_year_birth",  # age
        "third_party_1_postal_code",
        "third_party_1_injured",
        "third_party_1_vehicle_type",
        "third_party_1_form",
        "third_party_1_year_birth",
        "third_party_1_country",
        "repair_postal_code",
        "repair_form",
        "repair_year_birth",
        "repair_country",
        "repair_sla",
        "policy_coverage_type",
        "driver_postal_code",
        "driver_form",
        "driver_year_birth",
        "driver_country",
        "driver_injured",
        "claim_amount"
    ]

    # date variables are also dropped for first analysis
    drop_dates = ["claim_date_registered",
                  "claim_date_occured",
                  # could be interesting to calculate time between occured and registered
                  "claim_vehicle_date_inuse",
                  "policy_date_start",
                  "policy_date_next_expiry",
                  "policy_date_last_renewed"
                  ]

    # all ID variables are dropped
    drop_forever = ["claim_vehicle_id",
                    "claim_id",
                    "policy_holder_id",
                    "policy_holder_expert_id",
                    "driver_id",
                    "driver_expert_id",
                    "driver_vehicle_id",
                    "third_party_1_id",
                    "third_party_1_vehicle_id",
                    "third_party_1_expert_id",
                    "repair_id"
                    ]
    df.drop(columns=drop_temp + drop_dates + drop_forever, inplace=True)

    # exclude columns with over 50k missing values (i.e. over 90%)
    missing_over_90prct = df.isna().sum().index[np.where(df.isna().sum() > 50000)]
    df.drop(columns=missing_over_90prct, inplace=True)

    return df
