import pandas as pd
import numpy as np


# cleaning functions
from sklearn.preprocessing import OneHotEncoder


def text_to_binary(col_name, bin_1, bin_0, df):
    return df[col_name].replace({bin_0:0,bin_1:1}, inplace=True)


def add_extra_features(df):
    # Add new features first, then drop all unneeded columns
    # New feature: add a new column with the number of claims for this vehicle (claim_vehicle_id)
    df['claim_vehicle_id_count'] = df.groupby(by='claim_vehicle_id')['claim_id'].transform('count')
    df['policy_holder_id_count'] = df.groupby(by='policy_holder_id')['claim_id'].transform('count')
    df['driver_id_count'] = df.groupby(by='driver_id')['claim_id'].transform('count')
    df['driver_vehicle_id_count'] = df.groupby(by='driver_vehicle_id')['claim_id'].transform('count')
    df['third_party_1_id_count'] = df.groupby(by='third_party_1_id')['claim_id'].transform('count')
    df['third_party_1_vehicle_id_count'] = df.groupby(by='third_party_1_vehicle_id')['claim_id'].transform('count')

    df['claim_vehicle_id_count'].fillna(0, inplace=True)
    df['policy_holder_id_count'].fillna(0, inplace=True)
    df['driver_id_count'].fillna(0, inplace=True)
    df['driver_vehicle_id_count'].fillna(0, inplace=True)
    df['third_party_1_id_count'].fillna(0, inplace=True)
    df['third_party_1_vehicle_id_count'].fillna(0, inplace=True)

    # Look up the involved experts in our blacklist-of-fraudulent-experts
    df['blacklisted_expert_id'] = df.apply(lambda x:is_fraudulent_expert_id(x.driver_expert_id) or \
              is_fraudulent_expert_id(x.policy_holder_expert_id), axis=1)

    # Convert policy_type to dummy variables

    return df

def update_dataset_features(df):
    """
    Add, remove and encode features from the insurance claim dataset for modelling.

    :param df: dataframe with the original data
    :return: df: dataframe with new columns
             claim_cause_ohe: one hot encoder object for the claim_cause column
    """
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

    # create a one hot encoder to create the dummies and fit it to the data
    # Note: use this method, so that we can use exactly the same one hot encoding
    # also on the submission data
    claim_cause_ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    claim_cause_ohe.fit(df[['claim_cause']])
    df = encode_claim_cause(claim_cause_ohe, df)

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

    df = add_extra_features(df)

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

    return df, claim_cause_ohe


def encode_claim_cause(claim_cause_ohe, df):
    """
    Use a fitted OneHotEncoder for the claim_cause column and replace it with the
    encode columns
    To be fitted on the training set and applied both on training set and submission set.
    :param claim_cause_ohe:
    :param df:
    :return: df
    """
    trf = claim_cause_ohe.transform(df[['claim_cause']])
    nr_of_cols = len(trf[0])
    col_names = ['cc%d' % i for i in range(1, nr_of_cols + 1)]
    df2 = pd.DataFrame(trf, columns=col_names, index=df.index)
    df = pd.concat([df, df2], axis='columns')
    del df['claim_cause']
    return df


# def encode_ph_postal_code(ohe, df):
#     """
#     Use a fitted OneHotEncoder for the policy_holder_postal_code column and replace it with the
#     encode columns
#     To be fitted on the training set and applied both on training set and submission set.
#     :param ohe:
#     :param df:
#     :return: df
#     """
#     trf = ohe.transform(df['policy_holder_postal_code'].astype(str).str[:2])
#     nr_of_cols = len(trf[0])
#     col_names = ['phpc%d' % i for i in range(1, nr_of_cols + 1)]
#     df2 = pd.DataFrame(trf, columns=col_names, index=df.index)
#     df = pd.concat([df, df2], axis='columns')
#     del df['policy_holder_postal_code']
#     return df


def is_fraudulent_expert_id(expert_id):
    """
    This function maintains a list of expert_id's that are known to be involved in
    fraudulent claims. This can be used as knowledge extracted manually from the
    training set. TODO: how can we learn this?
    :param expert_id: <string> id of the expert
    :return: True if known involved in fraudulent cases, False if not
    """
    criminal_expert_ids = ["M2JkNDRlMTBiNDYzMDZmOTIyOWUyMTViM2YwMmUyMTk",
    "M2NmODVkY2JkNDk4ZWRmYTRmOTM0NzNmMTUyZGE5Mjc",
    "MjUyMmQ2ZTM2OTdlZWU5NzU3NjJhMTA1MGNmZDc2MDE",
    "MmUzNGQxYTVjMGFiMDQ4OTY4MDZiOGY1NjU1NGExOTI",
    "MmZkMTE1N2FhODdiYTI1ZDNlNzUxMTNhMDYwNTE1Yjk",
    "MWJmNzg1ZmYxMjEyNzQ5ZTY5NDhhN2I2ZTA2ZWI3YWY",
    "MWNlYzcyZDk3OWM5N2ZjZTFjYWNmZDQwZmY2MWU2OGE",
    "NDUzN2I1YmYwYTRhM2FjZGVkMjdiMzg0N2MwOTdhNmI",
    "NGE5OGQ4ZDc1NjFjNjg1NTIxZmRjNzI5N2YwY2FhY2M",
    "NGYyM2U1ZmIyY2MxYWJmZWQ3OWZmOTRiYTZhZTEwNDQ",
    "Njc0YWNlNzQyYjBiMDAwYzg4ZjAyMjE1MGUwY2EyNWM",
    "NjYzNzBjODhlNGZiYjU0NzE0ZTQ5NjFmYTNhNjFjMzQ",
    "NTUxNDM2MzdkMGZiZmYzYTcxYTEyNGNjNmJjNjJhYzE",
    "ODFmNzhlNDY2MmU3YTk0NWZkNjQwYzU4N2EzZjQ0Yzg",
    "OGE3ZTgyMTFhZDQzMGJlZDg2MWVjZTRkZTA5M2JlNTA",
    "OGEwMzgzNzBiODVhZDRmZTUyMWFhZjkzMzU0NzdkZTg",
    "OGIzNzgyMmQzYWEyNGViMDNlYzRkMzAxODcxMDdhOTI",
    "OTliOTAzYWU3YzljYzdkYmVmNzNmYWMyZjNjNzQ5OWE",
    "OWJiNmVkNjM5ZDFmOWY2YWMxNGQxZWQ1NDdlNDU0YWM",
    "OWY4MWE5MGMzNmRkM2M2MGZmMDZmM2M4MDBhZTRjMWI",
    "Y2ZkYTMzYWM1ZjQ0ZGM1M2Q2NGQ4ZTBlZWNmMzRhYzg",
    "YjJjM2RhYjc5N2U1MmNhMzI4MmI0ZmRiMTk5MDcxMjI",
    "YmMyZDQ4YmIwYWNlYjE1ZTAwYTIyMzUxYWFiMzk4NjM",
    "YmNkZjNiZTQ1MmYzNGQ5YTRmNjI0N2U1YzdiOWUzYzE",
    "YTQyNDg4MGU1MDgzYmNkNzZlZDkzYjVhODJmYmMyNmY",
    "Yzc3Y2E0ZmRkZmMzZjMwNTYyNDRiNjRkYTZiYzMwMjE",
    "YzhhMzJmM2RiNzdiYWE3NjI0NDU3Mjc1ZTNlYzJkOTU",
    "ZDEyMDViM2NkN2I2N2ZmMWZmZjYwN2UxYjMxNzNhOGM",
    "ZDU4ZDExMTNhYmM4YmNlY2QzZmNjOWJjMDRjMGNmZmQ",
    "ZmRlZTE4NDVmOTg1YTc0MGViYjZhOGZhZGFiNmI2OWE",
    "ZTZmNmNmMDkxZDM1MjFlYjdmNWE0YzRiY2Q0NGFkMzU",
    "ZWI4YTBjYzMyOTQ3OTJiYzZhNmRiNjBmNjk5NjdjMWM"]

    return expert_id in criminal_expert_ids