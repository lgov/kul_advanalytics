import pandas as pd
import numpy as np


# cleaning functions
from sklearn.preprocessing import OneHotEncoder


def text_to_binary(col_name, bin_1, bin_0, df):
    return df[col_name].replace({bin_0:0,bin_1:1}, inplace=True)


def convert_pct_bitmap_to_features(df):
    """
    Convert the bitmap in policy_coverage_type to individual features
    e.g. #111110000 becomes pct1 pct2 pct3 pct4 pct5 pct6 pct7 pct8 pct9
                               1    1    1    1    1    0    0    0    0
    :param df: a dataframe
    :return: dataframe with new features
    """
    # split text of form #111110000 into separate columns and drop redundant columns
    bit_df = df['policy_coverage_type'].str.split('', expand=True).drop(columns=[0,1,11])
    # rename into wished columns
    bit_df.columns = ["pct1", "pct2", "pct3", "pct4", "pct5", "pct6", "pct7", "pct8", "pct9"]
    # merge with original dataframe (can be improved because merge is slow)
    df = df.drop(columns=['policy_coverage_type']).merge(bit_df, left_index=True, right_index=True)
    return df


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
    df['blacklisted_expert_id'] = [((is_fraudulent_expert_id(i)) or (is_fraudulent_expert_id(j)))for i, j in zip(df["driver_expert_id"],df["policy_holder_expert_id"])]

    # Look up the involved repair shops in our blacklist-of-fraudulent-experts
    df['blacklisted_repair_id'] = [is_fraudulent_repair_id(i) for i in df["repair_id"]]


# Seems to have a negative effect on our score, removing.
#    # Calculate age of policy_holder at time of accident
#    df['policy_holder_age'] = df["claim_date_occured"].dt.year - df["policy_holder_year_birth"]

    # Split the policy_coverage_type bitmap in individual features.
    # e.g. #111110000 becomes pct1 pct2 pct3 pct4 pct5 pct6 pct7 pct8 pct9
    #                            1    1    1    1    1    0    0    0    0
    df = convert_pct_bitmap_to_features(df)

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
#    df["claim_amount"] = pd.to_numeric(df["claim_amount"].str.replace(",", "."))

    # create a one hot encoder to create the dummies and fit it to the data
    # Note: use this method, so that we can use exactly the same one hot encoding
    # also on the submission data
    claim_cause_ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    claim_cause_ohe.fit(df[['claim_cause']])
    df = encode_claim_cause(claim_cause_ohe, df)

    # create a one hot encoder to create the dummies and fit it to the data
    # Note: use this method, so that we can use exactly the same one hot encoding
    # also on the submission data
    repair_form_ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    repair_form_ohe.fit(df[['repair_form']])
    df = encode_repair_form(repair_form_ohe, df)

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
#        "repair_form",
        "repair_year_birth",
        "repair_country",
        "repair_sla",
#        "policy_coverage_type",
        "driver_postal_code",
        "driver_form",
        "driver_year_birth",
        "driver_country",
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
                    "repair_id",
                    "claim_amount"   # Remove again when we have the regression model.
                    ]
    df.drop(columns=drop_temp + drop_dates + drop_forever, inplace=True)

    # exclude columns with over 50k missing values (i.e. over 90%)
    missing_over_90prct = df.isna().sum().index[np.where(df.isna().sum() > 50000)]
    df.drop(columns=missing_over_90prct, inplace=True)

    return df, claim_cause_ohe, repair_form_ohe


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


def encode_repair_form(repair_form_ohe, df):
    """
    Use a fitted OneHotEncoder for the repair_form column and replace it with the
    encode columns
    To be fitted on the training set and applied both on training set and submission set.
    :param repair_form_ohe:
    :param df:
    :return: df
    """
    trf = repair_form_ohe.transform(df[['repair_form']])
    nr_of_cols = len(trf[0])
    col_names = ['rf%d' % i for i in range(1, nr_of_cols + 1)]
    df2 = pd.DataFrame(trf, columns=col_names, index=df.index)
    df = pd.concat([df, df2], axis='columns')
    del df['repair_form']
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


def is_fraudulent_repair_id(repair_id):
    """
    This function maintains a list of repair_id's that are known to be involved in
    fraudulent claims. This can be used as knowledge extracted manually from the
    training set. TODO: how can we learn this?
    :param repair_id: <string> id of the repair shop
    :return: True if known involved in fraudulent cases, False if not
    """
    criminal_repair_ids = [
    "M2FiMDA0ZTNiOWRhNzFkOTc0NGM2YmZlOTFmZDFlNGI",
    "M2FmNDRkNjE4YTZlYmU2NmQyZjVlZTY1NDU3MTUyNTU",
    "M2RmOGE4MjAwMTJjMDlmZDczNDQ3YWI0OTFhMGMxNWM",
    "MDBjNWZlOThkN2Y1MTM5NjMzZjVmYjlhOTVlNGNhOGE",
    "MDE0NjI2ZjUwMmNlOTU4ZTM0ODI3NjVhNTVjZWYyOWY",
    "MDg4YzM3ZmQ1Yjc1NzI5YmI4ZmY1MjJmYjRmZTgxZDU",
    "MDhjYTAwZjQ0ZjhjMjMyYTU1ZDY4MjJmZjU1NDg5ZDc",
    "MDIxY2Q0NmM2NDg4NzdhYTY2NjhmMjI3ZjYxYzE4ODg",
    "MDM5NzQxNWQxMGQ2NGY4NTYyNGM3M2Y3MTIxYWIzZDc",
    "MDYzNWVjODY1Yjk5OWM5MWE0OTIyODVhOWIwNmRhMDI",
    "MGM1OWRkOGJhNzA2OGEwYjFmNDQ4NWY2OTFjMWU4Yjk",
    "MGVlNjM4MjJmYWIzYjI3ZTZiYzdmZmY5YjY5N2U1MGM",
    "MjM1Y2Q4MTBmODZmOTYyNmExNGFjNWViN2MxNjFlYjA",
    "MjMzOTEzZDQxN2FkYmFmYmQyYjI2ZDY2NjI2MGI5OGM",
    "MjNjOTViOTc4OWUyNTZmN2FhMjRmMWFmYjZjZjFjZTU",
    "MjNjY2RmMDUxZGNjZWI2OWU4NWYwOTliYzNlMDZlZjk",
    "MmMxNzcwNzI5M2FjNmNhZmU4ZjUwOWE3NGQyOWMwMzQ",
    "MmNkZGEyOTk2M2IwZGY4ODZlNTdmODg1NDMwZTdhY2E",
    "MTc0MmI2MjNjZTk4OTZkZjVkNTE3ZGM4NzdiNjMwM2U",
    "MTc4MzhkZTI0NDgzNjY0OGEyZGRlNWY3ZjkwMGVkMTA",
    "MTJmMDA3ZGVjMDM3NGYyZGYxZDBiNzAxZjFmZjliY2U",
    "MTMwNDI4OTE2MzZhMDk5Y2FkODc5ODVkMTVhNDIxN2E",
    "MTU3Mzc5ZWE3ZmE2Mjg3OWU5YmMwNTNmNGI4OGE3ZDE",
    "MTVkOGVmMDEzMDAxZWE0YjQxZjJkMjZmMmU3MThjMWQ",
    "MWU1NWM1N2UyODRhYTI4MjUyYjAwNDVmYTBlNWQwYzQ",
    "MWUzMTVmNTdiY2NjZTY5NzE3ZDNmM2IwNWFlZWNiZDM",
    "Mzc2NTUxM2VjMzAzMTUyMGE3OTdkNjdkNDM5MDcxNzk",
    "MzE3NjMzZDgyZDc2MTIzZGYwYzYyOWQwM2QzZjE0MTc",
    "Mzg4YTMxOTJjYzkxOWRmMjU1YWRkZTU2ZjllY2ZmN2Q",
    "Mzg4ZjQwNTIxZWVmYWQxMmM1Njc5NmY2ZTg5NTEyN2Y",
    "N2EyMTdkYzY4Mzc3YmEzOWY1MGYwMjMyYzVmMzdhZWQ",
    "NDc4ZDliOTkzNjFiMDE2YzRkMGE5YzE1NGUxNGU2NmI",
    "NDdhOGMyN2Q0MjM1NTY3MzZkZTM5MTExYzk4YzAyMDM",
    "NDhiZDZmNWIwY2U2NjdjMzVjYmEwMzU2MGE1YzdhOTM",
    "NDM5ZjhmYjI0MDg3MzkyNmEyMjgyMGQ5NjU4MDM0MGI",
    "NDNiMzFlMGZkMWZhOGQ0MjI2NTEyNzI2ZDc0YjY3ZmI",
    "NDRhYWMwYWQ1Njg1M2JiNTE2ZmVmZjA5OGE5MjRhNjE",
    "NDY1N2EzNjllYWY1MWVjYzMwYzBjNzQ2ODI1MGM5YzM",
    "NGFkNzc2OTE4ZjZjMTc4YWNjYzI5ZTY2NGQxZjQ4ZmU",
    "NGIzYWQ5YjQ0NzIxODliNDk0MjMxOGY0MTk2YjZhMjY",
    "NGRiOTljYzI3YzNmZGNkNTY1MDM2MGIxZWNmZWY4YzU",
    "NGRjZjQ0NjMxOWI5ZWNhNTlhNWJhYjc4MmQ5MjFkNzI",
    "NGYxNmFlNzUyMTdhZjg1ZjYzMzU4MTRmMzgwODVkNmQ",
    "NGYyYWY3Yzk0ZTRjYmVhNjg0MzQ5MDAwYzJmNDQ2YTU",
    "NjA1NDhhMDhiOGI3NTYyYjNhYzRhZmYxNzg1ZWI3YjM",
    "NjBkYmU2YjY5NzUwNjgyMzAzY2QyYmQyYTdlZWYzYTk",
    "NjBlMDI3ZGYzYzRmMDQ1OTM1MjY4YmU4YjM5OGY0NzE",
    "Njc5YWUyYTEzMjNhNGEyNTJjM2M1M2RlOGViOTlkM2E",
    "NjM0NGVlM2ZhZGQ3ZDQ5ODUzYWU4MDNhOGUxZDdmZTE",
    "NjNhZWIxMTQ0OTJjZTg0NzEzZjBlOWM2NmU2ZjFjNDQ",
    "NjQzODYzYjI2OGM0MGI4ZTBlMmUyMzM0ODU5MTNkYjY",
    "NmE0ZmY1ODJkYzZlNjMwOWY2MjcyZjFmMWVmNzIzMjM",
    "NmU0N2NmNDc3OTgzZDEzMzIxOWQ1NWI2ZTIyNTMxNzY",
    "NmVlOGE3YmU4YWQ3MWUyZjE4ZjcwN2NiNzUxMThhY2I",
    "NTBhODM5NTZlNTljYWI1MWRlNGJkYzliOGUzODM4MTY",
    "NTBiNmMyY2UxZTBkMjY3YjI3YzQ0NTM3NmI2ZTMwOTI",
    "NTc1ZWNjYzE5ZmY4ZmRlZjhjZTRjMGVhMDJhYjM0NjM",
    "NTdjYzgwMzcyMzllZTczNzNmOGE0N2U2YTA3OWFlZDg",
    "NTY3YmY0ODMzMmNiMzdlNTNkMzM1YzM0YTg5NzhiOTY",
    "NTZlNjFhMjY3NjkyMGJjNGEwMzVhOTQyZjI2MzBlMzg",
    "NWFkMDc1YzAyYzAyMTdjYmFmYWIyMjU2NzU1NDZjZTA",
    "NWRlYmE5MTI5YzhiYTc3N2I1YmVlNTUwYmU2MmZmYzE",
    "NWUyN2FiNDVjMzIxYjQyMWI3M2I0NmU0MmU1ODcwYzk",
    "NWVkNjQ3NGRjYWMwYTczNmFhYzVhMTY1OTZkMGJiMTE",
    "NWYyOGQ3ZTBmMmIwNDZkNGM4NGVjMTY5ZWVhMDcxOTk",
    "NzcyOTEyNzY4Y2E0YjUzOWE4NzllN2UwZTFmNDdiNmM",
    "NzdlZGI4ODZiY2M0YzI5N2RjY2EwNDQ3YjdkNzNmMzc",
    "NzE4ZGFlMjVlNDU3YjhmNjQyZDlkNTIwN2RlNTJmNmM",
    "NzQyNGQxNjdiMjljNGFiNDg2N2JjOTQ2YzYzODhkMWM",
    "ODhiY2JhNjZmNDRjNjY3MmE2MDBjM2E0OTExN2ZiOTk",
    "ODY2NmIxNWZkMGY0MTUwYzFkYjBkZDU1Y2EzMDI1OTQ",
    "OGQ3NjA3MjE1N2IzZjA0MjI0YjgwYTYzMmI4MDRhZjI",
    "OTcyNDMxZTNmZGNkZjI0ZWVkMGVlZDhjMDQ4NmY2Mjk",
    "OTFiODBhN2NkODI1NjAzYTdmN2JhNDY3OTQ4MWMwOTc",
    "OThjYjU3MDFiODVhNjI1NzdlMDc0NDdjZjg1N2Q0NTM",
    "OTJmNGUxZTQ2Zjg0YzBhMzRmY2VhZTk5MGI5ZGE1Mjc",
    "OTU5OWJkYTY0ZjAxOWRjZjJkYTNlYzVhZGQ2ZjdkZTQ",
    "OWIwNDVhMDFjYmQ3Y2ZkMWYyNGQ3M2EzMDQ5MzE4ZmQ",
    "OWNiOWZjZWZmNTQzNWIwNjFhYjczYzkyYmQxYzBiMWQ",
    "OWQwNjA3Mzk5Zjk4ZmIxZTRjNTNjNTIzYTgwMzk4Mjg",
    "OWZlODZjOWRkNDdmMjY4ZTBmODViMzZmODFlZjFlNjU",
    "Y2ExZDU2NGM0NWQxMWY3YzYwZjFhMGUxMGQ3ZTBmZGY",
    "Y2Q2MDMyZGI2ZTcwODU3MTYzMjVkMDI1MzY5YWQ4ZDg",
    "Y2RlOGMwZGM2M2I0Y2YwNmEzNGE3MTQ1NjAwNDMwMDg",
    "YjAyZjEwMjEwNDIzNGU1YzBkNjNmY2E0MDhhZDEyODM",
    "YjdlZmI1NTI0MDg2ZWMwM2UzMzBkZTFlNzI0ZmY2MWM",
    "YjE2NjhiYWRmZmY0Njg3ZjIxNWQyN2VhOTBhODUzNzE",
    "YjM0MjkyOTBhNThjMTI0ZGEzM2VlNWU0OTE4Njk3ZTI",
    "YjQ3NTg4OTgwZjIxMmU3Zjg2YWJkOGE3OWY0Yzc2NWI",
    "YjRkMTI2NzY1ODdkMjMwY2VkYTgyOGFiMTQ1NDljZGI",
    "YmE1NTVjY2Y0ZDg1ZmU3NDA4Zjg3YjI4Zjg2ZDc2OGY",
    "YmI2YTJkOWFhZGNlM2FmN2ZmMDk0YTA1YmU5MjE2ZDk",
    "YmNjOTM1NjcwNDE3YWI0MDA3OWUzYjEzMTI1YTNjMjE",
    "YTI2OWY1ZTgwZDFkNjNlZTc2ZTE4NTlmZTgzY2Y1NTE",
    "YTJmNTkxZDM4OThjNGJhMWE2OTI2NjExZTA0MjlkYTQ",
    "YTUxMmY0NWUyODJjOGQ3MWJjMzZmODM0MTBlMThlZDk",
    "YWE0ZGU3OTZkNDdiMmUxNWIzNTJiNGE1NWM1OTMyMjI",
    "YWI2NjMxZWI0MzE3MDZkODgyY2RjOGEyYTg3ZmVkYjY",
    "YWJiOTM1YThjYWNlYWQ3ZGQ4ODJmNTRjMjBhM2FlODY",
    "YWY1MjM0M2YyMWFmZGZjYWVjNWM3YWM1YzRhZmVhNDQ",
    "YzFhYWViYmI4YWViOWZhNWJkOWQ0YWNjNTk0OTZjOTA",
    "YzZiNDVhZDM5MDcxYjQxYzVmMTU4YzZmNzgxM2U4OGY",
    "ZDA5NWQzNjVlOTdkYzI3NTQyYjQ0MGU1NTkyZjQ2Zjc",
    "ZDBjNGM2NWM2MzMzMzliMTlkMjQzYjk3YjdjNjEyYmU",
    "ZDczMjM5YjMzMzIyNGVmOThhNzI4Nzg5MzQ3ODlmZmU",
    "ZDE0N2FjOWE0M2Q4ZWJjOWFhMWQ2MWIxOTY4NGZjNjU",
    "ZDQ3NDEzYTFjMTRhMzllMmNkOWE3NTgyMzNlMThlNTM",
    "ZDVkYmUxZjhhZmEyNmJkYzliODAwM2RiY2ZkY2QyZTg",
    "ZDY1NTJmNzZkODI1MDgxZWI0MWYzMzhkM2YyYTQxNjA",
    "ZGI0N2MxODBkMjMxYWQyZjQ2NGE3Yjc2OWYwNDY2Mjg",
    "ZGIwYzVkYzA3MTk4OTdkYzU0OGRmMDZiNTRkYzdkZGU",
    "ZGQwNjUxNTg3NjdlNTYzZGVlYzViYjMxODY0NDNmYzc",
    "ZGUxY2Q2MTcxYzgyNjAwZmM0Zjk5N2UyZGZjZjAyZDQ",
    "ZGUzODllYTY0YTRkZjQ4N2FjNjE5ZWZhYTZlMzE5ZjU",
    "ZGVhNTQ0OTQxYTdiYWRhNzlkMjA1ZmVjNjFiOGRhNzQ",
    "ZjdhZjQ3NDIxMTJjOWVjODBhZjk4OTU5OGQwYTczZDA",
    "ZjE1MDM5OWMzZjJlZGFiODY1ODM5NGNiZTNmMWY2OTM",
    "ZjNhNjlkZTY2ZGNkOTJhYmEyNTBiYTI0OGRiMjg1NmE",
    "ZmEyYWMxOTM2NzVlYzFjZWE3M2UwZGY3MjQwYjU5OTI"]

    return repair_id in criminal_repair_ids