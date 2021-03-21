import logging
import unittest
import pandas as pd
import numpy as np

from assignment1.features import update_dataset_features


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s')  # include timestamp

    def test_add_count_of_claim_id(self):
        df = pd.read_csv("unittest.csv", sep=";", encoding='utf-8')
        print(df['claim_vehicle_id'])
        df = update_dataset_features(df)

        # The first row is duplicated, so we expect 2's for the counts
        first_row = df.iloc[0]
        self.assertEqual(2, first_row['claim_vehicle_id_count'])
        self.assertEqual(2, first_row['policy_holder_id_count'])
        self.assertEqual(2, first_row['driver_id_count'])
        self.assertEqual(2, first_row['driver_vehicle_id_count'])
        self.assertEqual(2, first_row['third_party_1_id_count'])
        self.assertEqual(2, first_row['third_party_1_vehicle_id_count'])

        # The second isn't duplicated, but has some NaN values for some id's. If an
        # ID is NaN, the corresponding count will be set to 0.
        second_row = df.iloc[1]
        self.assertEqual(0, second_row['claim_vehicle_id_count'])
        self.assertEqual(1, second_row['policy_holder_id_count'])
        self.assertEqual(1, second_row['driver_id_count'])
        self.assertEqual(0, second_row['driver_vehicle_id_count'])
        self.assertEqual(0, second_row['third_party_1_id_count'])
        self.assertEqual(0, second_row['third_party_1_vehicle_id_count'])

        # The fourth row has a blacklisted driver expert id.
        fourth_row = df.iloc[3]
        self.assertTrue(fourth_row['blacklisted_expert_id'])
        pass
