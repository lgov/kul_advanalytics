from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the training data, split in train an validation dataset.
# -------------------------------------------------------------
df = pd.read_csv('train.csv', sep=";", encoding="utf-8-sig")
df = df.drop(columns=["claim_amount"])
# claim_amount is currently dropped since poor performance
train_df,val_df = train_test_split(df, test_size=0.2,  random_state=96)

train_data = TabularDataset(train_df)
val_data = TabularDataset(val_df)

# Run the tabular predictor, use the F1 score as evaluation metric.
# -----------------------------------------------------------------
# https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-quickstart.html#maximizing-predictive-performance
# This gives a very bad result, don't use it as is.
metric = 'f1_macro'
# auto_stack: let AutoGluon find the best bagging/stacking parameters
predictor = TabularPredictor(label='fraud', eval_metric=metric).fit(train_data, time_limit=3600, presets='best_quality',
                                                num_bag_folds=5, num_bag_sets=1, num_stack_levels=1)  # Fit models for 3600s
leaderboard = predictor.leaderboard(val_data)
print(leaderboard)

# Use the best model to predict on the test set and prepare the output for submission.
# -------------------------------------------------------------------------------------
submission = pd.read_csv('test.csv', sep=";", encoding="utf-8-sig")
submission["prediction"] = predictor.predict_proba(submission)['Y']
submit_pred = submission.copy()
# final submission set initialization
submission = submission.reset_index()[['claim_id']]
submission["prediction"] = submit_pred['prediction']
submission.columns = ["ID", "PROB"]
print(submission)
submission.to_csv("submission_V0.78.csv", sep=',', index=False)
