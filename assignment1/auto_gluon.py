from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv', sep=";", encoding="utf-8-sig")
df = df.drop(columns=["claim_amount"])
# claim_amount is currently dropped since poor performance
train_df,val_df = train_test_split(df, test_size=0.2,  random_state=96)

train_data = TabularDataset(train_df)
val_data = TabularDataset(val_df)

# https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-quickstart.html#maximizing-predictive-performance
# This gives a very bad result, don't use it as is.
# metric = 'average_precision'
# auto_stack: let AutoGluon find the best bagging/stacking parameters
predictor = TabularPredictor(label='fraud').fit(train_data, time_limit=3600, presets='best_quality',
                                                auto_stack=True)  # Fit models for 3600s
leaderboard = predictor.leaderboard(val_data)
print(leaderboard)

submission = pd.read_csv('test.csv', sep=";", encoding="utf-8-sig")
submission["prediction"] = predictor.predict_proba(submission)['Y']
submit_pred = submission.copy()
# final submission set initialization
submission = submission.reset_index()[['claim_id']]
submission["prediction"] = submit_pred['prediction']
submission.columns = ["ID", "PROB"]
print(submission)
submission.to_csv("submission_V0.72.csv", sep=',', index=False)
