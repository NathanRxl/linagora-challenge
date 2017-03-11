from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime

from model_evaluation import metrics
from general_tools import split_xy, true_recipients
from linagora_models import FrequencyPredictor, LinagoraWinningPredictor


initial_time = time()

print("Validation script", end="\n\n")

path_to_data = "data/"

# Load data
complete_train_df = pd.read_csv(path_to_data + 'preprocessed_train.csv',
                                index_col="mid", parse_dates=["date"])
complete_train_index = complete_train_df.index.tolist()
complete_train_df = complete_train_df.sort_values(by="date")

# Use temporal folds
# Number of days between the oldest and the most recent message in test
TIME_PERIOD_TEST = datetime.timedelta(days=234 / 2)
TRAIN_TEST_OVERLAP = datetime.timedelta(days=127 / 2)
NB_MESSAGES_TO_PREDICT_IN_TEST = 2362
TRAIN_MOST_RECENT_DATE = complete_train_df['date'].max()

print("Start validation ... ", end="", flush=True)

train_df = complete_train_df[
    complete_train_df["date"]
    < TRAIN_MOST_RECENT_DATE - TIME_PERIOD_TEST - TRAIN_TEST_OVERLAP / 2
]

overlap_df = complete_train_df[
    complete_train_df["date"]
    <= TRAIN_MOST_RECENT_DATE - TIME_PERIOD_TEST + TRAIN_TEST_OVERLAP / 2
]

overlap_df = overlap_df[
    overlap_df["date"]
    >= TRAIN_MOST_RECENT_DATE - TIME_PERIOD_TEST - TRAIN_TEST_OVERLAP / 2
]

validation_df = complete_train_df[
    complete_train_df["date"]
    > TRAIN_MOST_RECENT_DATE - TIME_PERIOD_TEST + TRAIN_TEST_OVERLAP / 2
]

validation_scores = list()
# Select a sample of test fold df to be in a situation close to test

overlap_train_df, overlap_test_df = train_test_split(
    overlap_df,
    test_size=0.94,
    random_state=2
)

train_fold_df = pd.concat([train_df, overlap_train_df])
validation_fold_df = pd.concat([validation_df, overlap_test_df])

# Assure that there is the same number of senders in train and in test
unique_train_senders = train_fold_df["sender"].unique().tolist()
validation_fold_df = validation_fold_df[validation_fold_df["sender"].isin(unique_train_senders)]

validation_fold_df = validation_fold_df.sample(
    n=NB_MESSAGES_TO_PREDICT_IN_TEST,
    random_state=2
)

X_train, y_train = split_xy(train_fold_df)

# Fit model and make predictions
model = LinagoraWinningPredictor(recency=[20], non_recipients=1.0)
model.fit(X_train, y_train)

X_test, y_test = split_xy(validation_fold_df)

y_predict = model.predict(
    X_test,
    y_true=None,
    store_scores=False,
)

# Compute true mids_prediction dict from df
y_test_true_recipients = true_recipients(y_test)

# Compute score
validation_fold_score = metrics.mean_average_precision(
    mids_prediction=y_predict,
    mids_true_recipients=y_test_true_recipients
)

print("\nValidation score :", validation_fold_score)

print("\nValidation script completed in %0.2f seconds"
      % (time() - initial_time))
