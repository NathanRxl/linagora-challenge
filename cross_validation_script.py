from time import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from model_evaluation import metrics
from general_tools import split_xy, true_recipients
from linagora_models import FrequencyPredictor, LinagoraWinningPredictor


initial_time = time()
n_splits = 3

print("Cross-validation script", end="\n\n")

path_to_data = "data/"

# Load data
complete_train_df = pd.read_csv(path_to_data + 'preprocessed_train.csv',
                                index_col="mid", parse_dates=["date"])
complete_train_index = complete_train_df.index.tolist()

# Use KFold from sklearn
kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)

cv_scores = []
cv_split_indexes = kf.split(complete_train_index)
for n_fold, (train_fold_index, test_fold_index) in enumerate(cv_split_indexes):
    print(
        "Start working on fold number", n_fold + 1, "... ",
        end="", flush=True
    )
    train_fold_df = complete_train_df.iloc[train_fold_index]
    test_fold_df = complete_train_df.iloc[test_fold_index]

    X_train, y_train = split_xy(train_fold_df)
    X_test, y_test = split_xy(test_fold_df)

    # Fit model and make predictions
    model = LinagoraWinningPredictor(recency=[20], non_recipients=1.0)
    model.fit(X_train, y_train)

    precomputed_cooccurrences = (
        path_to_data + "co_occurrences_cv_{}-3.json".format(n_fold + 1)
    )
    y_predict = model.predict(
        X_test,
        use_cooccurrences=True,
        precomputed_cooccurrences=precomputed_cooccurrences,
        y_true=None,
        store_scores=False
    )

    # Compute true mids_prediction dict from df
    y_test_true_recipients = true_recipients(y_test)

    # Compute score
    fold_score = metrics.mean_average_precision(
        mids_prediction=y_predict,
        mids_true_recipients=y_test_true_recipients
    )
    print(fold_score)
    cv_scores.append(fold_score)


cv_score_mean = round(np.mean(cv_scores), 5)
cv_score_std = round(np.std(cv_scores), 5)
cv_score_min = round(np.min(cv_scores), 5)
cv_score_max = round(np.max(cv_scores), 5)

print("\nMean score :", cv_score_mean)
print("Standard deviation :", cv_score_std)
print("Min score :", cv_score_min)
print("Max score :", cv_score_max)


print("\nCross-validation script completed in %0.2f seconds"
      % (time() - initial_time))
