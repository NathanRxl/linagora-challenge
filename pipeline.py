import pandas as pd
from time import time

from general_tools import split_xy, true_recipients, Submissioner
from linagora_models import (FrequencyPredictor, LinagoraWinningPredictor,
    LinagoraTfidfPredictor, LinagoraKnnPredictor
)
from model_evaluation import metrics

initial_time = time()

print("Pipeline script", end="\n\n")

path_to_data = "data/"
compute_training_score = True
use_cooccurrences = False
precomputed_train_cooc = path_to_data + "co_occurrences_train_pipeline.json"
precomputed_test_cooc = path_to_data + "co_occurrences_test_pipeline.json"

print("\tLoad preprocessed train data ... ", end="", flush=True)
# Load training data
train_df = (
    # pd.read_csv(path_to_data + 'preprocessed_train.csv',
    #             index_col="mid", parse_dates=["date"])
    pd.read_csv(path_to_data + 'text_preprocessed_train.csv', index_col="mid")
)

# Split training data into X_train and y_train
X_train, y_train = split_xy(train_df)
print("OK")

# Initiate the model
model = LinagoraWinningPredictor()
# model = LinagoraKnnPredictor()

print("\tFit the model to the train data ... ")
# Fit the model with the training data
model.fit(X_train, y_train)

# # Compute the training score
# print("\tTraining score: ", end="", flush=True)
# y_predict_train = model.predict(
#     X_train,
#     y_true=y_train,
#     use_cooccurences=False
# )
# true_mids_prediction = true_recipients(y_train)
# train_score = metrics.mean_average_precision(
#     y_predict_train,
#     true_mids_prediction
# )
# print(round(train_score, 5), end="\n\n")

print("\tLoad preprocessed test data ... ", end="", flush=True)
# Load test data
# X_test = pd.read_csv(path_to_data + 'preprocessed_test.csv',
#                      index_col="mid", parse_dates=["date"])
X_test = pd.read_csv(path_to_data + 'text_preprocessed_test.csv', index_col="mid")
print("OK")

print("\tMake predictions on test data ... ", end="", flush=True)
# Predict the labels of X_test
y_pred = model.predict(
    X_test,
    use_cooccurrences=False,
    # precomputed_cooccurrences=precomputed_cooccurrences,
    y_true=None,
    store_scores=False
)
print("OK", end="\n\n")


print(
    "\tCreate Kaggle submission in submissions/ folder ... ",
    end="",
    flush=True
)
# Create Kaggle submission
submission_folder_path = "submissions/"

Submissioner.create_submission(
    y_pred,
    output_filename="predictions_knn_features.txt"
)
print("OK")


print("\nPipeline script completed in %0.2f seconds" % (time() - initial_time))
