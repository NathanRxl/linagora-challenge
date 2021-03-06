import pandas as pd
import numpy as np
from time import time

import tools
from linagora_models import LinagoraWinningPredictor, LinagoraKnnPredictor

initial_time = time()
np.random.seed(1)

print("Pipeline script", end="\n\n")

path_to_data = "data/"
compute_training_score = True
use_cooccurrences = False
precomputed_train_cooc = path_to_data + "co_occurrences_train_pipeline.json"
precomputed_test_cooc = path_to_data + "co_occurrences_test_pipeline.json"

print("\tLoad preprocessed train data ... ", end="", flush=True)
# Load training data
train_df = (
    pd.read_csv(path_to_data + 'text_preprocessed_train.csv', index_col="mid")
)

# Split training data into X_train and y_train
X_train, y_train = tools.split_xy(train_df)
print("OK")

# Initiate the model
model = LinagoraKnnPredictor()

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
X_test = pd.read_csv(
    path_to_data + 'text_preprocessed_test.csv',
    index_col="mid"
)
print("OK")

print("\tMake predictions on test data ... ", end="", flush=True)
y_pred = model.predict(X_test)
print("OK", end="\n\n")


print(
    "\tCreate Kaggle submission in submissions/ folder ... ",
    end="",
    flush=True
)
# Create Kaggle submission
submission_folder_path = "submissions/"

tools.create_submission(
    y_pred,
    output_filename="best_submission.txt"
)
print("OK")


print("\nPipeline script completed in %0.2f seconds" % (time() - initial_time))
