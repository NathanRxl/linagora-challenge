import pandas as pd
from time import time

from general_tools import split_xy, Submissioner
from models import LinagoraFrequencyPredictor

initial_time = time()

print("Pipeline script", end="\n\n")

path_to_data = "data/"

print("\tLoad preprocessed train data ... ", end="", flush=True)
# Load training data
train_df = (
    pd.read_csv(path_to_data + 'preprocessed_train.csv', index_col="mid")
)

# Split training data into X_train and y_train
X_train, y_train = split_xy(train_df)
print("OK")

# Initiate the model
model = LinagoraFrequencyPredictor()

print("\tFit the model to the train data ... ", end="", flush=True)
# Fit the model with the training data
model.fit(X_train, y_train)
print("OK", end="\n\n")


print("\tLoad preprocessed test data ... ", end="", flush=True)
# Load test data
X_test = pd.read_csv(path_to_data + 'preprocessed_test.csv', index_col="mid")
print("OK")

print("\tMake predictions on test data ... ", end="", flush=True)
# Predict the labels of X_test
y_pred = model.predict(X_test)
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
    output_filename=submission_folder_path + "predictions_frequency.txt"
)
print("OK")


print("\nPipeline script completed in %0.2f seconds" % (time() - initial_time))
