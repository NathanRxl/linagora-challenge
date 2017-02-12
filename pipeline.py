from collections import Counter
import pandas as pd
from time import time

from general_tools import Submissioner


initial_time = time()

print("Pipeline script", end="\n\n")

path_to_data = "data/"

# Load training data
train_df = pd.read_csv(path_to_data + 'preprocessed_train.csv', index_col="mid")

# Load test data
test_df = pd.read_csv(path_to_data + 'preprocessed_test.csv', index_col="mid")

# Create address book with frequency information for each user
print("Create address book with frequency information for each user", end="\n\n")
address_books = dict()

# Save all unique sender names in train
all_train_senders = train_df["sender"].unique().tolist()
# Save all unique user names
all_users = set(all_train_senders)

for sender_nb, sender in enumerate(all_train_senders):

    sender_recipients = []
    for recipients in train_df[train_df["sender"] == sender]["recipients"].tolist():
        sender_recipients.extend(
            [recipient for recipient in recipients.split(' ') if '@' in recipient]
        )

    # Save recipients counts
    address_books[sender] = Counter(sender_recipients)

    # Save all unique recipient names
    for recipient in sender_recipients:
        all_users.add(recipient)

    if (sender_nb + 1) % 12.5 == 0:
        print("\t %d senders have been added to the address book for now" % (sender_nb + 1))

# Ultimately change the format of all_users
all_users = list(all_users)

# ---------------------------------
# Prediction strategies (baselines)
# ---------------------------------

# Will contain email ids, predictions for random baseline, and predictions for frequency baseline
predictions_per_sender = dict()

# Save all unique sender names in test
all_test_senders = test_df["sender"].unique().tolist()

for sender in all_test_senders:
    most_frequent_predictions = []
    # Select most frequent recipients of the sender
    most_frequent_recipients = (
        [recipient for recipient, frequency in address_books[sender].most_common(10)]
    )
    # Message ids for which a recipient prediction is needed
    mids_to_predict = [int(mid) for mid in test_df[test_df["sender"] == sender].index.tolist()]

    for mid_to_predict in mids_to_predict:
        # for the frequency baseline, the predictions are always the same
        most_frequent_predictions.append(most_frequent_recipients)
    predictions_per_sender[sender] = {
        "mids": mids_to_predict,
        "most_frequent_prediction": most_frequent_predictions
    }


# Build the mids_prediction dict for both baselines
frequency_mids_prediction = {}
random_mids_prediction = {}
for sender in predictions_per_sender.keys():
    mids = predictions_per_sender[sender]["mids"]
    frequency_predictions = predictions_per_sender[sender]["most_frequent_prediction"]

    for mid_idx, prediction in enumerate(frequency_predictions):
        frequency_mids_prediction[mids[mid_idx]] = prediction


# Create Kaggle submission
submission_folder_path = "submissions/"

Submissioner.create_submission(
    frequency_mids_prediction,
    output_filename=submission_folder_path + "predictions_frequency.txt"
)

print("\nPipeline script completed in %0.2f seconds" % (time() - initial_time))
