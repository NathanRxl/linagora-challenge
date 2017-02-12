from collections import Counter
import pandas as pd
import random
from time import time

from general_tools import Submissioner


initial_time = time()

print("Pipeline script", end="\n\n")

path_to_data = "data/"

# Load training data
training_df = pd.read_csv(path_to_data + 'training_set.csv', index_col='sender')
training_info_df = pd.read_csv(path_to_data + 'training_info.csv', index_col='mid')

# Load test data
test_df = pd.read_csv(path_to_data + 'test_set.csv', index_col='sender')
# test_info_df = pd.read_csv(path_to_data + 'test_info.csv', index_col='mid')

# ----------------------------
# Create some handy structures
# ----------------------------

# Convert training set to dictionary
mids_per_sender = dict()
for sender, sender_series in training_df.iterrows():
    mids_per_sender[sender] = [int(mid) for mid in sender_series['mids'].split(' ')]

# Save all unique sender names
all_training_senders = list(training_df.index)

# Create address book with frequency information for each user
print("Create address book with frequency information for each user", end="\n\n")
address_books = dict()

for sender_nb, (sender, mids) in enumerate(mids_per_sender.items()):
    sender_recipients = []
    for mid in mids:
        mid_recipients = (
            [recipient
             for recipient in training_info_df.loc[mid]['recipients'].split(' ')
             if '@' in recipient]
        )
        sender_recipients.append(mid_recipients)
    # Flatten sender recipients
    sender_recipients = [recipient for recipients in sender_recipients for recipient in recipients]
    # Save recipients counts
    address_books[sender] = Counter(sender_recipients)

    if (sender_nb + 1) % 12.5 == 0:
        print("\t %d senders have been added to the address book for now" % (sender_nb + 1))

# Save all unique recipient names
all_recipients = (
    list(set([recipient
              for recipient_counters in address_books.values()
              for recipient in recipient_counters]))
)

# Save all unique user names
all_users = list(set(all_training_senders + all_recipients))

# ---------------------------------
# Prediction strategies (baselines)
# ---------------------------------

# Will contain email ids, predictions for random baseline, and predictions for frequency baseline
predictions_per_sender = dict()

for sender, sender_series in test_df.iterrows():
    # get IDs of the emails for which recipient prediction is needed
    mids_to_predict = [int(mid) for mid in sender_series['mids'].split(' ')]
    random_predictions = []
    most_frequent_predictions = []
    # Select most frequent recipients for the sender
    most_frequent_recipients = (
        [recipient for recipient, frequency in address_books[sender].most_common(10)]
    )
    for mid_to_predict in mids_to_predict:
        # select k users at random
        random_predictions.append(random.sample(all_users, 10))
        # for the frequency baseline, the predictions are always the same
        most_frequent_predictions.append(most_frequent_recipients)
    predictions_per_sender[sender] = {
        "mids": mids_to_predict,
        "random_prediction": random_predictions,
        "most_frequent_prediction": most_frequent_predictions
    }


# Build the mids_prediction dict for both baselines
frequency_mids_prediction = {}
random_mids_prediction = {}
for sender in predictions_per_sender.keys():
    mids = predictions_per_sender[sender]["mids"]
    random_predictions = predictions_per_sender[sender]["random_prediction"]
    frequency_predictions = predictions_per_sender[sender]["most_frequent_prediction"]

    for mid_idx, prediction in enumerate(random_predictions):
        random_mids_prediction[mids[mid_idx]] = prediction

    for mid_idx, prediction in enumerate(frequency_predictions):
        random_mids_prediction[mids[mid_idx]] = prediction


# Create Kaggle submission
submission_folder_path = "submissions/"

Submissioner.create_submission(frequency_mids_prediction,
        output_filename=submission_folder_path + "predictions_frequency.txt")

Submissioner.create_submission(random_mids_prediction,
        output_filename=submission_folder_path + "predictions_random.txt")


print("\nBaseline script completed in %0.2f seconds" % (time() - initial_time))
