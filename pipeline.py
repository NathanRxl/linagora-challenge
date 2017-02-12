from collections import Counter
import pandas as pd
import random
from time import time

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

# Create Kaggle submission

sub_folder_path = "submissions/"

with open(sub_folder_path + 'predictions_random.txt', 'w') as random_pred_file:
    random_pred_file.write('mid,recipients' + '\n')
    for sender, prediction_dict in predictions_per_sender.items():
        mids = prediction_dict["mids"]
        random_predictions = prediction_dict["random_prediction"]
        for mid_idx, random_prediction in enumerate(random_predictions):
            random_pred_file.write(str(mids[mid_idx]) + ',' + ' '.join(random_prediction) + '\n')

with open(sub_folder_path + 'predictions_frequency.txt', 'w') as most_frequent_pred_file:
    most_frequent_pred_file.write('mid,recipients' + '\n')
    for sender, prediction_dict in predictions_per_sender.items():
        mids = prediction_dict["mids"]
        most_frequent_predictions = prediction_dict["most_frequent_prediction"]
        for mid_idx, most_frequent_prediction in enumerate(most_frequent_predictions):
            most_frequent_pred_file.write(
                str(mids[mid_idx]) + ',' + ' '.join(most_frequent_prediction) + '\n'
            )

print("\nBaseline script completed in %0.2f seconds" % (time() - initial_time))
