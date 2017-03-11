import pandas as pd
import preprocessing
from time import time

initial_time = time()

print("Preprocessing script", end="\n\n")

path_to_data = "data/"


print("\tLoad train data ... ", end="", flush=True)
train_df = pd.read_csv(path_to_data + 'training_set.csv', index_col='sender')
train_info_df = (
    pd.read_csv(path_to_data + 'training_info.csv', index_col='mid')
)
print("OK")

print("\tPreprocess train data ... ", end="", flush=True)
preprocessed_train_df = preprocessing.merge(train_df, train_info_df)
preprocessed_train_df['date'] = (
    preprocessing.preprocess_dates(preprocessed_train_df)
)
preprocessed_train_df = preprocessed_train_df.sort_values(by="date")
preprocessed_train_df = preprocessed_train_df[preprocessed_train_df['date'] >= '2001-06']
preprocessed_train_df['recipients'] = (
    preprocessed_train_df['recipients'].apply(
        lambda recipients: [recipient
                            for recipient in set(recipients.split(" "))
                            if '@' in recipient]
    ).apply(lambda recipient_list: " ".join(recipient_list))
)
print("OK")

preprocessed_train_filename = "preprocessed_train.csv"

print(
    "\tCreate " + path_to_data + preprocessed_train_filename + " file ... ",
    end="",
    flush=True
)
preprocessed_train_df.to_csv(path_to_data + preprocessed_train_filename)
print("OK", end="\n\n")


print("\tLoad test data ... ", end="", flush=True)
test_df = pd.read_csv(path_to_data + 'test_set.csv', index_col='sender')
test_info_df = pd.read_csv(path_to_data + 'test_info.csv', index_col='mid')
print("OK")

print("\tPreprocess test data ... ", end="", flush=True)
preprocessed_test_df = preprocessing.merge(test_df, test_info_df)
preprocessed_test_df['date'] = (
    preprocessing.preprocess_dates(preprocessed_test_df)
)
print("OK")

preprocessed_test_filename = "preprocessed_test.csv"

print(
    "\tCreate " + path_to_data + preprocessed_test_filename + " file ... ",
    end="",
    flush=True
)
preprocessed_test_df.to_csv(path_to_data + preprocessed_test_filename)
print("OK")


print(
    "\nPreprocessing script completed in %0.2f seconds"
    % (time() - initial_time)
)
