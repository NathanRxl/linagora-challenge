import pandas as pd
import preprocessing
from time import time

initial_time = time()

print("Text preprocessing script", end="\n\n")

path_to_data = "data/"


print("\tLoad train data ... ", end="", flush=True)
train_df = (
    pd.read_csv(path_to_data + 'preprocessed_train.csv', index_col="mid")
)
print("OK")

print("\tPreprocess train data ... ")
preprocessed_training_df = preprocessing.clean_emails_bodies(train_df)
print("\tOK")

preprocessed_train_filename = "text_preprocessed_train.csv"
print(
    "\tCreate " + path_to_data + preprocessed_train_filename + " file ... ",
    end="",
    flush=True
)
preprocessed_training_df.to_csv(path_to_data + preprocessed_train_filename)
print("OK", end="\n\n")


# print("\tLoad test data ... ", end="", flush=True)
# test_df = (
#     pd.read_csv(path_to_data + 'preprocessed_test.csv', index_col="mid")
# )
# print("\tOK")

# print("\tPreprocess test data ... ")
# preprocessed_testing_df = preprocessing.clean_emails_bodies(test_df)
# print("OK")

# preprocessed_test_filename = "text_preprocessed_test.csv"
# print(
#     "\tCreate " + path_to_data + preprocessed_test_filename + " file ... ",
#     end="",
#     flush=True
# )
# preprocessed_testing_df.to_csv(path_to_data + preprocessed_test_filename)
# print("OK", end="\n\n")


print(
    "\nText preprocessing script completed in %0.2f seconds"
    % (time() - initial_time)
)
