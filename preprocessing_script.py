# TODO: 1-Preprocess invalid sender emails
# TODO: 1T-Think of a way to extract features from the text : theme, "FYI", cited names "Hi Ben",
# TODO: "Thanks David", ...
# TODO: 2T-Try and use pre-trained word embeddings (e.g., Google News), part-of-speech taggers

# TODO: 1G-Think of a way to use a well designed graph and build it
# TODO: 1G-Nodes = all possible recipients, Edges = message sent at least once, Weight = number of
# TODO: messages sent
# TODO: 2G-Use possible relations between recipients
# TODO: (for example "Ben is almost always informed with Franck about football")

import pandas as pd
import preprocessing
from time import time

initial_time = time()

print("Preprocessing script", end="\n\n")

path_to_data = "data/"


print("\tLoad train data ... ", end="", flush=True)
training_df = pd.read_csv(path_to_data + 'training_set.csv', index_col='sender')
training_info_df = pd.read_csv(path_to_data + 'training_info.csv', index_col='mid')
print("OK")

print("\tPreprocess train data ... ", end="", flush=True)
preprocessed_training_df = preprocessing.tools.merge(training_df, training_info_df)
print("OK")

preprocessed_train_filename = "preprocessed_train.csv"

print("\tCreate " + path_to_data + preprocessed_train_filename + " file ... ", end="", flush=True)
preprocessed_training_df.to_csv(path_to_data + preprocessed_train_filename)
print("OK", end="\n\n")


print("\tLoad test data ... ", end="", flush=True)
test_df = pd.read_csv(path_to_data + 'test_set.csv', index_col='sender')
test_info_df = pd.read_csv(path_to_data + 'test_info.csv', index_col='mid')
print("OK")

print("\tPreprocess test data ... ", end="", flush=True)
preprocessed_test_df = preprocessing.tools.merge(test_df, test_info_df)
print("OK")

preprocessed_test_filename = "preprocessed_test.csv"

print("\tCreate " + path_to_data + preprocessed_test_filename + " file ... ", end="", flush=True)
preprocessed_test_df.to_csv(path_to_data + preprocessed_test_filename)
print("OK")


print("\nPreprocessing script completed in %0.2f seconds" % (time() - initial_time))
