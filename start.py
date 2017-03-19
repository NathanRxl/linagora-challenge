import os


"""
This script is to be run in order to compute our best submission.
This submission is in the file : submissions/best_submission.txt
"""

# Check that the data folder exists
if not os.path.isdir("data"):
    print("You first have to create the data directory and put the input " \
          "files in it")

# Check that the input files are in the data folder
input_filenames = [
    "training_info.csv", "training_set.csv", "test_info.csv", "test_set.csv"
]
for file_name in input_filenames:
    if not os.path.exists("data/" + file_name):
        print("You first have to put the file {file_name} in the data directory"
              .format(file_name=file_name)
        )

# Run the preprocessing script
os.system("python3 preprocessing_script.py")

# Run the text preprocessing script
os.system("python3 text_preprocessing_script.py")

# Run the pipeline to build the submission
os.system("python3 pipeline.py ")
