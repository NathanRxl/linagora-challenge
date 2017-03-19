from .Submissioner import create_submission
import pandas as pd
import unittest


def mids_prediction_from_file(submission_file):
    """
    Compute the mids_prediction from the submission_file. (For test purposes)
    """
    predictions_df = pd.read_csv(submission_file, index_col='mid')
    mids_prediction = {}
    for mid, prediction_series in predictions_df.iterrows():
        prediction = prediction_series["recipients"].split(' ')
        mids_prediction[mid] = prediction
    return mids_prediction


class TestSubmissioner(unittest.TestCase):

    def test_submissioner(self):
        submission_folder = "../submissions/"
        baseline_file = "predictions_frequency.txt"
        baseline_pred = mids_prediction_from_file(
            submission_folder + baseline_file
        )

        baseline_test_file = "predictions_frequency_test.txt"
        create_submission(
            baseline_pred,
            submission_folder_path=submission_folder,
            output_filename=baseline_test_file
        )
        test_baseline_pred = mids_prediction_from_file(
            submission_folder + baseline_test_file
        )

        self.assertEqual(baseline_pred, test_baseline_pred)


if __name__ == "__main__":
    unittest.main()
