from datetime import datetime


def create_submission(mids_prediction, submission_folder_path="submissions/",
                      output_filename=None):
    """
    Create a submit csv file from mids_prediction.
    mids_prediction should be of the form : {mid: list_of_recipients}
    """
    if output_filename is None:
        output_filename = (
            "submission_" + datetime.now().strftime("%d_%m_%H_%M_%S") + ".txt"
        )

    with open(submission_folder_path + output_filename, 'w') as output_file:
        output_file.write('mid,recipients' + '\n')
        for mid, prediction in mids_prediction.items():
            output_file.write(str(mid) + ',' + ' '.join(prediction) + '\n')
