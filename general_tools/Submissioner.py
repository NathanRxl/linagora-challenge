from datetime import datetime


def create_submission(mids_prediction, output_filename=None):
    """
    Create a submit csv file from mids_prediction.
    mids_prediction should be of the form : {mid: list_of_recipients}
    """
    if output_filename is None:
        output_filename = (
            "submissions/submission_"+ datetime.now().strftime("%Y_%m_%d_%M_%S") + ".txt"
        )

    with open(output_filename, 'w') as output_file:
        output_file.write('mid,recipients' + '\n')
        for mid, prediction in mids_prediction.items():
            output_file.write(str(mid) + ',' + ' '.join(prediction) + '\n')
