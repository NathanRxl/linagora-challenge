import numpy as np


def average_precision(recipients_pred, true_recipients):
    """
    Compute average precision@10 between 2 lists
    (one recipients prediction for one message)
    """
    precision = 0.0
    true_recipients_in_predictions = 0

    for prediction_rank, prediction in enumerate(recipients_pred):
        if (prediction in true_recipients and
                prediction not in recipients_pred[:prediction_rank]):
            true_recipients_in_predictions += 1
            precision += true_recipients_in_predictions / (prediction_rank + 1)

    return round(precision / min(len(true_recipients), 10), 5)


def mean_average_precision(mids_prediction, mids_true_recipients):
    """
    Compute *mean* average precision@10 between 2 dictionaries of predictions
    (all recipients prediction for all messages)
    The dictionaries are assumed to have mids as keys and a list of prediction
    as values.
    """

    each_prediction_average_precision = list()

    for mid, true_recipients in mids_true_recipients.items():
        each_prediction_average_precision.append(
            average_precision(mids_prediction[mid], true_recipients)
        )

    return round(np.mean(each_prediction_average_precision), 5)
