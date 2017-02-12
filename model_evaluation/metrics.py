import numpy as np
import warnings


def average_precision(recipients_pred, true_recipients):
    """
    Compute average precision@10 between 2 lists (one recipients prediction for one message)
    """
    precision = 0.0
    true_recipients_in_predictions = 0

    for prediction_rank, prediction in enumerate(recipients_pred):
        if prediction in true_recipients and prediction not in recipients_pred[:prediction_rank]:
            true_recipients_in_predictions += 1
            precision += true_recipients_in_predictions / (prediction_rank + 1)

    return round(precision / min(len(true_recipients), 10), 5)


def mean_average_precision(recipients_preds, true_recipients_list):
    """
    Compute *mean* average precision@10 between 2 lists of lists
    (all recipients prediction for all messages)
    """

    if all(isinstance(recipients_pred, list) for recipients_pred in recipients_preds):
        each_prediction_average_precision = list()

        for recipients_pred, true_recipients in zip(recipients_preds, true_recipients_list):
            each_prediction_average_precision.append(
                average_precision(recipients_pred, true_recipients)
            )

        return np.mean(each_prediction_average_precision)

    else:
        warnings.warn(
            "Use average precision to compute an average precision@10 between two lists",
            UserWarning
        )
        return average_precision(recipients_preds, true_recipients_list)
