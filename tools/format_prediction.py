def true_recipients(y_df):
    """
    This function takes as arguments a data-frame that has mids as index and
    the other column is "recipients"
    It returns the corresponding mids_prediction dict.
    """
    y_test = {}
    mids = y_df.index.tolist()
    predictions = y_df["recipients"]
    for mid in mids:
        recipients = predictions.loc[mid]
        prediction = [recipient for recipient in recipients.split(' ')]
        y_test[mid] = prediction

    return y_test
