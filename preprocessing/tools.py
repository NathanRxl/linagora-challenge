import pandas as pd


def merge(X_set, X_info):
    """
    From X_set (index:sender, mids) and X_info (index:mid, date, body, recipients), create a
    X_preprocessed (index:mid, sender, date, body, recipients)
    """
    sender_per_mid = dict()

    for sender, sender_series in X_set.iterrows():
        for mid in sender_series['mids'].split(' '):
            sender_per_mid[int(mid)] = sender

    X_preprocessed = X_info.copy()
    X_preprocessed["sender"] = pd.Series(data=sender_per_mid, index=sender_per_mid.keys())

    return X_preprocessed
