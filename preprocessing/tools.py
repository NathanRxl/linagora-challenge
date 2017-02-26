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


def preprocess_dates(X):
    X_date = X['date'].copy()
    for date_idx, date in enumerate(X['date'].values):
        if '0001' in date:
            X_date.iloc[date_idx] = '2001' + X['date'].iloc[date_idx][4:]
        if '0002' in date:
            X_date.iloc[date_idx] = '2002' + X['date'].iloc[date_idx][4:]
    return pd.to_datetime(X_date, format="%Y-%m-%d %H:%M:%S")
