import pandas as pd
import text_tools


def merge(X_set, X_info):
    """
    From X_set (index:sender, mids) and
    X_info (index:mid, date, body, recipients), create a
    X_preprocessed (index:mid, sender, date, body, recipients)
    """
    sender_per_mid = dict()

    for sender, sender_series in X_set.iterrows():
        for mid in sender_series['mids'].split(' '):
            sender_per_mid[int(mid)] = sender

    X_preprocessed = X_info.copy()
    X_preprocessed["sender"] = pd.Series(
        data=sender_per_mid,
        index=sender_per_mid.keys()
    )

    return X_preprocessed


def preprocess_dates(X):
    X_date = X['date'].copy()
    for date_idx, date in enumerate(X['date'].values):
        if '0001' in date:
            X_date.iloc[date_idx] = '2001' + X['date'].iloc[date_idx][4:]
        if '0002' in date:
            X_date.iloc[date_idx] = '2002' + X['date'].iloc[date_idx][4:]
    return pd.to_datetime(X_date, format="%Y-%m-%d %H:%M:%S")


def clean_emails_bodies(df):
    """
    This function takes the test or train dataframe and returns the same
    dataframe with cleaned email bodies.
    E-mail bodies are cleaned using text_tools.clean_text
    """
    cleaned_df = df
    messages_bodies = df["body"].tolist()

    n_messages = len(messages_bodies)
    percentage_cleaned = 0
    percentage_step = 0.05

    cleaned_messages_bodies = []
    for i, message_body in enumerate(messages_bodies):
        if i > (percentage_cleaned + percentage_step) * n_messages:
            percentage_cleaned += percentage_step
            print("\t\t{percentage}% of the messages have been cleaned"
                  .format(percentage=int(round(100 * percentage_cleaned,0))))

        cleaned_messages_bodies.append(text_tools.clean_text(message_body))
    cleaned_df["body"] = cleaned_messages_bodies

    return cleaned_df
