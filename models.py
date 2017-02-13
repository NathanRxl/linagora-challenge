from collections import Counter


class LinagoraFrequencyPredictor:
    def __init__(self):
        """
        self.address_books:
            type: dict
            key: sender,
            values: Counter(recipient, frequency of sending to this recipient))
        """
        self.address_books = dict()
        self.all_train_senders = list()
        self.all_users = list()

    def fit(self, X_train, y_train):
        # Save all unique sender names in X
        self.all_train_senders = X_train["sender"].unique().tolist()
        # Save all unique user names
        self.all_users = set(self.all_train_senders)

        for sender_nb, sender in enumerate(self.all_train_senders):

            sender_recipients = []
            sender_idx = X_train["sender"] == sender
            for recipients in y_train[sender_idx]["recipients"].tolist():
                sender_recipients.extend(
                    [recipient
                     for recipient in recipients.split(' ')
                     if '@' in recipient]
                )

            # Save recipients counts
            self.address_books[sender] = Counter(sender_recipients)

            # Save all unique recipient names
            for recipient in sender_recipients:
                self.all_users.add(recipient)

        # Ultimately change the format of all_users
        self.all_users = list(self.all_users)

    def predict(self, X_test):
        # Will contain message ids and predictions for frequency predictions
        sender_predictions = dict()

        # Save all unique sender names in test
        all_test_senders = X_test["sender"].unique().tolist()

        for sender in all_test_senders:
            # Select most frequent recipients of the sender
            addrs_book_most_common = self.address_books[sender].most_common(10)
            most_frequent_recipients = (
                [recipient for recipient, frequency in addrs_book_most_common]
            )
            # Message ids for which a recipient prediction is needed
            sender_idx = X_test["sender"] == sender
            mids_to_predict = (
                [int(mid) for mid in X_test[sender_idx].index.tolist()]
            )

            most_frequent_predictions = []
            for mid_to_predict in mids_to_predict:
                # The predictions are always the same for a same sender
                most_frequent_predictions.append(most_frequent_recipients)

            sender_predictions[sender] = {
                "mids": mids_to_predict,
                "most_frequent": most_frequent_predictions
            }

        # Build the mids_prediction dict
        frequency_mids_prediction = {}
        for sender in sender_predictions.keys():
            mids = sender_predictions[sender]["mids"]
            frequency_predictions = sender_predictions[sender]["most_frequent"]

            for mid, frequency_prediction in zip(mids, frequency_predictions):
                frequency_mids_prediction[mid] = frequency_prediction

        return frequency_mids_prediction
