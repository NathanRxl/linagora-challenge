from collections import Counter


class LinagoraFrequencyPredictor:
    def __init__(self):
        """
        self.address_books:
            dict, (key:sender, values:Counter(recipient, frequency of sending to this recipient))
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
            for recipients in y_train[X_train["sender"] == sender]["recipients"].tolist():
                sender_recipients.extend(
                    [recipient for recipient in recipients.split(' ') if '@' in recipient]
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
        predictions_per_sender = dict()

        # Save all unique sender names in test
        all_test_senders = X_test["sender"].unique().tolist()

        for sender in all_test_senders:
            # Select most frequent recipients of the sender
            most_frequent_recipients = (
                [recipient for recipient, frequency in self.address_books[sender].most_common(10)]
            )
            # Message ids for which a recipient prediction is needed
            mids_to_predict = [int(mid) for mid in
                               X_test[X_test["sender"] == sender].index.tolist()]

            most_frequent_predictions = []
            for mid_to_predict in mids_to_predict:
                # for the frequency baseline, the predictions are always the same
                most_frequent_predictions.append(most_frequent_recipients)

            predictions_per_sender[sender] = {
                "mids": mids_to_predict,
                "most_frequent_prediction": most_frequent_predictions
            }

        # Build the mids_prediction dict
        frequency_mids_prediction = {}
        for sender in predictions_per_sender.keys():
            mids = predictions_per_sender[sender]["mids"]
            frequency_predictions = predictions_per_sender[sender]["most_frequent_prediction"]

            for mid, frequency_prediction in zip(mids, frequency_predictions):
                frequency_mids_prediction[mid] = frequency_prediction

        return frequency_mids_prediction
