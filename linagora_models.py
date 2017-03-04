from collections import Counter, defaultdict
import math
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def compute_ab(sender, X_train, y_train, recency=None):
    sender_recipients = list()

    sender_idx = X_train["sender"] == sender

    if recency is None:
        sender_recipients_list = y_train.loc[sender_idx]["recipients"].tolist()
    else:
        recent_sender_idx = (
            X_train[sender_idx]
            .sort_values(by=["date"], ascending=[1])
            .index.tolist()[-recency:]
        )
        sender_recipients_list = (
            y_train.loc[recent_sender_idx]["recipients"].tolist()
        )

    for recipients in sender_recipients_list:
        sender_recipients.extend(
            [recipient for recipient in recipients.split(' ')]
        )
    # Return recipients counts
    return Counter(sender_recipients)


class LinagoraWinningPredictor:
    def __init__(self, recency=None, non_recipients=0.2):
        self.recency = recency
        self.all_train_senders = list()
        # Dict of logreg (one per user)
        self.logreg = dict()
        self.global_ab = dict()
        self.recent_ab = dict()
        self.all_users = list()
        self.non_recipients = non_recipients

    def _build_internal_train_sets(self, sender, X_train, y_train):
        # i_Xtr for internal X train
        i_Xtr = defaultdict(list)
        # i_ytr for internal y train
        i_ytr = list()
        sender_idx = X_train["sender"] == sender
        global_sender_ab_list = list(self.global_ab[sender].elements())
        for mid in X_train[sender_idx].index.tolist():
            potential_non_recipients = list(set(global_sender_ab_list))
            # Fill lines with target 1
            recipients = y_train.loc[mid]["recipients"].split(" ")
            for recipient in recipients:
                i_ytr.append(1)
                potential_non_recipients.remove(recipient)
                # Add global frequency feature
                i_Xtr["global_sent_frequency"].append(
                    self.global_ab[sender][recipient]
                    / len(global_sender_ab_list)
                )
                # Add recency features
                if self.recency is not None:
                    for recency in self.recency:
                        recency_feature_name = (
                            "{}_recent_sent_frequency".format(recency)
                        )
                        sender_recent_ab = self.recent_ab[sender][recency]
                        i_Xtr[recency_feature_name].append(
                            sender_recent_ab[recipient]
                            / len(list(sender_recent_ab.elements()))
                        )
            # Fill lines with target 0
            nb_non_recipients_to_select = min(
                len(potential_non_recipients),
                max(
                    len(recipients),
                    math.ceil(
                        self.non_recipients * len(potential_non_recipients)
                    )
                )
            )

            if nb_non_recipients_to_select != 0:
                non_recipients_sample = random.sample(
                    potential_non_recipients,
                    nb_non_recipients_to_select
                )

                for non_recipient in non_recipients_sample:
                    i_ytr.append(0)
                    # Add global frequency feature
                    i_Xtr["global_sent_frequency"].append(
                        self.global_ab[sender][non_recipient]
                        / len(global_sender_ab_list)
                    )
                    # Add recency features
                    if self.recency is not None:
                        for recency in self.recency:
                            recency_feature_name = (
                                "{}_recent_sent_frequency".format(recency)
                            )
                            sender_recent_ab = self.recent_ab[sender][recency]
                            i_Xtr[recency_feature_name].append(
                                sender_recent_ab[non_recipient]
                                / len(list(sender_recent_ab.elements()))
                            )
            else:
                # Add a line of 0 to guarantee the two classes are
                # always represented.
                #
                # wsmith@wordsmith.org
                # schwabalerts.marketupdates@schwab.com
                # alex@pira.com
                # Perhaps another strategy of prediction could be applied to
                # those 3 users. Because these users have very few emails in
                # the train, with very few recipients, always the same ones
                i_ytr.append(0)
                # Add global frequency feature
                i_Xtr["global_sent_frequency"].append(0.0)
                # Add recency features
                if self.recency is not None:
                    for recency in self.recency:
                        recency_feature_name = (
                            "{}_recent_sent_frequency".format(recency)
                        )
                        i_Xtr[recency_feature_name].append(0.0)

        # Return the internal train set
        return pd.DataFrame(data=i_Xtr), i_ytr

    def fit(self, X_train, y_train):
        # Save all unique sender names in X
        self.all_train_senders = X_train["sender"].unique().tolist()
        self.all_users = self.all_train_senders

        for sender in self.all_train_senders:
            # Define the sender predictive model
            self.logreg[sender] = LogisticRegression(C=100000, n_jobs=-1)
            # Compute sender address book
            self.global_ab[sender] = (
                compute_ab(sender, X_train, y_train, recency=None)
            )
            # Compute sender recent address book
            if self.recency is not None:
                self.recent_ab[sender] = dict()
                for recency in self.recency:
                    self.recent_ab[sender][recency] = (
                        compute_ab(sender, X_train, y_train, recency=recency)
                    )
            # Build internal train sets
            sender_internal_Xtr, sender_internal_ytr = (
                self._build_internal_train_sets(sender, X_train, y_train)
            )
            # Fit sender logreg to the internal train sets
            self.logreg[sender].fit(sender_internal_Xtr, sender_internal_ytr)

    def _build_internal_test_set(self, sender, X_test):
        # i_Xte for internal X test
        i_Xte = dict()
        sender_idx = X_test["sender"] == sender
        # Thanks God every sender in test appears in train
        potential_recipients = list(
            set(list(self.global_ab[sender].elements()))
        )
        for mid in X_test[sender_idx].index.tolist():
            i_Xte[mid] = defaultdict(list)
            for recipient in potential_recipients:
                # Add global frequency feature
                i_Xte[mid]["global_sent_frequency"].append(
                    self.global_ab[sender][recipient]
                    / len(potential_recipients)
                )
                # Add recency features
                if self.recency is not None:
                    for recency in self.recency:
                        recency_feature_name = (
                            "{}_recent_sent_frequency".format(recency)
                        )
                        sender_recent_ab = self.recent_ab[sender][recency]
                        i_Xte[mid][recency_feature_name].append(
                            sender_recent_ab[recipient]
                            / len(list(sender_recent_ab.elements()))
                        )
            i_Xte[mid] = pd.DataFrame(data=i_Xte[mid])

        # Return the internal test set
        return i_Xte

    def predict(self, X_test):
        predictions = dict()
        # Save all unique sender names in X
        all_test_senders = X_test["sender"].unique().tolist()
        for sender in all_test_senders:
            i_Xte = self._build_internal_test_set(sender, X_test)

            sender_idx = X_test["sender"] == sender
            for mid in X_test[sender_idx].index.tolist():
                mid_pred_probas = (
                    self.logreg[sender].predict_proba(i_Xte[mid])[:, 1]
                )
                global_sender_ab_list = (
                    np.array(
                        list(set(list(self.global_ab[sender].elements())))
                    )
                )
                best_recipients = np.argsort(mid_pred_probas)[::-1][:10]
                prediction = global_sender_ab_list[best_recipients]
                predictions[mid] = prediction

        return predictions


class FrequencyPredictor:
    def __init__(self, recency=None):
        """
        self.address_books:
            type: dict
            key: sender,
            values: Counter(recipient, frequency of sending to this recipient))
        """
        self.address_books = dict()
        self.all_train_senders = list()
        self.recency = recency
        self.recent_address_books = dict()

    def fit(self, X_train, y_train, verbose=True):
        # Save all unique sender names in X
        self.all_train_senders = X_train["sender"].unique().tolist()

        for sender_nb, sender in enumerate(self.all_train_senders):

            # Save complete recipients counts
            self.address_books[sender] = (
                compute_ab(X_train, y_train, sender, recency=None)
            )

            # Save recent recipients counts
            self.recent_address_books[sender] = (
                compute_ab(X_train, y_train, sender, recency=self.recency)
            )

        if verbose:
            # Compute and print some statistics about the train set
            ab_mean_size = 0
            nb_train_senders = len(self.all_train_senders)

            for user in self.all_train_senders:
                ab_mean_size += (
                    len(self.address_books[user]) / nb_train_senders
                )

            print(
                "\t\tNumber of unique senders in the train:",
                nb_train_senders)
            print("\t\tUser address book mean size:", int(ab_mean_size))

    def predict(self, X_test):
        # Will contain message ids and predictions for frequency predictions
        sender_predictions = dict()

        # Save all unique sender names in test
        all_test_senders = X_test["sender"].unique().tolist()

        for sender in all_test_senders:
            if self.recency is not None:
                recent_addrs_book_most_common = (
                    self.recent_address_books[sender].most_common(10)
                )
                addrs_book_most_common = recent_addrs_book_most_common
                nb_pred = len(recent_addrs_book_most_common)
                if nb_pred < 10:
                    address_book_filtered = {
                        recipient: frequency
                        for recipient, frequency
                        in self.address_books[sender].items()
                        if recipient not in recent_addrs_book_most_common
                    }
                    addrs_book_most_common += (
                        Counter(address_book_filtered)
                        .most_common(10 - nb_pred)
                    )
            else:
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
