from collections import Counter, defaultdict
import math
import random
import warnings
import tables
from time import time
from datetime import datetime
import json
import atexit

from model_evaluation import metrics
import general_tools
import text_tools

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def use_precomputed_cache(cache):
    def memoize(f):
        def use_cache(*args):
            args_key = str(args[0]) + "/" + str(args[1]) + "/" + str(args[2])
            if args_key in cache:
                return cache[args_key]
            else:
                return cache.update({args_key: f(*args[:3])}) or cache[args_key]
        return use_cache
    return memoize


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


def compute_first_names_ab(sender_ab):

    SURNAMES = {
        "robert": ["bob"],
        "susan": ["sue"],
        "james": ["jim", "jimbo"],
        "jturnure": ["jim"],
        "david": ["dave"],
        "patrick": ["pat"],
        "jdasovic": ["jeff"],
        "jrb": ["jim"],
        "styler": ["sally"],
        "jw1000mac": ["jim"],
        "nielsenj": ["jim"],
        "jborowic": ["jim"],
        "steven": ["steve"],
    }

    FIRST_NAME_NOT_IN_ADDRESS = {
        "c..williams@enron.com": "bob",  # 32056 + another
        "a..hueter@enron.com": "barbara",  # 49594
        "j..noske@enron.com": "linda",  # 49604
        "l..lawrence@enron.com": "linda",  # 49611
        "w..cantrell@enron.com": "becky",  # 49575
    }

    first_names_ab = defaultdict(list)
    unique_sender_contacts = list(set(list(sender_ab.elements())))

    for contact in unique_sender_contacts:
        first_name = contact[:contact.find(".")]
        if '@' in first_name:
            # This surprisingly potentially reduce the cv score a bit
            # compares to
            # continue
            first_name = contact[:contact.find("@")]
        if contact in FIRST_NAME_NOT_IN_ADDRESS:
            first_name = FIRST_NAME_NOT_IN_ADDRESS[contact]
        if len(first_name) > 2:
            first_names_ab[first_name].append(contact)
            if first_name in SURNAMES.keys():
                for surname in SURNAMES[first_name]:
                    first_names_ab[surname].append(contact)

    return first_names_ab


class LinagoraWinningPredictor:
    def __init__(self, recency=None, non_recipients=0.2):
        self.recency = recency
        self.all_train_senders = list()
        # Dict of lgbm (one per user)
        self.lgbm = dict()
        self.global_ab = dict()
        self.recent_ab = dict()
        self.first_names_ab = dict()
        self.all_users = list()
        self.non_recipients = non_recipients
        self.X_train = None
        self.y_train = None

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
            # mid_hour = X_train.loc[mid]["date"].hour
            for recipient in recipients:
                i_ytr.append(1)
                potential_non_recipients.remove(recipient)
                # Add working hours feature
                # i_Xtr["working_hours"].append(int(6 <= mid_hour <= 21))
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
                    # Add working hours feature
                    # i_Xtr["working_hours"].append(int(6 <= mid_hour <= 21))
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
                # Add working hours feature
                # i_Xtr["working_hours"].append(int(6 <= mid_hour <= 21))
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
        return pd.DataFrame.from_dict(data=i_Xtr, orient='columns'), i_ytr

    def fit(self, X_train, y_train):
        # Save all unique sender names in X
        self.all_train_senders = X_train["sender"].unique().tolist()
        self.all_users = self.all_train_senders
        # Store X_train and y_train in the model to access it in predict
        # and avoid unnecessary computations
        self.X_train = X_train
        self.y_train = y_train

        print(
            "\n\tBuild i_Xtr and fit model for each sender ... ",
            end="",
            flush=True
        )
        for sender in self.all_train_senders:
            # Define the sender predictive model
            self.lgbm[sender] = Pipeline([
                ("std", StandardScaler()),
                ("lgbm", LGBMClassifier())
            ])
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
            # Compute first names address book
            self.first_names_ab[sender] = (
                compute_first_names_ab(self.global_ab[sender])
            )

            # Build internal train sets
            sender_internal_Xtr, sender_internal_ytr = (
                self._build_internal_train_sets(sender, X_train, y_train)
            )
            # Fit sender logreg to the internal train sets
            self.lgbm[sender].fit(sender_internal_Xtr, sender_internal_ytr)
        print("OK")

    def _build_internal_test_set(self, sender, X_test):
        # i_Xte for internal X test
        i_Xte = dict()
        sender_idx = X_test["sender"] == sender
        # Thanks God every sender in test appears in train
        global_sender_ab_list = list(self.global_ab[sender].elements())
        potential_recipients = list(set(global_sender_ab_list))
        for mid in X_test[sender_idx].index.tolist():
            i_Xte[mid] = defaultdict(list)
            # mid_hour = X_test.loc[mid]["date"].hour
            for recipient in potential_recipients:
                # Add working hours feature
                # i_Xte[mid]["working_hours"].append(int(6 <= mid_hour <= 21))
                # Add global frequency feature
                i_Xte[mid]["global_sent_frequency"].append(
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
                        i_Xte[mid][recency_feature_name].append(
                            sender_recent_ab[recipient]
                            / len(list(sender_recent_ab.elements()))
                        )
            i_Xte[mid] = pd.DataFrame.from_dict(
                data=i_Xte[mid],
                orient='columns'
            )

        # Return the internal test set
        return i_Xte

    def compute_co_occurrence(self, sender, contact_i, contact_j):

        if contact_i == contact_j:
            return 0.0

        n_co_occurences = 0
        sender_idx = self.X_train["sender"] == sender
        for mid in self.X_train[sender_idx].index.tolist():
            mid_recipients = self.y_train.loc[mid]["recipients"].split(" ")
            if contact_i in mid_recipients and contact_j in mid_recipients:
                n_co_occurences += 1
        n_messages_to_contact_i = self.global_ab[sender][contact_i]

        return n_co_occurences / n_messages_to_contact_i

    def predict(self, X_test, y_true=None, store_scores=True,
                use_cooccurrences=True, store_i_Xte=None,
                precomputed_i_Xte=None, store_cooccurrences=None,
                precomputed_cooccurrences=None):
        if use_cooccurrences and precomputed_cooccurrences is not None:
            try:
                co_occurrences_cache = json.load(
                    open(precomputed_cooccurrences, 'r')
                )
            except (IOError, ValueError):
                co_occurrences_cache = dict()
        predictions = dict()
        # Save all unique sender names in X
        all_test_senders = X_test["sender"].unique().tolist()
        if store_i_Xte is not None:
            hdf_file = pd.HDFStore(store_i_Xte)
        print("\n\tBuild i_Xte and predict for each sender ... ")
        if y_true is not None:
            scores = dict()
        # Number of the current prediction in print
        n_prediction = 1
        for sender in all_test_senders:
            t0 = time()
            sender_idx = X_test["sender"] == sender
            if precomputed_i_Xte is not None:
                i_Xte = dict()
                for mid in X_test[sender_idx].index.tolist():
                    i_Xte[mid] = pd.read_hdf(
                        precomputed_i_Xte,
                        key="{}/{}".format(sender, mid)
                    )
                    assert(
                        self._build_internal_test_set(sender, X_test)[mid]
                            .equals(i_Xte[mid])
                    )
            else:
                i_Xte = self._build_internal_test_set(sender, X_test)
            if store_i_Xte is not None:
                warnings.filterwarnings(
                    'ignore',
                    category=tables.NaturalNameWarning
                )
                for mid in X_test[sender_idx].index.tolist():
                    hdf_file.put("{}/{}".format(sender, mid), i_Xte[mid])
            print(
                "\t\t{}/125 Predict for {}".format(n_prediction, sender)
                .ljust(60, " ") + "... ",
                end="",
                flush=True
            )
            for mid in X_test[sender_idx].index.tolist():
                mid_pred_probas = (
                    self.lgbm[sender].predict_proba(i_Xte[mid])[:, 1]
                )
                unique_r_ab = (
                    np.array(
                        list(set(list(self.global_ab[sender].elements())))
                    )
                )
                # Predict first the first names found in the body
                truncated_body = (
                    text_tools.truncate_body(X_test.loc[mid]["body"])
                )
                best_r = np.array(list(), int)
                prediction_outside_ab = []

                if "task assignment" in X_test.loc[mid]["body"][:50].lower():
                    try:
                        no_address_idx = (
                            np.where("no.address@enron.com" == unique_r_ab)[0][0]
                        )
                        best_r = np.append(best_r, no_address_idx)
                        mid_pred_probas[no_address_idx] = 0.0
                    except IndexError:
                        prediction_outside_ab.append("no.address@enron.com")

                for first_name in self.first_names_ab[sender].keys():
                    if first_name in truncated_body:
                        if first_name == "jim" and len(truncated_body[truncated_body.find("jim"):]) < 5:
                            # james.d.steffes tends to write very short mails
                            # and sign Jim. False positives.
                            continue
                        possible_r = self.first_names_ab[sender][first_name]
                        recipient_idxs = list()
                        recipient_pred_probas = list()
                        for recipient in possible_r:
                            recipient_idxs.append(
                                np.where(recipient == unique_r_ab)[0][0]
                            )
                            recipient_pred_probas.append(
                                mid_pred_probas[recipient_idxs[-1]]
                            )
                        max_probas_named_r = np.max(recipient_pred_probas)
                        mask = (
                            np.array(
                                recipient_pred_probas > max_probas_named_r - 0.01,
                                int
                            )
                        )
                        masked_recipient_pred_probas = (
                            recipient_pred_probas * mask
                        )
                        most_probable_named_r = (
                            np.argsort(masked_recipient_pred_probas)[::-1]
                        )
                        for most_probable_r in most_probable_named_r:
                            if masked_recipient_pred_probas[most_probable_r] != 0:
                                recipient_idx = recipient_idxs[most_probable_r]
                                best_r = np.append(best_r, recipient_idx)
                                mid_pred_probas[recipient_idx] = 0.0

                if not use_cooccurrences or len(unique_r_ab) <= 10:
                    still_to_predict = 10 - len(best_r)
                    for pred in np.argsort(mid_pred_probas)[::-1][:still_to_predict]:
                        best_r = np.append(best_r, pred)
                else:
                    pred_to_add = max(2, 7 - len(best_r))
                    for pred in np.argsort(mid_pred_probas)[::-1][:pred_to_add]:
                        best_r = np.append(best_r, pred)
                        mid_pred_probas[pred] = 0.0

                    first_best_r = best_r.copy()
                    for first_pred in first_best_r[:(10 - len(first_best_r))]:

                        next_best_pred_probas = mid_pred_probas.copy()
                        co_best = np.zeros(shape=(len(unique_r_ab), ))

                        for r, recipient in enumerate(unique_r_ab):
                            if precomputed_cooccurrences is not None:
                                co_best[r] = (
                                    use_precomputed_cache(co_occurrences_cache)
                                    (self.compute_co_occurrence)(
                                        sender,
                                        unique_r_ab[first_pred],
                                        recipient
                                    )
                                )
                            else:
                                co_best[r] = (
                                    self.compute_co_occurrence(
                                        sender,
                                        unique_r_ab[first_pred],
                                        recipient
                                    )
                                )

                        next_best_pred_probas = next_best_pred_probas * co_best
                        next_best_pred = np.argmax(next_best_pred_probas)

                        # what if the argmax is random (all score are 0.0) ?
                        best_r = np.append(best_r, next_best_pred)
                        mid_pred_probas[next_best_pred] = 0.0

                prediction = (
                    (prediction_outside_ab + list(unique_r_ab[best_r]))[:10]
                )
                predictions[mid] = prediction

            if use_cooccurrences and store_cooccurrences is not None:
                atexit.register(
                    lambda: json.dump(
                        co_occurrences_cache,
                        open(store_cooccurrences, 'w'))
                )

            if y_true is not None:
                y_sender_true_recipients = (
                    general_tools.true_recipients(y_true[sender_idx])
                )
                scores[sender] = metrics.mean_average_precision(
                    mids_prediction=predictions,
                    mids_true_recipients=y_sender_true_recipients
                )
                print(
                    "score %.5f | execution time %.5f"
                    % (scores[sender], time() - t0)
                )
            else:
                print("execution time %.5f" % (time() - t0))

            n_prediction += 1

        if store_i_Xte is not None:
            # close hdf file
            hdf_file.close()
        if y_true is not None and store_scores:
            score_df = pd.DataFrame.from_dict(data=scores, orient='index')
            score_df.columns = ["score"]
            score_df.to_csv(
                "data/scores_"
                + datetime.now().strftime("%d_%m_%H_%M_%S")
                + ".csv", index=True, index_label="sender"
            )
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
                compute_ab(sender, X_train, y_train, recency=None)
            )

            # Save recent recipients counts
            self.recent_address_books[sender] = (
                compute_ab(sender, X_train, y_train, recency=self.recency)
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
