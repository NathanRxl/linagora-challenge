from collections import Counter, defaultdict
from time import time
from datetime import timedelta
import tools

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
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


def compute_ab(sender, X_train, y_train, time_recency=None,
               message_recency=None):
    sender_recipients = list()
    sender_idx = X_train["sender"] == sender

    if time_recency is not None:
        first_recent_date = (
            X_train[sender_idx]['date'].max() - timedelta(days=time_recency)
        )
        recent_sender_idx = (
            (X_train[sender_idx]['date'] >= first_recent_date).index
        )
        sender_recipients_list = (
            y_train.loc[recent_sender_idx]["recipients"].tolist()
        )
    elif message_recency is not None:
        recent_sender_idx = (
            X_train[sender_idx]
            .sort_values(by=["date"], ascending=[1])
            .index.tolist()[-message_recency:]
        )
        sender_recipients_list = (
            y_train.loc[recent_sender_idx]["recipients"].tolist()
        )
    else:
        sender_recipients_list = y_train.loc[sender_idx]["recipients"].tolist()

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
        "jdasovic@enron.com": "jeff",
        "aminard@azdailysun.com": "anne",
        "ameytina@bear.com": "anna", # 30440
        "mklemont@bear.com": "mary kay", # 30688
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
    def __init__(self, time_recency=None, message_recency=None):
        self.time_recency = (
            time_recency if time_recency is not None else list()
        )
        self.msg_recency = (
            message_recency if message_recency is not None else list()
        )
        self.all_train_senders = list()
        # Dict of lgbm (one per sender)
        self.lgbm = dict()
        self.global_ab = dict()
        self.time_recent_ab = defaultdict(dict)
        self.msg_recent_ab = defaultdict(dict)
        self.first_names_ab = dict()
        self.X_train = None
        self.y_train = None

        self.train_senders_recipients = dict()
        self.train_senders_tfidf = dict()
        self.train_senders_vectorizers = dict()

    def _build_internal_train_sets(self, sender, X_train, y_train):
        i_Xtr = defaultdict(dict)
        i_ytr = dict()
        sender_idx = X_train["sender"] == sender
        global_sender_ab_list = list(self.global_ab[sender].elements())

        time_recent_contacts = dict()
        msg_recent_contacts = dict()

        for time_recency in self.time_recency:
            time_recent_contacts[time_recency] = (
                list(self.time_recent_ab[time_recency][sender].elements())
            )
        for msg_recency in self.msg_recency:
            msg_recent_contacts[msg_recency] = (
                list(self.msg_recent_ab[msg_recency][sender].elements())
            )

        all_recipients = self.global_ab[sender].keys()

        for mid in X_train[sender_idx].index.tolist():
            # Fill lines with target 1
            true_recipients = y_train.loc[mid]["recipients"].split(" ")
            for recipient in all_recipients:
                i_ytr[(mid, recipient)] = int(recipient in true_recipients)
                i_Xtr["frequency_scores"][(mid, recipient)] = (
                    self.global_ab[sender][recipient]
                    / len(global_sender_ab_list)
                )

                for time_recency in self.time_recency:
                    time_recency_feature_name = (
                        "time_recency_{}_frequency".format(time_recency)
                    )
                    i_Xtr[time_recency_feature_name][(mid, recipient)] = (
                        self.time_recent_ab[time_recency][sender][recipient]
                        / len(time_recent_contacts)
                    )

                for msg_recency in self.msg_recency:
                    msg_recency_feature_name = (
                        "msg_recency_{}_frequency".format(msg_recency)
                    )
                    i_Xtr[msg_recency_feature_name][(mid, recipient)] = (
                        self.msg_recent_ab[msg_recency][sender][recipient]
                        / len(msg_recent_contacts)
                    )

        features_df = pd.DataFrame.from_dict(data=i_Xtr, orient='columns')
        labels_df = pd.DataFrame.from_dict(
            data={"labels": i_ytr},
            orient='columns'
        )
        return features_df, labels_df

    def _build_internal_knn_train_set(self, sender, X_train):
        sender_idx = X_train["sender"] == sender
        X_train_sender = X_train[sender_idx]
        sender_msgs = X_train_sender["body"].tolist()
        n_sender_mesages = len(sender_msgs)
        sender_mids = np.array(X_train_sender.index.tolist())
        train_recipients = self.train_senders_recipients[sender]

        # Compute knn features for the sender
        i_Xtr = dict()

        # Number of splits
        n_splits = 10
        if n_sender_mesages < n_splits:
            n_splits = 2
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)
        complete_train_indexes = np.arange(n_sender_mesages)
        ct_split_indexes = kf.split(complete_train_indexes)

        for keep_fold_indexes, leave_fold_indexes in ct_split_indexes:
            keep_msgs = np.array(sender_msgs)[keep_fold_indexes]
            leave_msgs = np.array(sender_msgs)[leave_fold_indexes]
            tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            # Fit the tfidf
            keep_tfidf = tfidf_vectorizer.fit_transform(keep_msgs).todense()

            # Make predictions on the fold left
            leave_tfidf = tfidf_vectorizer.transform(leave_msgs).todense()
            leave_keep_distances = leave_tfidf.dot(keep_tfidf.T)
            order = np.argsort(leave_keep_distances)
            order = np.fliplr(order)
            nb_keep_msgs = order.shape[1]

            mids_to_predict = sender_mids[leave_fold_indexes]
            for i, mid_to_predict in enumerate(mids_to_predict):
                # Compute the score of each recipient
                recipients_score = dict()
                for recipient in self.global_ab[sender].keys():
                    recipients_score[recipient] = 0

                # Parameter
                n_selected_msgs = 30
                for j in range(min(n_selected_msgs, nb_keep_msgs)):
                    msg_keep_idx = order[i, j]
                    msg_real_idx = keep_fold_indexes[order[i, j]]
                    msg_recipients = train_recipients[msg_real_idx]
                    for recipient in msg_recipients:
                        recipients_score[recipient] += (
                            leave_keep_distances[i, msg_keep_idx]
                        )

                # Fill the feature dict
                for recipient, score in recipients_score.items():
                    i_Xtr[(mid_to_predict,recipient)] = score

        # Compute dataframes from the features dict and the labels dict
        features_df = pd.DataFrame.from_dict(
            data={"knn_scores": i_Xtr},
            orient='columns'
        )
        return features_df

    def fit(self, X_train, y_train):
        # Save all unique sender names in X
        self.all_train_senders = X_train["sender"].unique().tolist()
        # Store X_train and y_train in the model to access it in predict
        # and avoid unnecessary computation
        self.X_train = X_train
        self.y_train = y_train

        print("\n\tBuild i_Xtr and fit model for each sender ... ")

        # Printing parameters
        n_senders = len(self.all_train_senders)
        percentage_trained = 0
        percentage_step = 0.05

        for i, sender in enumerate(self.all_train_senders):
            # Print the advancement of the training
            if i > (percentage_trained + percentage_step) * n_senders:
                percentage_trained += percentage_step
                print(
                    "\t\t{percentage}% of the senders have been trained"
                    .format(
                        percentage=int(round(100 * percentage_trained, 0))
                    )
                )

            # Compute the train recipients per sender
            self.train_senders_recipients[sender] = list()
            sender_idx = X_train["sender"] == sender
            for recipients in y_train[sender_idx]["recipients"].tolist():
                self.train_senders_recipients[sender].append(
                    [recipient for recipient in recipients.split(' ')]
                )

            # Compute sender address book
            self.global_ab[sender] = (
                compute_ab(sender, X_train, y_train)
            )

            # Compute recent time sender address book
            for time_recency in self.time_recency:
                self.time_recent_ab[time_recency][sender] = (
                    compute_ab(sender, X_train, y_train,
                               time_recency=time_recency)
                )

            # Compute recent message sender address book
            for msg_recency in self.msg_recency:
                self.msg_recent_ab[msg_recency][sender] = (
                    compute_ab(sender, X_train, y_train,
                               message_recency=msg_recency)
                )

            # Compute first names address book
            self.first_names_ab[sender] = (
                compute_first_names_ab(self.global_ab[sender])
            )

            # Build internal train sets
            features_df, labels_df = (
                self._build_internal_train_sets(sender, X_train, y_train)
            )
            knn_features_df = (
                self._build_internal_knn_train_set(sender, X_train)
            )

            train_df = features_df.join(labels_df).join(knn_features_df)
            # train_df = &ls"] = labels_df["labels"]
            feature_names = ["knn_scores"]  # , "frequency_scores"]

            for msg_recency in self.msg_recency:
                feature_names.append(
                    "msg_recency_{}_frequency".format(msg_recency)
                )

            for time_recency in self.time_recency:
                feature_names.append(
                    "time_recency_{}_frequency".format(time_recency)
                )

            i_Xtr = train_df[feature_names].as_matrix()
            i_ytr = train_df["labels"].as_matrix()

            if 0 not in i_ytr:
                # Handle the case when there are no 0 in i_ytr
                # In that case, the algorithm cannot learn
                self.lgbm[sender] = 1
            else:
                # Fit sender logreg to the internal train sets
                self.lgbm[sender] = Pipeline(
                    [
                        ("std", StandardScaler()),
                        ("lgbm", LGBMClassifier())
                    ]
                )
                self.lgbm[sender].fit(i_Xtr, i_ytr)

            # Compute the tfidf matrices and vectorizers.
            # They will be used for the predictions.
            sender_idx = X_train["sender"] == sender
            X_train_sender = X_train[sender_idx]
            sender_msgs = X_train_sender["body"].tolist()

            tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            tfidf_matrix = (
                tfidf_vectorizer.fit_transform(sender_msgs).todense()
            )

            self.train_senders_tfidf[sender] = tfidf_matrix
            self.train_senders_vectorizers[sender] = tfidf_vectorizer

        print("\tOK")

    def _build_internal_test_set(self, sender, X_test):
        # i_Xte for internal X test
        i_Xte = defaultdict(lambda: defaultdict(float))
        sender_idx = X_test["sender"] == sender
        # Thanks God every sender in test appears in train
        global_sender_contacts = list(self.global_ab[sender].elements())
        time_recent_contacts = dict()
        msg_recent_contacts = dict()

        for time_recency in self.time_recency:
            time_recent_contacts[time_recency] = (
                list(self.time_recent_ab[time_recency][sender].elements())
            )

        for msg_recency in self.msg_recency:
            msg_recent_contacts[msg_recency] = (
                list(self.msg_recent_ab[msg_recency][sender].elements())
            )

        potential_recipients = self.global_ab[sender].keys()
        for mid in X_test[sender_idx].index.tolist():
            for recipient in potential_recipients:
                i_Xte["frequency_scores"][(mid, recipient)] = (
                    self.global_ab[sender][recipient]
                    / len(global_sender_contacts)
                )
                if self.time_recency is not None:
                    for time_recency in self.time_recency:
                        time_recency_feature_name = (
                            "time_recency_{}_frequency".format(time_recency)
                        )
                        i_Xte[time_recency_feature_name][(mid, recipient)] = (
                            self.time_recent_ab[time_recency][sender][recipient]
                            / len(time_recent_contacts)
                        )
                if self.msg_recency is not None:
                    for msg_recency in self.msg_recency:
                        msg_recency_feature_name = (
                            "msg_recency_{}_frequency".format(msg_recency)
                        )
                        i_Xte[msg_recency_feature_name][(mid, recipient)] = (
                            self.msg_recent_ab[msg_recency][sender][recipient]
                            / len(msg_recent_contacts)
                        )

        test_features_df = pd.DataFrame.from_dict(data=i_Xte, orient='columns')

        # Return the internal test set
        return test_features_df

    def _build_internal_knn_test_set(self, sender, X_test):
        i_Xte = {}

        # Get the already computed vectorizer, tfidf_matrix
        # and recipients lists
        vectorizer = self.train_senders_vectorizers[sender]
        train_tfidf_matrix = self.train_senders_tfidf[sender]
        train_recipients = self.train_senders_recipients[sender]

        sender_idx = X_test["sender"] == sender
        X_test_sender = X_test[sender_idx]
        sender_msgs = X_test_sender["body"].tolist()
        test_sender_mids = X_test_sender.index.tolist()

        # Compute the tfidf of the test messages
        test_tfidf_matrix = vectorizer.transform(sender_msgs).todense()

        # Compute the matrix of cosine similarities and its argsort
        distances = test_tfidf_matrix.dot(train_tfidf_matrix.T)
        order = np.argsort(distances)
        order = np.fliplr(order)

        nb_train_msgs = order.shape[1]

        for i, mid_to_predict in enumerate(test_sender_mids):
            # Parameter
            n_selected_msgs = 30
            recipients_score = dict()
            for recipient in self.global_ab[sender].keys():
                recipients_score[recipient] = 0

            for j in range(min(n_selected_msgs, nb_train_msgs)):
                train_msg_idx = order[i, j]
                msg_recipients = train_recipients[train_msg_idx]
                for recipient in msg_recipients:
                    recipients_score[recipient] += distances[i, train_msg_idx]
            # Fill the features dict
            for recipient, score in recipients_score.items():
                i_Xte[(mid_to_predict,recipient)] = score

        # Compute data frames from the features dict and the labels dict
        test_features_df = pd.DataFrame.from_dict(
            data={"knn_scores": i_Xte},
            orient='columns',
        )
        return test_features_df

    def compute_co_occurrence(self, sender, contact_i, contact_j):

        if contact_i == contact_j:
            return 0.0

        n_co_occurences = 0
        sender_idx = self.X_train["sender"] == sender
        for mid in self.X_train[sender_idx].index.tolist():
            mid_recipients = self.y_train.loc[mid]["recipients"].split(" ")
            if contact_i in mid_recipients and contact_j in mid_recipients:
                n_co_occurences += 1
        n_msgs_to_contact_i = self.global_ab[sender][contact_i]

        return n_co_occurences / n_msgs_to_contact_i

    def predict(self, X_test):
        predictions = dict()
        # Save all unique sender names in X
        all_test_senders = X_test["sender"].unique().tolist()
        # Number of the current prediction in print
        n_prediction = 1
        print("\tBuild i_Xte and predict for each sender ... ")
        for sender in all_test_senders:
            t0 = time()
            sender_idx = X_test["sender"] == sender
            test_features_df = self._build_internal_test_set(sender, X_test)
            test_knn_features_df = self._build_internal_knn_test_set(sender, X_test)

            # Merge the features
            test_features_df = test_features_df.join(test_knn_features_df)

            print(
                "\t\t{}/125 Predict for {}".format(n_prediction, sender)
                .ljust(60, " ") + "... ",
                end="",
                flush=True
            )

            for mid in X_test[sender_idx].index.tolist():
                feature_names = ["knn_scores"]

                for msg_recency in self.msg_recency:
                    feature_names.append(
                        "msg_recency_{}_frequency".format(msg_recency)
                    )

                for time_recency in self.time_recency:
                    feature_names.append(
                        "time_recency_{}_frequency".format(time_recency)
                    )

                i_Xte = test_features_df[feature_names].loc[mid].as_matrix()

                recipients_list = (
                    test_features_df.loc[mid].index.tolist()
                )
                if self.lgbm[sender] == 1:
                    # Handle the case when there were no 0 in i_ytr
                    # In that case, the algorithm could not learn
                    predictions[mid] = recipients_list[:10]
                else:
                    mid_pred_probas = (
                        self.lgbm[sender].predict_proba(i_Xte)[:, 1]
                    )

                    unique_r_ab = np.array(recipients_list)
                    # Predict first the first names found in the body
                    trunc_body = tools.truncate_body(X_test.loc[mid]["body"])
                    best_r = np.array(list(), int)
                    prediction_outside_ab = []

                    for first_name in self.first_names_ab[sender].keys():
                        if first_name in trunc_body:
                            if len(trunc_body[trunc_body.find(first_name):]) < 5:
                                # james.d.steffes and rick.buy@enron.com
                                # tend to write very short mails
                                # and sign Jim or Rick. This is an attempt to
                                # reduce these kind of false positives.
                                continue

                            if first_name == " don " and " don t" in trunc_body:
                                # Obvious false positive
                                continue

                            possible_r = (
                                self.first_names_ab[sender][first_name]
                            )
                            recipient_idxs = list()
                            r_pred_probas = list()
                            for recipient in possible_r:
                                recipient_idxs.append(
                                    np.where(recipient == unique_r_ab)[0][0]
                                )
                                r_pred_probas.append(
                                    mid_pred_probas[recipient_idxs[-1]]
                                )
                            max_probas_named_r = np.max(r_pred_probas)
                            mask = (
                                np.array(
                                    r_pred_probas >
                                    max_probas_named_r - 0.01,
                                    int
                                )
                            )
                            masked_r_pred_probas = r_pred_probas * mask
                            most_probable_named_r = (
                                np.argsort(masked_r_pred_probas)[::-1]
                            )
                            for most_probable_r in most_probable_named_r:
                                if masked_r_pred_probas[most_probable_r] != 0:
                                    recipient_idx = (
                                        recipient_idxs[most_probable_r]
                                    )
                                    best_r = np.append(best_r, recipient_idx)
                                    mid_pred_probas[recipient_idx] = 0.0

                    still_to_predict = 10 - len(best_r)
                    pred_to_add = (
                        np.argsort(mid_pred_probas)[::-1][:still_to_predict]
                    )
                    for pred in pred_to_add:
                        best_r = np.append(best_r, pred)

                    prediction = (
                        (prediction_outside_ab +
                         list(unique_r_ab[best_r]))[:10]
                    )
                    predictions[mid] = prediction

            print("execution time %.5f" % (time() - t0))
            n_prediction += 1
        return predictions


class LinagoraKnnPredictor:
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
        self.train_senders_tfidf = dict()
        self.train_senders_vectorizers = dict()
        self.train_senders_recipients = dict()
        self.first_names_ab = dict()

    def _build_internal_test_set(self, sender, X_test):
        i_Xte = {}

        # Get the already computed vectorizer, tfidf_matrix and
        # recipients lists
        vectorizer = self.train_senders_vectorizers[sender]
        train_tfidf_matrix = self.train_senders_tfidf[sender]
        train_recipients = self.train_senders_recipients[sender]

        sender_idx = X_test["sender"] == sender
        X_test_sender = X_test[sender_idx]
        sender_messages = X_test_sender["body"].tolist()
        test_sender_mids = X_test_sender.index.tolist()

        # Compute the tfidf of the test messages
        test_tfidf_matrix = vectorizer.transform(sender_messages).todense()

        # Compute the matrix of cosine similarities and its argsort
        distances = test_tfidf_matrix.dot(train_tfidf_matrix.T)
        order = np.argsort(distances, kind="mergesort")
        order = np.fliplr(order)

        nb_train_messages = order.shape[1]

        for i, mid_to_predict in enumerate(test_sender_mids):
            n_selected_messages = 30
            recipients_score = dict()
            for recipient in self.address_books[sender].keys():
                recipients_score[recipient] = 0

            for j in range(min(n_selected_messages, nb_train_messages)):
                train_message_idx = order[i, j]
                message_recipients = train_recipients[train_message_idx]
                for recipient in message_recipients:
                    # The commented line below was used in the best submission
                    # we have submitted before the deadline but this was only
                    # a try we should have removed to submit.
                    # The correct line is the uncommented one.

                    # recipients_score[recipient] += (
                    #     distances[i, train_message_idx] ** 2
                    # )
                    recipients_score[recipient] += (
                        distances[i, train_message_idx]
                    )
            # Fill the features dict
            for recipient, score in recipients_score.items():
                i_Xte[(mid_to_predict,recipient)] = score

        # Compute dataframes from the features dict and the labels dict
        test_features_df = pd.DataFrame.from_dict(
            data={"knn_scores": i_Xte},
            orient='columns',
        )
        return test_features_df

    def fit(self, X_train, y_train):
        # Save all unique sender names in X
        self.all_train_senders = X_train["sender"].unique().tolist()
        # Save all unique user names
        self.all_users = set(self.all_train_senders)
        for sender in self.all_train_senders:
            self.train_senders_recipients[sender] = []
            sender_recipients = []
            sender_idx = X_train["sender"] == sender
            for recipients in y_train[sender_idx]["recipients"].tolist():
                recipients_list = (
                    [recipient for recipient in recipients.split(' ')]
                )
                sender_recipients.extend(recipients_list)
                self.train_senders_recipients[sender].append(recipients_list)

            # Save recipients counts
            self.address_books[sender] = Counter(sender_recipients)

            # Compute the names address book
            self.first_names_ab[sender] = (
                compute_first_names_ab(self.address_books[sender])
            )

            # Save all unique recipient names
            for recipient in sender_recipients:
                self.all_users.add(recipient)

        # Printing parameters
        n_senders = len(self.all_train_senders)
        percentage_trained = 0
        percentage_step = 0.05

        # Compute the train features and labels
        for sender_nb, sender in enumerate(self.all_train_senders):
            # Print the advancement of the training
            if sender_nb > (percentage_trained + percentage_step) * n_senders:
                percentage_trained += percentage_step
                print(
                    "\t\t{}% of the senders have been trained"
                    .format(int(round(100 * percentage_trained, 0)))
                )

            # Compute the tfidf matrices and vectorizers.
            # They will be used for the predictions.
            sender_idx = X_train["sender"] == sender
            X_train_sender = X_train[sender_idx]
            sender_messages = X_train_sender["body"].tolist()

            tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            tfidf_matrix = (tfidf_vectorizer.fit_transform(sender_messages)
                                            .todense())

            self.train_senders_tfidf[sender] = tfidf_matrix
            self.train_senders_vectorizers[sender] = tfidf_vectorizer

        print("\t\t100% of the senders have been trained")
        # Ultimately change the format of all_users
        self.all_users = list(self.all_users)

    def predict(self, X_test):
        # Save all unique sender names in test
        all_test_senders = X_test["sender"].unique().tolist()

        # Compute the test features and make predictions
        predictions = {}
        for sender in all_test_senders:
            sender_idx = X_test["sender"] == sender
            test_features_df = self._build_internal_test_set(sender, X_test)

            for mid in X_test[sender_idx].index.tolist():
                # Compute predictions
                i_Xte = test_features_df.loc[mid].as_matrix()
                recipients_list = test_features_df.loc[mid].index.tolist()
                mid_pred_probas = i_Xte.flatten()

                # Use the names to compute the best recipients
                unique_r_ab = np.array(recipients_list)
                # Predict first the first names found in the body
                trunc_body = tools.truncate_body(X_test.loc[mid]["body"])
                best_r = np.array(list(), int)
                prediction_outside_ab = []

                for first_name in self.first_names_ab[sender].keys():
                    if first_name in trunc_body:
                        if len(trunc_body[trunc_body.find(first_name):]) < 5:
                            # james.d.steffes and rick.buy@enron.com
                            # tend to write very short mails
                            # and sign Jim or Rick. This is an attempt to
                            # reduce these kind of false positives.
                            continue

                        if first_name == " don " and " don t" in trunc_body:
                            # Obvious false positive
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
                        most_probable_named_r = (
                            np.argmax(recipient_pred_probas)
                        )

                        recipient_idx = recipient_idxs[most_probable_named_r]
                        best_r = np.append(best_r, recipient_idx)
                        mid_pred_probas[recipient_idx] = 0.0

                still_to_predict = 10 - len(best_r)
                pred_list = (
                    np.argsort(mid_pred_probas,
                               kind="mergesort")[::-1][:still_to_predict]
                )
                for pred in pred_list:
                    best_r = np.append(best_r, pred)

                prediction = (
                    (prediction_outside_ab + list(unique_r_ab[best_r]))[:10]
                )

                # Use the frequencies to modify the last predictions
                if len(prediction) == 10:
                    new_predictions = 0
                    most_common_recipients = (
                        self.address_books[sender].most_common()
                    )
                    for recipient, frequency in most_common_recipients:
                        if new_predictions == 2:
                            break
                        if recipient not in prediction[:8]:
                            prediction[8 + new_predictions] = recipient
                            new_predictions += 1

                # Fill the mid_prediction dict
                predictions[mid] = prediction

        return predictions
