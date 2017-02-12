import unittest
from . import metrics
import warnings


class TestAveragePrecisionAt10(unittest.TestCase):

    def setUp(self):
        self.recipients_prediction = [
            'jack@enron.com',
            'ben@enron.com',
            'maria@enron.com',
            'peter@enron.com',
            'christian@enron.com',
            'marc@enron.com',
            'hugues@enron.com',
            'nathan@enron.com',
            'robin@enron.com',
            'sophie@enron.com',
        ]

    def assertAveragePrecisionEquals(self, true_recipients, p):
        self.assertEqual(
            metrics.average_precision(
                recipients_pred=self.recipients_prediction,
                true_recipients=true_recipients
            ), p
        )

    def test_avg_precision_score_one_length_one(self):
        true_recipients = ['jack@enron.com']
        self.assertAveragePrecisionEquals(true_recipients, 1)

    def test_avg_precision_score_zero_length_one(self):
        true_recipients = ['michelle@enron.com']
        self.assertAveragePrecisionEquals(true_recipients, 0)

    def test_avg_precision_score_order_matters(self):
        true_recipients = ['ben@enron.com']
        self.assertAveragePrecisionEquals(true_recipients, 0.5)

    def test_avg_precision_score_order_matters_length_two(self):
        true_recipients = ['ben@enron.com', 'michelle@enron.com']
        self.assertAveragePrecisionEquals(true_recipients, 0.25)

    def test_avg_precision_complex_score_length_ten(self):
        true_recipients = [
            'ben@enron.com',         # 1: In predictions
            'michelle@enron.com',    # 2: Not in predictions
            'jack@enron.com',        # 3: In predictions
            'sophie@enron.com',      # 4: Not in predictions
            'tarzan@enron.com',      # 5: Not in predictions
            'jane@enron.com',        # 6: Not in predictions
            'christophe@enron.com',  # 7: Not in predictions
            'kevin@enron.com',       # 8: Not in predictions
            'brian@enron.com',       # 9: Not in predictions
            'nigel@enron.com'        # 10: Not in predictions
        ]
        self.assertAveragePrecisionEquals(true_recipients, 0.23)

    def test_avg_precision_complex_score_length_eleven(self):
        true_recipients = [
            'ben@enron.com',         # 1: In predictions
            'michelle@enron.com',    # 2: Not in predictions
            'jack@enron.com',        # 3: In predictions
            'sophie@enron.com',      # 4: Not in predictions
            'tarzan@enron.com',      # 5: Not in predictions
            'jane@enron.com',        # 6: Not in predictions
            'christophe@enron.com',  # 7: Not in predictions
            'kevin@enron.com',       # 8: Not in predictions
            'brian@enron.com',       # 9: Not in predictions
            'nigel@enron.com',       # 10: Not in predictions
            'milie@enron.com'        # 11: Not in predictions
        ]
        self.assertAveragePrecisionEquals(true_recipients, 0.23)


class TestMeanAveragePrecisionAt10(unittest.TestCase):

    def setUp(self):
        self.recipients_predictions = [
            [
                'jack@enron.com',
                'ben@enron.com',
                'maria@enron.com',
                'peter@enron.com',
                'christian@enron.com',
                'marc@enron.com',
                'hugues@enron.com',
                'nathan@enron.com',
                'robin@enron.com',
                'sophie@enron.com',
            ],
            [
                'jack@enron.com',
                'ben@enron.com',
                'maria@enron.com',
                'peter@enron.com',
                'christian@enron.com',
                'marc@enron.com',
                'hugues@enron.com',
                'nathan@enron.com',
                'robin@enron.com',
                'sophie@enron.com',
            ]
        ]

    def assertMeanAveragePrecisionEquals(self, true_recipients, p):
        self.assertEqual(
            metrics.mean_average_precision(
                recipients_preds=self.recipients_predictions,
                true_recipients_list=true_recipients
            ), p
        )

    def test_mean_avg_precision_complex_score_length_ten(self):
        true_recipients_list = [
            [
                'ben@enron.com',         # 1: In predictions
                'michelle@enron.com',    # 2: Not in predictions
                'jack@enron.com',        # 3: In predictions
                'sophie@enron.com',      # 4: Not in predictions
                'tarzan@enron.com',      # 5: Not in predictions
                'jane@enron.com',        # 6: Not in predictions
                'christophe@enron.com',  # 7: Not in predictions
                'kevin@enron.com',       # 8: Not in predictions
                'brian@enron.com',       # 9: Not in predictions
                'nigel@enron.com'        # 10: Not in predictions
            ],
            [
                'jack@enron.com',        # 1: In predictions
                'ben@enron.com',         # 2: In predictions
                'maria@enron.com',       # 3: In predictions
                'peter@enron.com',       # 4: In predictions
                'christian@enron.com',   # 5: In predictions
                'marc@enron.com',        # 6: In predictions
                'hugues@enron.com',      # 7: In predictions
                'nathan@enron.com',      # 8: In predictions
                'robin@enron.com',       # 9: In predictions
                'sophie@enron.com',      # 10: In predictions
            ]
        ]
        self.assertMeanAveragePrecisionEquals(true_recipients_list, (0.23 + 1) / 2)

    def test_mean_avg_precision_can_be_used_as_avg_precision(self):
        self.recipients_predictions = [
            'jack@enron.com',
            'ben@enron.com',
            'maria@enron.com',
            'peter@enron.com',
            'christian@enron.com',
            'marc@enron.com',
            'hugues@enron.com',
            'nathan@enron.com',
            'robin@enron.com',
            'sophie@enron.com',
        ]

        true_recipients_list = [
            'ben@enron.com',         # 1: In predictions
            'michelle@enron.com',    # 2: Not in predictions
            'jack@enron.com',        # 3: In predictions
            'sophie@enron.com',      # 4: Not in predictions
            'tarzan@enron.com',      # 5: Not in predictions
            'jane@enron.com',        # 6: Not in predictions
            'christophe@enron.com',  # 7: Not in predictions
            'kevin@enron.com',       # 8: Not in predictions
            'brian@enron.com',       # 9: Not in predictions
            'nigel@enron.com'        # 10: Not in predictions
        ]

        with warnings.catch_warnings(record=True) as warning:
            self.assertMeanAveragePrecisionEquals(
                true_recipients_list,
                metrics.average_precision(self.recipients_predictions, true_recipients_list)
            )
            self.assertTrue(len(warning) == 1)

if __name__ == "__main__":
    unittest.main()
