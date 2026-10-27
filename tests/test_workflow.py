"""Low-level training/validation adapter check using a synthetic explainer."""

import unittest

import numpy as np

from fc_selection import fit_fc


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(21)
        self.X = rng.uniform(1, 10, (12, 6))
        self.y = rng.uniform(1, 5, (12, 3))
        self.options = dict(group_names=("a", "b"), points_per_group=3)
        self.events = []
        events = self.events

        class Estimator:
            def fit(self, X, y):
                events.append(("fit", X.copy(), y.copy()))
                self.mean = y.mean(axis=0)
                return self

            def predict(self, X):
                events.append(("predict", X.copy()))
                return np.tile(self.mean, (len(X), 1))

        class Explainer:
            def __init__(self, estimator):
                self.estimator = estimator

            def shap_values(self, X):
                events.append(("explain", X.copy()))
                values = np.ones((*X.shape, getattr(self.estimator, "n_outputs_", 3)))
                values[:, :3, :] = 2
                return values

        self.options.update(estimator_factory=Estimator, explainer_factory=Explainer)

    def test_low_level_fit_has_no_test_argument_and_predict_never_refits(self):
        model = fit_fc(self.X[:6], self.y[:6], self.X[6:9], self.y[6:9], **self.options)
        fits_before = sum(event[0] == "fit" for event in self.events)
        model.predict(self.X[9:])
        self.assertEqual(sum(event[0] == "fit" for event in self.events), fits_before)
