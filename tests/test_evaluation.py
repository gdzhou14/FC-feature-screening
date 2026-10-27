"""Small adapter/metric checks, not a scientific evaluation protocol test."""

import unittest
import numpy as np

from fc_selection import validate_groups, wafer_metrics, select_groups


class EvaluationTests(unittest.TestCase):
    def test_wafer_aggregation_differs_from_global_rmse(self):
        truth = np.array([[1, 2], [2, 4]], dtype=float)
        pred = np.array([[1, 2], [4, 6]], dtype=float)
        scores = wafer_metrics(truth, pred)
        self.assertAlmostEqual(scores["rmse"], 1)
        self.assertNotAlmostEqual(scores["rmse"], np.sqrt(2))
        self.assertAlmostEqual(scores["mae"], 1)
        self.assertAlmostEqual(scores["r2"], -1)
        self.assertAlmostEqual(scores["mape"], 37.5)

    def test_zero_mape_and_shape_guard(self):
        self.assertTrue(np.isnan(wafer_metrics(np.zeros((1, 2)), np.zeros((1, 2)))["mape"]))
        with self.assertRaises(ValueError):
            wafer_metrics(np.zeros((2, 2)), np.zeros((1, 2)))

    def test_selection_columns_metrics_and_first_tie(self):
        instances = []

        class Estimator:
            def fit(self, X, y):
                self.columns = X.shape[1]
                self.value = X[0, 0]
                instances.append(self)
                return self

            def predict(self, X):
                return np.full((len(X), 2), self.value)

        X = np.array([[0, 0, 2, 2], [0, 0, 2, 2]], dtype=float)
        y = np.ones((2, 2))
        groups = validate_groups(X, ("a", "b"), 2)
        result = select_groups(X, y, X, y, ((1,), (0,), (0, 1)), Estimator, groups=groups)
        self.assertEqual(result["groups"], (1,))
        self.assertEqual(result["columns"], (2, 3))
        self.assertIs(result["estimator"], instances[0])
        self.assertEqual([model.columns for model in instances], [2, 2, 4])
        self.assertEqual(len(result["scores"]), 3)

    def test_empty_and_invalid_candidates_rejected_before_fit(self):
        X = np.ones((2, 4))
        y = np.ones((2, 2))
        groups = validate_groups(X, ("a", "b"), 2)

        def forbidden_factory():
            raise AssertionError("must validate before fitting")

        for candidates in ((), ((0,), (2,)), ((1, 0),), ((0, 0),)):
            with self.assertRaises(ValueError):
                select_groups(X, y, X, y, candidates, forbidden_factory, groups=groups)

    def test_lower_rmse_replaces_initial_candidate(self):
        class Estimator:
            def fit(self, X, y):
                self.value = X[0, 0]
                return self

            def predict(self, X):
                return np.full((len(X), 2), self.value)

        X = np.array([[0, 0, 2, 2], [0, 0, 2, 2]], dtype=float)
        y = np.full((2, 2), 2.)
        groups = validate_groups(X, ("a", "b"), 2)
        result = select_groups(X, y, X, y, ((0,), (1,)), Estimator, groups=groups)
        self.assertEqual(result["groups"], (1,))
        self.assertEqual(result["metrics"]["rmse"], 0.)
        self.assertEqual([row["metrics"]["rmse"] for row in result["scores"]], [2., 0.])
