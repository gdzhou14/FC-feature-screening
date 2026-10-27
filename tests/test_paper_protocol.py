"""Synthetic equivalence checks against a direct original-script RF loop."""
import unittest
from itertools import combinations
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from fc_selection import evaluate_fc


class PaperProtocolTests(unittest.TestCase):
    def test_original_split_scaling_explanation_and_search(self):
        rng = np.random.RandomState(17)
        X = rng.normal(size=(20, 8))
        X[0, 4:] = X[0, :4][[1, 3, 0, 2]]
        y = rng.normal(size=(20, 49))
        explained = []

        class Explainer:
            def __init__(self, model):
                pass

            def shap_values(self, values):
                explained.append(values.copy())
                scores = np.ones((*values.shape, 49))
                scores[:, :4] = 2
                return scores

        result = evaluate_fc(X, y, group_names=("a", "b"),
                             points_per_group=4, explainer_factory=Explainer)
        train, val = train_test_split(np.arange(20), test_size=0.2, random_state=42)
        np.testing.assert_array_equal(result["indices"]["train"], train)
        np.testing.assert_array_equal(result["indices"]["validation"], val)
        self.assertNotIn("test_metrics", result)
        np.testing.assert_array_equal(explained[0], MinMaxScaler().fit_transform(X)[val])
        r = pearsonr(X[0, :4], X[0, 4:])[0]
        self.assertAlmostEqual(result["model"].correlation[0, 1], r)
        candidates = [c for size in (1, 2) for c in combinations(range(2), size)
                      if 0 in c and not (abs(r) > .7 and 0 in c and 1 in c)]
        expected = []
        for candidate in candidates:
            cols = [col for group in candidate for col in range(4*group, 4*(group+1))]
            scaler = MinMaxScaler()
            values = scaler.fit_transform(X[:, cols])
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(values[train], y[train])
            pred = model.predict(values[val])
            rmse = np.sqrt(((y[val]-pred)**2).mean(axis=1)).mean()
            expected.append((float(rmse), candidate, scaler, model, cols))
        best = min(expected, key=lambda entry: entry[0])
        actual = result["model"]
        self.assertEqual(actual.selection["groups"], best[1])
        self.assertEqual([s["groups"] for s in actual.selection["scores"]], candidates)
        for score, entry in zip(actual.selection["scores"], expected):
            self.assertAlmostEqual(score["metrics"]["rmse"], entry[0])
        np.testing.assert_array_equal(actual.scaler.data_min_, X[:, best[4]].min(axis=0))
        np.testing.assert_array_equal(actual.scaler.data_max_, X[:, best[4]].max(axis=0))
        np.testing.assert_allclose(actual.predict(X),
                                   best[3].predict(best[2].transform(X[:, best[4]])))
        self.assertEqual(actual.predict(X).shape, (20, 49))

    def test_invalid_targets_fail_before_factory(self):
        def forbidden():
            raise AssertionError("Training must not start")
        with self.assertRaises(ValueError):
            evaluate_fc(np.ones((10, 434)), np.ones((9, 49)),
                        estimator_factory=forbidden)
