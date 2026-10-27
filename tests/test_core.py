"""In-memory synthetic checks only; no experiment data or output files."""

import unittest
from itertools import combinations

import numpy as np

from fc_selection import (
    validate_groups, group_correlations, exclusion_pairs,
    group_shap_importance, explain_groups, valid_combinations,
)


class CoreTests(unittest.TestCase):
    def test_36_combinations_match_independent_exhaustive_oracle(self):
        actual = valid_combinations(7, 0, ((1, 2), (3, 4)))
        expected = set()
        for mask in range(1 << 7):
            selected = tuple(i for i in range(7) if mask & (1 << i))
            if 0 in selected and not ({1, 2} <= set(selected) or {3, 4} <= set(selected)):
                expected.add(selected)
        self.assertEqual(len(actual), 36)
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(list(actual), sorted(actual, key=lambda item: (len(item), item)))
        for size in range(1, 8):
            for selected in combinations(range(7), size):
                if 0 not in selected or {1, 2} <= set(selected) or {3, 4} <= set(selected):
                    self.assertNotIn(selected, actual)

    def test_required_group_conflicts_and_unconstrained_case(self):
        self.assertEqual(valid_combinations(1, 0), ((0,),))
        self.assertEqual(len(valid_combinations(7, 0)), 64)
        self.assertEqual(valid_combinations(3, 0, ((0, 1), (0, 2))), ((0,),))
        for args in ((0, 0, ()), (3, 3, ()), (3, 0, ((1, 1),)), (3, 0, ((1, 3),))):
            with self.assertRaises(ValueError):
                valid_combinations(*args)

    def test_group_validation(self):
        groups = validate_groups(np.zeros((2, 434)))
        self.assertEqual([(g.start, g.stop) for g in groups],
                         [(i * 62, (i + 1) * 62) for i in range(7)])
        for array in (np.zeros((2, 433)), np.empty((0, 434)), np.full((2, 434), np.nan)):
            with self.assertRaises(ValueError):
                validate_groups(array)

    def test_first_sample_only_and_constant_trace_semantics(self):
        X = np.array([[0, 1, 2, 2, 1, 0, 5, 5, 5],
                      [1, 2, 3, 1, 2, 3, 1, 2, 3]], dtype=float)
        groups = validate_groups(X, ("a", "b", "c"), 3)
        corr = group_correlations(X, groups)
        self.assertAlmostEqual(corr[0, 1], -1)
        self.assertTrue(np.isnan(corr[0, 2]))
        X[1] = -100
        np.testing.assert_allclose(group_correlations(X, groups), corr, equal_nan=True)
        self.assertEqual(exclusion_pairs(corr), ((0, 1),))

    def test_strict_absolute_threshold(self):
        corr = np.array([[1, .7, -.71], [.7, 1, np.nan], [-.71, np.nan, 1]])
        self.assertEqual(exclusion_pairs(corr), ((0, 2),))

    def test_signed_cancellation_and_tie(self):
        groups = validate_groups(np.zeros((1, 4)), ("a", "b"), 2)
        values = np.array([[[2, -2], [4, -4], [1, 1], [1, 1]]], dtype=float)
        scores, required = group_shap_importance(values, groups)
        np.testing.assert_allclose(scores, [0, 1])
        self.assertEqual(required, 1)
        time_cancel = np.array([[[2, 2], [-2, -2], [1, 1], [1, 1]]], dtype=float)
        np.testing.assert_allclose(group_shap_importance(time_cancel, groups)[0], [0, 1])
        self.assertEqual(group_shap_importance(np.ones((1, 4, 2)), groups)[1], 0)
        for malformed in (np.zeros((1, 4)), [np.zeros((1, 4))] * 2):
            with self.assertRaises(ValueError):
                group_shap_importance(malformed, groups)

    def test_explainer_adapter_with_synthetic_values(self):
        X = np.zeros((2, 4))
        groups = validate_groups(X, ("a", "b"), 2)
        marker = object()

        class Explainer:
            def __init__(self, estimator):
                self.estimator = estimator

            def shap_values(self, array):
                if self.estimator is not marker:
                    raise AssertionError("wrong estimator")
                return np.ones((*array.shape, 2))

        scores, required = explain_groups(marker, X, groups, Explainer)
        np.testing.assert_allclose(scores, [1, 1])
        self.assertEqual(required, 0)
