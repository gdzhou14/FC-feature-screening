"""F-C primitives preserving the original first-sample and signed-mean rules.

Functions consume in-memory arrays only. Numerical dependencies are imported
inside functions; importing the package does not load data or train a model.
"""

from itertools import combinations
from numbers import Integral


FEATURE_NAMES = (
    "Spin_rotation_speed", "Spin_Exh_Pressure", "RRC_Nozzle_flow",
    "Resist_temp", "Cup_humidity", "Motor_flange_temp", "Cup_temp",
)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _matrix(values, name):
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or 0 in array.shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite 2D numeric array")
    return array


def _group_layout(groups, n_columns):
    """Require ordered equal-width slices covering every input column."""
    groups = tuple(groups)
    if not groups:
        raise ValueError("at least one group is required")
    width = None
    end = 0
    for group in groups:
        if (not isinstance(group, slice) or group.step not in (None, 1)
                or not isinstance(group.start, Integral)
                or not isinstance(group.stop, Integral)
                or group.start != end or group.stop <= group.start):
            raise ValueError("groups must be contiguous ordered slices")
        size = group.stop - group.start
        if width is not None and size != width:
            raise ValueError("groups must have equal widths")
        width, end = size, group.stop
    if end != n_columns:
        raise ValueError("groups must cover all input columns exactly")
    return groups, int(width)


def validate_groups(X, group_names=FEATURE_NAMES, points_per_group=62):
    """Return contiguous group slices; reject malformed input, never truncate."""
    array = _matrix(X, "X")
    names = tuple(group_names)
    width = _positive_integer(points_per_group, "points_per_group")
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("group_names must be nonempty unique strings")
    if array.shape[1] != len(names) * width:
        raise ValueError("input width must equal number of groups * points_per_group")
    return tuple(slice(i * width, (i + 1) * width) for i in range(len(names)))


def group_correlations(X_reference, groups):
    """Pearson matrix from ONLY row zero, as in the original research script.

    Constant time traces retain undefined (NaN) pair correlations. This matches
    the original threshold comparison, where NaN does not create an exclusion.
    No cross-wafer aggregation or significance filtering is performed.
    """
    import numpy as np
    from scipy.stats import pearsonr

    array = _matrix(X_reference, "X_reference")
    groups, width = _group_layout(groups, array.shape[1])
    if width < 2:
        raise ValueError("Pearson correlation requires at least two time points")
    result = np.eye(len(groups), dtype=float)
    for i, j in combinations(range(len(groups)), 2):
        left, right = array[0, groups[i]], array[0, groups[j]]
        value = (float("nan") if np.ptp(left) == 0 or np.ptp(right) == 0
                 else float(pearsonr(left, right)[0]))
        result[i, j] = result[j, i] = value
    return result


def exclusion_pairs(correlation, threshold=0.7):
    """Return upper-triangle pairs with strict abs(Pearson r) > threshold.

    NaN is allowed and does not exclude a pair, preserving original semantics.
    """
    import numpy as np

    matrix = np.asarray(correlation, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] == 0
            or matrix.shape[0] != matrix.shape[1]
            or np.isinf(matrix).any()
            or not np.allclose(matrix, matrix.T, equal_nan=True)
            or np.any(np.abs(matrix[np.isfinite(matrix)]) > 1 + 1e-12)):
        raise ValueError("correlation must be a symmetric square Pearson matrix")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    return tuple((i, j) for i, j in combinations(range(len(matrix)), 2)
                 if abs(matrix[i, j]) > threshold)


def group_shap_importance(shap_values, groups):
    """Return scores and first argmax group using the original signed means.

    Accept only explicit (samples, input columns, outputs) layout. Average
    outputs, then time points, then take abs and average samples. Do not silently
    infer axes or convert legacy lists of output-specific SHAP arrays.
    """
    import numpy as np

    if isinstance(shap_values, (list, tuple)):
        raise ValueError("SHAP layout must be an explicit 3D array, not a legacy list")
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 3 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("SHAP values must have finite (samples, columns, outputs) layout")
    groups, width = _group_layout(groups, values.shape[1])
    signed = values.mean(axis=2).reshape(values.shape[0], len(groups), width).mean(axis=2)
    scores = np.abs(signed).mean(axis=0)
    return scores, int(np.argmax(scores))


def explain_groups(estimator, X_explain, groups, explainer_factory=None):
    """Compute SHAP on explicitly supplied data, then aggregate group scores.

    Defaults to TreeExplainer, imported only when called. A caller-supplied
    factory must return an object exposing shap_values(X). No data is selected
    here; callers must keep final test data out of feature selection.
    """
    array = _matrix(X_explain, "X_explain")
    groups, _ = _group_layout(groups, array.shape[1])
    if explainer_factory is None:
        from shap import TreeExplainer

        explainer_factory = TreeExplainer
    values = explainer_factory(estimator).shap_values(array)
    return group_shap_importance(values, groups)


def valid_combinations(n_groups, required_group, excluded_pairs=()):
    """Size-first lexicographic subsets containing the required group."""
    count = _positive_integer(n_groups, "n_groups")
    if (isinstance(required_group, bool) or not isinstance(required_group, Integral)
            or not 0 <= required_group < count):
        raise ValueError("required_group must be a valid group index")
    pairs = []
    for pair in excluded_pairs:
        pair = tuple(pair)
        if (len(pair) != 2 or pair[0] == pair[1]
                or any(isinstance(i, bool) or not isinstance(i, Integral)
                       or not 0 <= i < count for i in pair)):
            raise ValueError("each excluded pair must contain two distinct valid indices")
        pairs.append(pair)
    return tuple(comb for size in range(1, count + 1)
                 for comb in combinations(range(count), size)
                 if required_group in comb
                 and all(not (a in comb and b in comb) for a, b in pairs))
