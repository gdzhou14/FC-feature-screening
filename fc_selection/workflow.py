"""Original two-way experiment and low-level training/validation interface.

The default evaluate_fc follows the original two-way experiment with
an automatic 80/20 split and random_state=42.
"""

from dataclasses import dataclass

from .core import (
    FEATURE_NAMES, _matrix, validate_groups, group_correlations,
    exclusion_pairs, explain_groups, valid_combinations,
)
from .evaluation import select_groups, wafer_metrics


def _random_forest():
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=100, random_state=42)


@dataclass
class FCModel:
    """Selected estimator, its scaler and in-memory selection diagnostics."""

    scaler: object
    estimator: object
    columns: tuple
    selection: dict
    correlation: object
    excluded_pairs: tuple
    shap_scores: object
    required_group: int
    scale_selected_only: bool = False

    def predict(self, X):
        """Apply the stored scaler and estimator; never fit or select again."""
        array = _matrix(X, "X")
        if self.scale_selected_only:
            return self.estimator.predict(self.scaler.transform(array[:, self.columns]))
        scaled = self.scaler.transform(array)
        return self.estimator.predict(scaled[:, self.columns])


def fit_fc(X_train, y_train, X_validation, y_validation, *,
           group_names=FEATURE_NAMES, points_per_group=62, threshold=0.7,
           estimator_factory=None, explainer_factory=None):
    """Learn constraints/SHAP on training only; choose subsets on validation.

    Correlations use the first RAW training row. A MinMaxScaler is fitted only
    on training inputs; training/validation receive transform with clip=False.
    A full-group training model is explained on scaled training inputs only.
    The default TreeExplainer uses the training model's tree-path background.
    No target scaling, nested split or train+validation refit is performed.

    This low-level array interface cannot detect duplicate physical samples
    supplied as different arrays. Callers must ensure sample separation.
    Custom factories must be fresh and must not capture held-out data.
    """
    from sklearn.preprocessing import MinMaxScaler

    train, val = _matrix(X_train, "X_train"), _matrix(X_validation, "X_validation")
    ytrain, yval = _matrix(y_train, "y_train"), _matrix(y_validation, "y_validation")
    groups = validate_groups(train, group_names, points_per_group)
    if (train.shape[1] != val.shape[1] or train.shape[0] != ytrain.shape[0]
            or val.shape[0] != yval.shape[0] or ytrain.shape[1] != yval.shape[1]
            or ytrain.shape[1] < 2):
        raise ValueError("train/validation inputs and multi-output targets must align")
    correlation = group_correlations(train, groups)
    pairs = exclusion_pairs(correlation, threshold)
    scaler = MinMaxScaler(clip=False)
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_val = scaler.transform(val)
    factory = _random_forest if estimator_factory is None else estimator_factory
    full_model = factory()
    full_model.fit(scaled_train.copy(), ytrain.copy())
    shap_scores, required = explain_groups(
        full_model, scaled_train.copy(), groups, explainer_factory)
    candidates = valid_combinations(len(groups), required, pairs)
    selection = select_groups(scaled_train, ytrain, scaled_val, yval,
                              candidates, factory, groups=groups)
    return FCModel(scaler, selection["estimator"], selection["columns"],
                   selection, correlation, pairs, shap_scores, required)


def evaluate_fc(X, y, *, group_names=FEATURE_NAMES, points_per_group=62,
                threshold=0.7, estimator_factory=None, explainer_factory=None):
    """Original Relation-Tree.py protocol: full-data scaling and 80/20 split.

    The same 20% holdout is used for SHAP, subset selection and reported metrics.
    These are validation scores, not an independent final test estimate.
    Correlations use the first raw input row. Both splits use random_state=42.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    array, labels = _matrix(X, "X"), _matrix(y, "y")
    groups = validate_groups(array, group_names, points_per_group)
    if len(array) != len(labels) or labels.shape[1] < 2:
        raise ValueError("X and multi-output y must have matching sample counts")
    train, validation = train_test_split(
        np.arange(len(array)), test_size=0.2, random_state=42)
    correlation = group_correlations(array, groups)
    pairs = exclusion_pairs(correlation, threshold)
    full_scaler = MinMaxScaler()
    scaled = full_scaler.fit_transform(array)
    factory = _random_forest if estimator_factory is None else estimator_factory
    full_model = factory()
    full_model.fit(scaled[train], labels[train].copy())
    shap_scores, required = explain_groups(
        full_model, scaled[validation], groups, explainer_factory)
    candidates = valid_combinations(len(groups), required, pairs)
    best, best_scaler, scores = None, None, []
    for candidate in candidates:
        columns = tuple(c for i in candidate
                        for c in range(groups[i].start, groups[i].stop))
        # The original script reloads RAW inputs before its candidate loop.
        scaler = MinMaxScaler()
        selected = scaler.fit_transform(array[:, columns])
        estimator = factory()
        estimator.fit(selected[train], labels[train].copy())
        metrics = wafer_metrics(labels[validation], estimator.predict(selected[validation]))
        if not np.isfinite(metrics["rmse"]):
            raise ValueError("candidate RMSE must be finite")
        scores.append({"groups": candidate, "metrics": metrics})
        if best is None or metrics["rmse"] < best["metrics"]["rmse"]:
            best = {"groups": candidate, "columns": columns,
                    "metrics": metrics, "estimator": estimator}
            best_scaler = scaler
    selection = {**best, "scores": tuple(scores)}
    model = FCModel(best_scaler, best["estimator"], best["columns"],
                    selection, correlation, pairs, shap_scores, required,
                    scale_selected_only=True)
    return {"model": model, "validation_metrics": dict(best["metrics"]),
            "indices": {"train": train, "validation": validation}}
