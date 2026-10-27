"""Explicit-array selection adapter; no automatic split or scaling protocol."""

from .core import _group_layout, _matrix


def wafer_metrics(y_true, y_pred):
    """Original per-wafer RMSE/R2/MAE/MAPE, then average across wafers.

    Zero labels are omitted for MAPE. All-zero wafers do not contribute to MAPE;
    if none contribute, MAPE is NaN. R2 uses sklearn's default constant handling.
    Output width is generic rather than hard-coded to 49.
    """
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    truth, pred = _matrix(y_true, "y_true"), _matrix(y_pred, "y_pred")
    if truth.shape != pred.shape or truth.shape[1] < 2:
        raise ValueError("matching target arrays with at least two outputs are required")
    rmse, r2, mae, mape = [], [], [], []
    for actual, predicted in zip(truth, pred):
        rmse.append(np.sqrt(mean_squared_error(actual, predicted)))
        r2.append(r2_score(actual, predicted))
        mae.append(mean_absolute_error(actual, predicted))
        nonzero = actual != 0
        if np.any(nonzero):
            mape.append(np.mean(np.abs((actual[nonzero] - predicted[nonzero])
                                      / actual[nonzero])) * 100)
    return {"rmse": float(np.mean(rmse)), "r2": float(np.mean(r2)),
            "mae": float(np.mean(mae)),
            "mape": float(np.mean(mape)) if mape else float("nan")}


def select_groups(X_train, y_train, X_validation, y_validation,
                  candidates, estimator_factory, *, groups):
    """Fit fresh fit/predict estimators; select strictly lowest validation RMSE.

    Data partitions and preprocessing are supplied by the caller. This adapter
    never splits, scales, explains, refits on validation, or accesses a test set.
    It is not an end-to-end reproduction of the original test-based search.
    Equal RMSE retains the first candidate. Return its fitted estimator, group
    indices and column indices, plus in-memory scores for all candidates.
    A factory can supply RF or another estimator/pipeline implementing fit/predict.
    """
    import numpy as np
    from numbers import Integral

    train, val = _matrix(X_train, "X_train"), _matrix(X_validation, "X_validation")
    ytrain, yval = _matrix(y_train, "y_train"), _matrix(y_validation, "y_validation")
    if (train.shape[1] != val.shape[1] or train.shape[0] != ytrain.shape[0]
            or val.shape[0] != yval.shape[0] or ytrain.shape[1] != yval.shape[1]
            or ytrain.shape[1] < 2):
        raise ValueError("train/validation inputs and multi-output targets must align")
    groups, _ = _group_layout(groups, train.shape[1])
    # Validate all candidates before starting any estimator training.
    prepared = []
    for candidate in candidates:
        candidate = tuple(candidate)
        if (not candidate or any(isinstance(i, bool) or not isinstance(i, Integral)
                                 or not 0 <= i < len(groups) for i in candidate)
                or len(set(candidate)) != len(candidate)
                or tuple(sorted(candidate)) != candidate):
            raise ValueError("candidate must contain ascending unique valid group indices")
        columns = tuple(c for i in candidate for c in range(groups[i].start, groups[i].stop))
        prepared.append((candidate, columns))
    if not prepared:
        raise ValueError("at least one candidate is required")
    best = None
    scores = []
    for candidate, columns in prepared:
        estimator = estimator_factory()
        # Advanced indexing yields copies, isolating input feature arrays.
        estimator.fit(train[:, columns], ytrain.copy())
        metrics = wafer_metrics(yval, estimator.predict(val[:, columns]))
        if not np.isfinite(metrics["rmse"]):
            raise ValueError("candidate RMSE must be finite")
        scores.append({"groups": candidate, "metrics": metrics})
        if best is None or metrics["rmse"] < best["metrics"]["rmse"]:
            best = {"groups": candidate, "columns": columns,
                    "metrics": metrics, "estimator": estimator}
    return {**best, "scores": tuple(scores)}
