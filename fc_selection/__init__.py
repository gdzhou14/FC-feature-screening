"""F-C candidate primitives. Import performs no data loading or training."""

from .core import (
    FEATURE_NAMES, validate_groups, group_correlations, exclusion_pairs,
    group_shap_importance, explain_groups, valid_combinations,
)
from .evaluation import wafer_metrics, select_groups
from .workflow import FCModel, fit_fc, evaluate_fc

__all__ = [
    "FEATURE_NAMES", "validate_groups", "group_correlations", "exclusion_pairs",
    "group_shap_importance", "explain_groups", "valid_combinations",
    "wafer_metrics", "select_groups",
    "FCModel", "fit_fc", "evaluate_fc",
]
