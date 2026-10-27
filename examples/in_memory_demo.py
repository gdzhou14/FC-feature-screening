"""Synthetic API demonstration only; no industrial data and no file output.

Run from the candidate repository root with:
    python -B -m examples.in_memory_demo

Requires the real SHAP dependency and compatible 3D multi-output SHAP layout.
This example does not replace SHAP with synthetic importance scores.
"""


def make_synthetic_data(seed=21):
    """Create (40, 434) inputs and (40, 49) labels entirely in memory.

    These dimensionless random values are not a physical process simulation.
    The workflow applies the original random 80/20 split.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    X = rng.normal(size=(40, 7 * 62))
    summaries = X.reshape(40, 7, 62).mean(axis=2)
    spatial_weights = rng.normal(size=(3, 49))
    y = 10 + summaries[:, (0, 2, 6)] @ spatial_weights
    y += rng.normal(scale=0.02, size=y.shape)
    return X, y


def run_demo():
    """Return in-memory evaluation, using actual RF and actual TreeExplainer.

    No arrays, metrics, predictions, models or plots are written to disk.
    Missing/incompatible dependencies raise errors; no packages are installed.
    """
    from importlib.util import find_spec

    if find_spec("shap") is None:
        raise ModuleNotFoundError(
            "The full synthetic demo requires SHAP, which is not installed. "
            "Prepare a separately approved compatible environment; "
            "this example never installs dependencies automatically.",
            name="shap",
        )
    from fc_selection import evaluate_fc

    X, y = make_synthetic_data()
    return evaluate_fc(X, y)


def main():
    result = run_demo()
    # Report only completion metadata, never array rows or prediction values.
    print("Synthetic RF/F-C workflow completed; no output files were written.")
    print("This is an API demonstration, not reproduction of paper results.")
    print("Selected group count:", len(result["model"].selection["groups"]))


if __name__ == "__main__":
    main()
