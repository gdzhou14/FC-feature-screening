F-C Feature-Group Selection for Photoresist Thickness Prediction

A research implementation of F-C feature-group selection for multi-output photoresist thickness prediction. The method combines SHAP-based feature-group importance, Pearson correlation constraints, and validation-based subset selection with a random forest predictor.

Related paper: An Explainable Wafer-Scale Photoresist Thickness Prediction Model Using F-C Feature-Screened Process Monitoring Data.

Data Availability

The original industrial data are not publicly available due to confidentiality restrictions. This repository provides code to generate synthetic data in memory without saving any data files.

Quick Start

From the repository root, install dependencies:

python -m pip install -r requirements.txt

Run the synthetic demonstration:

python -B -m examples.in_memory_demo

Run the unit tests:

python -B -m unittest discover -s tests -v

Usage

The example at examples/in_memory_demo.py demonstrates the workflow using synthetic data.

For your own data, provide finite numeric arrays with the following shapes:

X: (N, 434), containing seven feature groups with 62 time points per group.

y: (N, 49), containing 49 thickness measurements per wafer.

Here, N is the number of wafers. Call evaluate_fc(X, y) to run the original 80/20 random training/validation split. The returned result contains the selected model and validation metrics.

Input Feature Layout

The 434 input columns are organized into seven consecutive groups of 62 time points, in the following order:

1. Spin_rotation_speed
2. Spin_Exh_Pressure
3. RRC_Nozzle_flow
4. Resist_temp
5. Cup_humidity
6. Motor_flange_temp
7. Cup_temp

Each row represents one wafer, and each target row contains its 49 thickness measurements. Keep the column order, target measurement order, and physical units consistent between training and prediction.

Pass full-width input features to the returned model using result["model"].predict(X), where result is the output of evaluate_fc(X, y). Scaling and column selection are performed internally. Other contiguous, equal-width group layouts can be configured through group_names and points_per_group.

Method

The F-C workflow consists of four main steps:

1. Feature-group importance: Compute SHAP-based importance scores for the input groups.

2. Correlation constraints: Exclude candidate subsets containing feature-group pairs whose absolute Pearson correlation is strictly greater than the threshold (default: 0.7).

3. Candidate subset generation: Enumerate valid subsets, requiring the highest-ranked feature group in every subset.

4. Subset selection: Fit a random forest on the training data for each valid subset and select the subset with the lowest validation RMSE.