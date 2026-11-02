# Beyond Boundaries: Enhancing Exoplanet Discovery with Machine Learning

## Overview

The discovery and validation of exoplanets involve analysing large volumes of astronomical observations and distinguishing genuine planetary signals from false-positive detections. Machine learning offers a useful approach to this classification task by identifying patterns in the physical and observational properties of planetary candidates.

**Beyond Boundaries** explores the application of supervised machine learning to the classification of Kepler Objects of Interest (KOIs). Using the Kepler cumulative KOI catalogue, the project evaluates multiple classification algorithms to distinguish objects labelled as *CANDIDATE* from those labelled as *FALSE POSITIVE*.

The project focuses on model comparison, data preprocessing, leakage-aware evaluation, and the challenges of applying machine learning to astronomical catalogues.

> **Scope:** This project predicts the disposition recorded in the Kepler catalogue. It does not independently confirm the existence of exoplanets, perform new planetary discoveries, or analyse raw Kepler photometric time-series data.

## Research Objectives

- Explore the application of machine learning to exoplanet-candidate classification.
- Investigate the predictive value of stellar and planetary catalogue parameters.
- Compare the performance of multiple supervised classification algorithms.
- Implement an evaluation strategy that accounts for objects associated with the same host star.
- Examine the limitations of catalogue-based classification and identify directions for future research.

## Dataset

The project uses the **Kepler Cumulative Objects of Interest (KOI) catalogue**, provided in `cumulative.csv`.

The dataset contains 9,564 records and 50 columns in the version used for this project. It includes information about KOIs, their host stars, transit characteristics, stellar parameters, and catalogue dispositions.

Examples of relevant parameters include:

- `koi_period`: Orbital period of the candidate.
- `koi_duration`: Transit duration.
- `koi_depth`: Transit depth.
- `koi_prad`: Estimated planetary radius.
- `koi_srad`: Stellar radius.
- `koi_steff`: Stellar effective temperature.
- `koi_slogg`: Stellar surface gravity.
- `koi_kepmag`: Kepler-band magnitude.

### Target variable

The prediction target is `koi_pdisposition`, which represents the disposition assigned by the Kepler data-analysis process.

| Label | Encoded value |
|---|---:|
| FALSE POSITIVE | 0 |
| CANDIDATE | 1 |

The target is a catalogue classification, not an independent ground-truth determination of planetary status.

### Feature selection

Identifiers and disposition-related fields are excluded from the predictor set. In particular, the implementation excludes fields such as `koi_disposition`, `koi_score`, the Kepler false-positive flags, `koi_tce_plnt_num`, and `kepid`.

The host-star identifier, `kepid`, is retained separately for group-aware splitting and is not used as a model feature.

## Methodology

The workflow consists of the following stages:

1. **Data loading and exploration:** Load the cumulative KOI catalogue and inspect its structure, distributions, and missing values.
2. **Target preparation:** Encode the two disposition classes for binary classification.
3. **Feature construction:** Select numerical catalogue features while excluding identifiers and disposition-derived fields.
4. **Missing-value handling:** Apply median imputation within model pipelines.
5. **Feature scaling:** Apply standardisation within pipelines where appropriate.
6. **Group-aware evaluation:** Use `kepid` to ensure that records associated with the same host star are not distributed across training and validation/test partitions.
7. **Model training and comparison:** Train and evaluate the selected classification algorithms.
8. **Performance analysis:** Examine classification metrics, cross-validation results, and confusion matrices.

### Group-aware data splitting

A key consideration is that multiple KOIs may be associated with the same host star. A conventional random row-wise split could therefore place objects from one host star in both training and testing sets.

To reduce this risk, the project uses **StratifiedGroupKFold** with `kepid` as the grouping variable. This preserves class proportions as far as the grouping constraints permit while keeping host-star groups separate across folds.

The same grouping principle is used for the holdout split and the cross-validation procedure.

### Preprocessing and leakage prevention

Imputation and scaling are incorporated into scikit-learn pipelines. This ensures that preprocessing parameters are learned from the relevant training partition rather than from the complete dataset before splitting.

Disposition-related columns that could reveal information about the target are excluded from the predictor set to reduce target leakage.

## Machine Learning Models

The notebook evaluates a collection of supervised classification algorithms, including:

- Logistic Regression
- Decision Tree
- Random Forest
- Gaussian Naive Bayes
- Extra Tree
- Histogram-based Gradient Boosting
- AdaBoost
- K-Nearest Neighbours
- XGBoost and XGBoost Random Forest, when the optional dependency is available
- LightGBM, when the optional dependency is available
- CatBoost, when the optional dependency is available

The models are evaluated using a common set of classification metrics.

## Deep Learning Approach

The project also explores a **Multi-Layer Perceptron (MLP)** as a neural-network-based classification approach.

The MLP is trained on the numerical catalogue features. Its preprocessing is fitted using the training subset, and a group-aware validation partition is used during training. Early stopping is used to monitor validation performance and help limit overfitting.

The MLP is treated as a separate neural-network experiment. Its evaluation setup should be considered when comparing its results with those of the classical machine learning models.

## Evaluation Metrics

The project uses the following metrics:

- **Accuracy:** Overall proportion of correctly classified records.
- **Precision:** Proportion of predicted candidates that belong to the positive class.
- **Recall:** Proportion of actual candidates correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability to distinguish between the two classes across classification thresholds.
- **PR-AUC / Average Precision:** Precision-recall performance, particularly useful when class distributions are imbalanced.

Cross-validation results are used for model comparison and selection. The held-out test partition is reserved for final performance evaluation.

## Limitations

- The analysis uses catalogue-derived numerical features rather than raw Kepler light curves.
- The target represents the disposition recorded in the Kepler catalogue and should not be interpreted as independent confirmation of planetary status.
- Catalogue labels and measurements may contain uncertainties and biases inherited from the original observation and vetting process.
- Group-aware splitting reduces potential dependence between records from the same host star, but does not eliminate every possible source of dataset bias.
- The MLP and classical models use different training and validation procedures, which should be considered when interpreting comparisons.
- Performance on the Kepler catalogue does not, by itself, establish generalisation to other missions or unseen astronomical populations.

## Repository Structure

```text
Beyond-Boundaries/
│
├── Dataset/
│   └── cumulative.csv
│
├── Research_code.ipynb
│
├── README.md
│
└── requirements.txt
```

The filenames above describe the intended repository organisation. Update them if the final GitHub repository uses different names.

## Installation and Execution

Clone the repository:

```bash
git clone <repository-url>
cd Beyond-Boundaries
```

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Optional gradient-boosting libraries used by the notebook can be installed separately:

```bash
pip install xgboost lightgbm catboost
```

TensorFlow is required for the MLP experiment:

```bash
pip install tensorflow
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the research notebook and execute the cells in sequence. Ensure that `cumulative.csv` is available in the expected `Dataset/` directory.

## Future Directions

Potential extensions of this work include:

- Incorporating Kepler photometric time-series data and transit light curves.
- Exploring time-series models for transit-signal analysis.
- Investigating feature importance and explainability techniques.
- Evaluating models on data from other exoplanet surveys, such as TESS.
- Exploring approaches that account for uncertainty in astronomical measurements.
- Investigating transfer learning and multimodal approaches combining catalogue parameters with light-curve information.

## Research

**Project:** Beyond Boundaries: Enhancing Exoplanet Discovery with Machine Learning

This project investigates the use of machine learning for the classification of Kepler Objects of Interest and forms part of a broader interest in the intersection of artificial intelligence, machine learning, and astrophysics.
