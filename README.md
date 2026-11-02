![transit](https://github.com/Tdas-christ/Research_Project/assets/139947958/e40adf6d-a530-49d1-b2bb-275f05ba18b5)

# Beyond Boundaries: Enhancing Exoplanet Discovery with Machine Learning

A comparative study of machine learning and deep learning approaches for classifying Kepler Objects of Interest (KOIs).

## Overview

**Beyond Boundaries** explores the application of machine learning techniques to exoplanet research, with a focus on distinguishing potential planetary candidates from false positives in data collected by NASA's Kepler Space Telescope.

The project combines a review of machine learning applications in exoplanet detection with an experimental comparison of classification algorithms using the Kepler Objects of Interest (KOI) dataset.

The objective is to examine how different learning algorithms perform on astronomical catalogue data and to understand the challenges involved in applying machine learning to exoplanet candidate classification.

## Scientific Background

The transit method is one of the principal techniques used to identify exoplanets. When a planet passes in front of its host star, it causes a small, periodic decrease in the star's observed brightness. These variations are recorded as light curves and analysed to identify potential planetary transits.

However, transit-like signals can also arise from other astrophysical phenomena and instrumental effects. Distinguishing genuine planetary candidates from false positives is therefore an important part of the detection and validation process.

Machine learning offers a way to analyse large astronomical datasets and identify patterns that may help classify these candidates.

## Dataset

The experimental implementation uses the **Kepler Objects of Interest cumulative dataset** (`cumulative.csv`).

The dataset contains catalogue-level measurements and derived parameters associated with Kepler targets, including:

- Orbital period and transit duration
- Transit depth and planet radius
- Stellar effective temperature
- Stellar surface gravity and radius
- Right ascension and declination
- Kepler magnitude

The classification target used in the notebook is `koi_pdisposition`, which distinguishes between:

- `CANDIDATE`
- `FALSE POSITIVE`

The dataset contains approximately 9,564 records and 50 original columns before preprocessing.

**Important:** This experiment uses catalogue-derived features, not the original Kepler light-curve time series. Consequently, it investigates classification using tabular astronomical parameters rather than direct transit-signal detection from raw photometric observations.

## Methodology

The experimental workflow includes the following stages:

1. **Data preprocessing:** inspection of missing values, removal of selected identifier and non-feature columns, and numerical feature preparation.
2. **Exploratory data analysis:** examination of feature distributions, correlations and potential outliers.
3. **Feature preparation:** selection of numerical features for classification.
4. **Model development:** training and comparison of multiple supervised learning algorithms.
5. **Evaluation:** comparison using training accuracy, test accuracy and cross-validation scores.

### Machine Learning Models

The notebook explores the following classifiers:

- Logistic Regression
- Decision Tree
- Random Forest
- Gaussian Naive Bayes
- Extra Tree
- HistGradientBoosting
- XGBoost
- XGBoost Random Forest
- LightGBM
- CatBoost
- AdaBoost
- K-Nearest Neighbours

### Deep Learning Model

A CNN-LSTM architecture was also implemented using TensorFlow and Keras.

The architecture consists of:

- One-dimensional convolutional layers
- Max-pooling layers
- LSTM layers
- Fully connected dense layers
- A sigmoid output layer for binary classification

The network was trained using the Adam optimizer and binary cross-entropy loss.

The CNN-LSTM experiment uses the same catalogue-level numerical features as the traditional classifiers, reshaped into a one-dimensional input sequence. It does not operate on raw Kepler light curves.

## Preliminary Results

The original notebook reported high classification accuracy across several tree-based ensemble models.

These values are retained here as **exploratory results from the initial implementation**, not as validated estimates of generalisation performance.

| Model | Initial test accuracy |
|---|---:|
| XGBoost | 99.09% |
| HistGradientBoosting | 98.95% |
| LightGBM | 98.95% |
| Random Forest | 98.92% |
| CatBoost | 99.06% |
| AdaBoost | 98.89% |
| Logistic Regression | 98.33% |
| K-Nearest Neighbours | 97.46% |
| Gaussian Naive Bayes | 96.41% |

The CNN-LSTM implementation reported an initial test accuracy of approximately 98.85%.

**These results require re-evaluation.** The original implementation contains preprocessing and validation issues, and some catalogue features may encode information closely associated with the Kepler disposition labels. These factors can inflate apparent model performance. The results should not be interpreted as evidence of reliable exoplanet detection until the revised experiments are complete.

## Methodological Limitations and Planned Improvements

The current implementation provides a starting point for comparative analysis, but several methodological aspects need to be addressed.

### 1. Potential target leakage

Features such as `koi_score` and the Kepler false-positive flags may contain information closely associated with the disposition process used to assign the target label. Their influence must be investigated, and leakage-prone features should be excluded from the primary evaluation where appropriate.

### 2. Preprocessing leakage

Missing-value imputation was originally performed before splitting the dataset. This allows information from the test set to influence preprocessing.

The revised implementation will fit imputers only on training data, using an appropriate preprocessing pipeline.

### 3. Cross-validation leakage

Feature scaling was performed before cross-validation in the original implementation. The revised workflow will place scaling and other preprocessing steps inside the model pipeline so that each validation fold is processed independently.

### 4. Evaluation improvements

The revised experiments will use stratified splitting and cross-validation, alongside metrics beyond accuracy, including:

- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC
- Confusion matrices

### 5. Deep learning architecture

The CNN-LSTM model currently receives tabular features reshaped as a sequence. Since these features do not inherently represent an ordered temporal signal, the suitability of this architecture needs further investigation.

The binary prediction and confusion-matrix implementation will also be corrected.

## Technologies Used

- **Programming:** Python
- **Data analysis:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn, XGBoost, LightGBM, CatBoost
- **Deep learning:** TensorFlow, Keras
- **Environment:** Jupyter Notebook

## Repository Structure

```text
Beyond-Boundaries/
│
├── Research.ipynb
├── cumulative.csv
├── README.md
└── requirements.txt
```

The dataset is not necessarily included in the repository. Obtain the appropriate Kepler KOI cumulative catalogue and place `cumulative.csv` in the project directory before running the notebook.

## Running the Project

1. Clone the repository.

```bash
git clone <repository-url>
cd Beyond-Boundaries
```

2. Install the required dependencies.

```bash
pip install -r requirements.txt
```

3. Download the Kepler KOI cumulative dataset and place `cumulative.csv` in the project directory.

4. Open the Jupyter Notebook.

```bash
jupyter notebook
```

5. Run the notebook cells sequentially.

Update the dataset path in the notebook if the CSV is stored in a different location.

## Future Work

Potential extensions of this work include:

- Developing leakage-resistant classification pipelines.
- Investigating feature selection and feature importance in greater depth.
- Evaluating models using raw Kepler light-curve observations.
- Exploring convolutional neural networks for transit-signal analysis.
- Investigating transfer learning and domain adaptation across Kepler and TESS observations.
- Studying model robustness under differences in observational conditions and data quality.

## Research

**Project:** Beyond Boundaries: Enhancing Exoplanet Discovery with Machine Learning

**Author:** Tanisha Das

**Affiliation:** Department of Computer Science, CHRIST (Deemed to be University)

