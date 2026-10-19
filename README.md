
# Practical Lab 2

## Multivariate Linear Regression, Non-Parametric Models and Cross-Validation

### Student Information

- **Name:** Sukhchain Singh
- **Student ID:** 9111541
- **Course:** Foundations of Machine Learning

## Project Overview

In this lab, I explored different machine learning algorithms to predict diabetes disease progression one year after baseline.

I used the Scikit-Learn Diabetes dataset and compared polynomial regression, decision trees, and kNN regression.

I also performed a separate Logistic Regression classification experiment to explore the classification requirement in the assignment.

My goal was to understand how different models perform, identify overfitting, and compare their ability to generalize to unseen observations.

## Dataset

**Source:** Scikit-Learn Diabetes Dataset

- Total observations: 442
- Input features: 10
- Target: Disease progression score one year after baseline
- Missing values: 0
- Duplicate rows: 0

The features include age, sex, BMI, blood pressure, and six blood measurements.

The input features are scaled by default.

The dataset is loaded directly using `load_diabetes(as_frame=True)`, so no separate CSV file is required.

## Data Preparation

I inspected descriptive statistics, missing values, duplicates, potential outliers, histograms, scatter plots, and correlations.

I kept all observations because the potential outliers were not confirmed to be errors.

The dataset was split into:

- Training: 332 observations (75.11%)
- Validation: 44 observations (9.95%)
- Testing: 66 observations (14.93%)

I used `random_state=42` for reproducibility.

## Models Implemented

### Univariate Polynomial Regression

Six BMI-only polynomial models with degrees 0 through 5.

### Multivariate Polynomial Regression

Two models using all ten features:

- Degree 2
- Degree 3

### Decision Tree Regression

Two models:

- max_depth=3
- max_depth=8

### kNN Regression

Two models:

- k=3
- k=15

### Additional Logistic Regression Experiment

Two binary classification models:

- C=0.1
- C=1.0

For this separate experiment, the training-set median progression score of 141 was used to define two statistical categories.

These categories are not medically validated risk groups. The intended assignment requirement is awaiting professor clarification.

## Evaluation Metrics

For regression:

- R-Squared (R²)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

For classification:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Key Regression Results

| Model | Validation R² | Validation MAE | Validation MAPE |
| --- | ---: | ---: | ---: |
| BMI Polynomial Degree 2 | 0.523 | 47.232 | 40.742% |
| Multivariate Polynomial Degree 2 | 0.436 | 48.451 | 37.346% |
| Decision Tree Depth 3 | 0.527 | 47.656 | 37.148% |
| kNN k=15 | 0.398 | 51.561 | 41.917% |

The Decision Tree depth 3 achieved the highest validation R² and lowest validation MAPE.

The BMI Polynomial degree 2 model achieved the lowest validation MAE.

Therefore, the best model depends on which evaluation metric is prioritized.

## Final BMI Polynomial Test Results

The BMI Polynomial Degree 2 model was selected using validation performance.

Its final test results were:

- Test R²: 0.215
- Test MAE: 54.842
- Test MAPE: 48.386%

These results indicate that BMI alone has limited predictive ability for individual disease progression scores.

## Cross-Validation

I performed an additional 5-Fold Cross-Validation check on the original training data only.

Among the four models examined, kNN with k=15 achieved:

- Mean R²: 0.387
- Mean MAE: 47.992
- Mean MAPE: 42.113%

It had the strongest average performance across the three regression metrics in this check.

The original validation and test sets were not included in cross-validation.

## Main Findings

- More complex models did not always produce better predictions.
- Multivariate Polynomial Degree 3 showed severe generalization problems.
- Decision Tree depth 8 showed strong evidence of overfitting.
- Different evaluation metrics produced different model rankings.
- Cross-validation helped examine model stability.
- The original regression task and additional classification experiment required different evaluation methods.

## Limitations

The dataset contains only 442 observations.

Some models showed substantial prediction errors on unseen data.

The selected BMI Polynomial model performed worse on the final test set than on validation data.

Correlation does not prove causation.

The models have not been clinically validated and should not be used alone for diagnosis, screening, or treatment decisions.

## How to Run

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Create and activate a Python 3.11 Conda environment.
4. Install the dependencies listed in `requirements.txt`.
5. Open `Practical_Lab_2.ipynb`.
6. Select the corresponding Python kernel.
7. Restart the kernel and run all notebook cells from top to bottom.

The notebook loads the diabetes dataset directly from Scikit-Learn.

## Environment and Dependencies

Python 3.11 with:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- ipykernel

Exact dependency versions are recorded in `requirements.txt` to support reproducibility.

I tested the notebook in a fresh Python 3.11 Conda environment by installing the dependencies from `requirements.txt` and running all notebook cells successfully.

## References

1. Scikit-Learn Diabetes Dataset:
   <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html>

2. Performance Metrics Classification Workshop:
   <https://github.com/UbaidUllah777/PerformanceMetricsClassification>

3. K-Nearest Neighbors Workshop:
   <https://github.com/UbaidUllah777/KNearestNeighbors_Workshop>

4. Logistic Regression Classifier Workshop:
   <https://github.com/UbaidUllah777/LogisticRegressionClassifier_Workshop>
