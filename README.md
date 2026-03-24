# Credit Default Prediction Using Machine Learning

**Comparing Logistic Regression, Decision Tree, Random Forest, and XGBoost on the Kaggle "Give Me Some Credit" Dataset**

Intro to AI Coursework — University of Essex, 2025

---

## Overview

This project builds and compares machine learning models to predict whether a borrower will experience serious financial distress within two years. It uses the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset from Kaggle, containing 150,000 borrower records with financial and demographic features.

The project covers the full ML pipeline: data cleaning, feature engineering, handling severe class imbalance with SMOTE, model training, evaluation, and comparison.

## Key Results

| Model | Accuracy | AUC-ROC | F1-Score (Default) |
|---|---|---|---|
| Random Forest | 90.68% | 0.828 | 0.364 |
| **XGBoost** | **92.13%** | **0.836** | **0.356** |

**Key findings:**
- XGBoost achieved the highest accuracy (92.13%) and AUC-ROC (0.836), demonstrating strong discriminatory power between defaulters and non-defaulters
- Both models struggled with the minority class (defaults) due to severe class imbalance (93.3% non-default vs 6.7% default), despite SMOTE oversampling
- AUC-ROC proved a more reliable evaluation metric than accuracy for this imbalanced classification problem
- Feature engineering (Debt-to-Income ratio, Credit Utilization) added meaningful signal to the models

## Dataset

The "Give Me Some Credit" dataset contains 150,000 records with 11 features:

| Feature | Description |
|---|---|
| `SeriousDlqin2yrs` | Target variable — person experienced 90+ days past due (binary) |
| `RevolvingUtilizationOfUnsecuredLines` | Total balance on credit cards / credit limits |
| `age` | Age of borrower |
| `NumberOfTime30-59DaysPastDueNotWorse` | Number of times 30–59 days past due |
| `DebtRatio` | Monthly debt payments / gross monthly income |
| `MonthlyIncome` | Monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Number of open loans and credit lines |
| `NumberOfTimes90DaysLate` | Number of times 90+ days past due |
| `NumberRealEstateLoansOrLines` | Number of mortgage and real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Number of times 60–89 days past due |
| `NumberOfDependents` | Number of dependents in family |

## Methodology

```
Raw Data (150K records)
    │
    ├── Data Cleaning (median imputation, drop index column)
    │
    ├── Feature Engineering (DebtToIncome, CreditUtilizationRatio)
    │
    ├── Train/Test Split (80/20)
    │
    ├── SMOTE Oversampling (balance classes: 110K → 110K each)
    │
    ├── Feature Scaling (StandardScaler)
    │
    ├── Model Training
    │   ├── Random Forest (100 trees, balanced class weights)
    │   └── XGBoost (gradient boosting, scale_pos_weight=1)
    │
    └── Evaluation (Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix)
```

### Preprocessing
- **Missing values:** MonthlyIncome (29,731 missing) and NumberOfDependents (3,924 missing) imputed with median values
- **Infinite values:** Replaced with NaN and imputed
- **Feature scaling:** StandardScaler applied to all features
- **Class imbalance:** SMOTE applied to training data, balancing default/non-default from 8,014/110,685 to 110,685/110,685

### Models Compared
- **Logistic Regression** — linear baseline for binary classification
- **Decision Tree** — captures non-linear relationships, easy to interpret
- **Random Forest** — ensemble of 100 decision trees with balanced class weights
- **XGBoost** — gradient boosted trees with sequential error correction

## Repository Structure

```
├── notebooks/
│   └── Credit_Default_Prediction.ipynb    # Full analysis pipeline
├── data/
│   ├── cs-training.csv                    # Training data (150K records)
│   ├── cs-test.csv                        # Test data
│   ├── sampleEntry.csv                    # Sample submission format
│   └── Data_Dictionary.xls               # Feature descriptions
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup & Usage

### Prerequisites
- Python 3.8+

### Installation

```bash
git clone https://github.com/hurairaqamar/credit-default-prediction.git
cd credit-default-prediction
pip install -r requirements.txt
```

### Running the Analysis

```bash
jupyter notebook notebooks/Credit_Default_Prediction.ipynb
```

The notebook runs end-to-end in Google Colab or locally with Jupyter.

## Limitations

- **Class imbalance remains challenging:** Despite SMOTE, both models struggled to accurately predict the minority class (defaults), with F1-scores around 0.36
- **Limited feature set:** The anonymised Kaggle dataset lacks detailed financial variables that would be available in production credit scoring systems
- **No hyperparameter tuning:** Models used largely default parameters; GridSearch/RandomizedSearch could improve performance
- **Single dataset:** Results may not generalise to other credit populations or lending markets

## Future Work

- Apply hyperparameter tuning (GridSearchCV) to optimise model performance
- Experiment with cost-sensitive learning as an alternative to SMOTE
- Add SHAP analysis for model explainability and feature importance
- Test on real-world credit datasets with richer feature sets
- Explore threshold tuning to optimise precision/recall trade-off for the default class

## Tools & Technologies

Python, Pandas, NumPy, scikit-learn, XGBoost, LightGBM, SMOTE (imbalanced-learn), Matplotlib, Seaborn

## Author

**Huraira Qamar**
- MSc Applied Data Science — University of Essex
- [LinkedIn](https://www.linkedin.com/in/huraira-qamar)

## License

This project is for educational and portfolio purposes. The dataset is from the [Kaggle "Give Me Some Credit" competition](https://www.kaggle.com/c/GiveMeSomeCredit).
