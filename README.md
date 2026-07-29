# 🏦 Bank Term Deposit Subscription Predictor

Predicting whether a bank customer will subscribe to a term deposit — built for CIS6005 (Computational Intelligence), Cardiff Metropolitan University, using Kaggle competition `sch2-reg-2026-d5-1`.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)

Kaggle competition: [Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/sch2-reg-2026-d5-1) 
## Overview

Seven models were built, evaluated, and compared across five technique categories (regression, ensemble bagging, ensemble boosting, SVM, neural network), then deployed as an interactive prediction app with per-prediction SHAP explainability — not just a prediction, but a reason for it.

**Kaggle leaderboard score: 0.96861 (ROC-AUC)**

## Key finding: the leakage trap

EDA surfaced that `duration` (last call length) dominates prediction quality — but it's only known *after* a call happens, making it unusable for real pre-call scoring. This was tested directly, not just asserted:

| Setup | Holdout AUC |
|---|---|
| Logistic Regression, with `duration` | 0.9434 |
| Logistic Regression, **without** `duration` | 0.7839 |

A 0.16 AUC swing from one feature — the deployed model keeps `duration` for leaderboard scoring, but this trade-off is documented, not hidden.

## Model comparison

| Model | Category | Holdout AUC |
|---|---|---|
| Logistic Regression | Regression | 0.9434 |
| Random Forest | Ensemble (bagging) | 0.9498 |
| **XGBoost (tuned)** | Ensemble (boosting) | **0.9676** |
| SVM (RBF, 20k subsample) | Support Vector Machine | 0.9425 |
| Neural Network (MLP) | Neural Network | 0.9627 |

XGBoost was selected as the deployed model — consistent with published findings that gradient-boosted trees typically outperform deep learning on tabular data of this size (Grinsztajn, Oyallon and Varoquaux, 2022).

## The app

Streamlit app takes a customer profile and returns a subscription probability — plus a live SHAP breakdown showing *which factors drove that specific prediction*, not a static, one-size-fits-all explanation. Run it locally (see below) to try it interactively.

## Tech stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `SHAP` · `Streamlit` · `joblib`

## Project structure
```
├── app.py # Streamlit prediction app
├── bank_classification.py # EDA, model training, evaluation
├── best_model_pipeline.joblib # Saved tuned XGBoost pipeline
├── requirements.txt
└── .streamlit/config.toml # App theme
```
## Running it locally

```bash
conda create -n cis6005 python=3.12
conda activate cis6005
pip install -r requirements.txt
streamlit run app.py
```

## Author

Nisith Saranga — BSc (Hons) Software Engineering, Cardiff Metropolitan University
