# Transaction Fraud Detection

A machine learning project that detects fraudulent credit card transactions using classification models, with a focus on handling severe class imbalance.

## Overview

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), containing 284,807 transactions where only 492 (0.17%) are fraudulent. The goal is to build and compare models that can reliably catch fraud despite this extreme imbalance.

## Dataset

- 284,807 transactions, 31 features (Time, Amount, V1–V28 PCA-transformed features, Class)
- Class distribution: 284,315 legitimate (99.83%) vs. 492 fraud (0.17%)
- No missing values

## Approach

1. **Exploratory Data Analysis** — visualized class imbalance and compared transaction amounts between fraud and legit transactions
2. **Preprocessing** — scaled `Amount` and `Time` (other features were already PCA-scaled), used a stratified train/test split to preserve the fraud ratio in both sets
3. **Modeling** — trained and compared three classifiers:
   - Logistic Regression (baseline, `class_weight="balanced"`)
   - Random Forest (`class_weight="balanced"`, 100 trees)
   - XGBoost (`scale_pos_weight` tuned to imbalance ratio)
4. **Evaluation** — used precision, recall, F1-score, and confusion matrices rather than accuracy, since accuracy is misleading on imbalanced data

## Results

| Model | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) |
|---|---|---|---|
| Logistic Regression | 0.06 | 0.92 | 0.11 |
| Random Forest | 0.91 | 0.79 | 0.84 |
| **XGBoost** | 0.87 | 0.83 | **0.85** |

**XGBoost performed best overall**, offering the strongest balance between catching fraud and minimizing false alarms. Logistic Regression, while simple and interpretable, produced far too many false positives (1,389) to be practical, despite having the highest recall.

## Key takeaway

Accuracy is a poor metric for imbalanced problems like fraud detection — a model predicting "not fraud" every time would still be 99.8% accurate. Precision and recall, and the tradeoff between them, matter far more in practice.

## Tech stack

- Python, pandas, NumPy
- scikit-learn (Logistic Regression, Random Forest, preprocessing, metrics)
- XGBoost
- Matplotlib, Seaborn (visualization)

## How to run

\`\`\`bash
pip install -r requirements.txt
python src/main.py
\`\`\`

## Project structure

\`\`\`
Transaction-Fraud-Detection/
├── data/               # dataset (not included, download from Kaggle)
├── src/
│   └── main.py         # full pipeline: load, explore, preprocess, train, evaluate
├── requirements.txt
└── README.md
\`\`\`