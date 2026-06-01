# 🔐 Credit Card Fraud Detection — Robust ML Pipeline

> Automatic detection of fraudulent bank transactions using a production-grade Machine Learning pipeline with zero data leakage.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Pipeline Architecture](#️-pipeline-architecture)
- [Models & Optimization](#-models--optimization)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-Installation)
- [Usage](#-usage)
- [Key Design Decisions](#-key-design-decisions)
- [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

Credit card fraud represents a major financial threat to banking institutions. This project implements a **complete, production-ready Machine Learning pipeline** for **automatically detecting fraudulent transactions** with a strong **precision/recall tradeoff**, applicable in near-real-time contexts.

**Target variable:** `Class` — a binary variable:
- `0` → Legitimate transaction
- `1` → Fraudulent transaction

---

## 📊 Dataset

| Attribute | Details |
|---|---|
| **Source** | [Credit Card Fraud Detection — Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Published by** | Université Libre de Bruxelles (ULB) |
| **Period** | September 2013 — 2 days of European transactions |
| **Total transactions** | 284,807 |
| **Fraudulent transactions** | 492 (0.172%) |
| **Class imbalance ratio** | 577:1 |
| **Missing values** | None |

**Features:**
- `Time` — Seconds elapsed since the first transaction
- `V1`–`V28` — Anonymized PCA components (protects banking privacy)
- `Amount` — Transaction amount
- `Class` — Target variable (0 = legitimate, 1 = fraud)

**Why this dataset?**
- ✅ Real banking transactions → credible results
- ✅ Community ML benchmark → comparable to prior research
- ✅ Extreme imbalance (577:1) → realistic business scenario, ideal for testing SMOTE
- ✅ No data cleaning required → focus on modeling
- ✅ PCA anonymization respects data privacy regulations

---

## 🏗️ Pipeline Architecture

```
Raw Data
    │
    ├─ ColumnTransformer
    │      ├─ RobustScaler         → V1..V28 + Amount  (resistant to outliers)
    │      └─ FunctionTransformer  → Time (sin/cos cyclical encoding)
    │
    ├─ SMOTE  (applied on X_train only — no leakage)
    │
    └─ Classifier (LogisticRegression / XGBoost)
           └─ GridSearchCV (scoring = f1_macro, StratifiedKFold k=5)
```

Two pipelines were used:

1. **`imblearn.Pipeline`** — Combines the `ColumnTransformer` (RobustScaler + cyclical Time encoding) with SMOTE, guaranteeing that no transformation is fitted on the test set (anti-leakage rule).

2. **`GridSearchCV` with `StratifiedKFold (k=5)`** — Wraps the above pipeline to optimize hyperparameters of both models (Logistic Regression and XGBoost), scored on `f1_macro`.

### ⚠️ Zero Data Leakage Rule

> **All transformations (fit) are learned exclusively on `X_train`. `X_test` is only transformed (transform), never fitted.**

---

## 🤖 Models & Optimization

| Model | Role | Key Parameters Tuned |
|---|---|---|
| **Logistic Regression** | Interpretable baseline | `C`, `solver`, `max_iter` |
| **XGBoost** | High-performance classifier | `n_estimators`, `max_depth`, `learning_rate`, `subsample` |

- **Resampling:** SMOTE (Synthetic Minority Over-sampling Technique) applied exclusively on training data
- **Cross-validation:** StratifiedKFold (k=5) — preserves fraud/legitimate ratio in each fold
- **Scoring metric:** `f1_macro` — appropriate for class imbalance

---

## 📈 Results

| Criterion | Status |
|---|---|
| sklearn Pipeline | ✅ |
| ColumnTransformer (RobustScaler + Cyclical) | ✅ |
| GridSearchCV (f1_macro, StratifiedKFold) | ✅ |
| Models: LR (baseline) + XGBoost | ✅ |
| SMOTE (imblearn.Pipeline) | ✅ |
| Zero data leakage | ✅ |
| Metrics: F1, AUC, Confusion Matrix, ROC curve | ✅ |
| Feature importance | ✅ |
| Optimized decision threshold | ✅ |
| `joblib` serialization | ✅ |

**Best model performance (on test set):**
- AUC-ROC > **0.97**
- F1-macro > **0.90**
- Fraud Recall > **0.80** (< 20% missed frauds)
- Inference: **< 1ms per transaction** (XGBoost)

The best pipeline is serialized with `joblib` for deployment:
```python
joblib.dump(best_pipeline, 'fraud_detector_best_pipeline.pkl')
```

---

## 📁 Project Structure

```
fraud-detection-pipeline/
│
├── fraud_detection_pipeline.ipynb   # Main notebook — full ML pipeline
├── fraud_detection_scenario.html    # Interactive scenario visualization
├── fraud_detection_presentation.pptx  # Project presentation slides
├── fraud_detector_best_pipeline.pkl # Serialized best model (generated on run)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

> ⚠️ The dataset `creditcard.csv` is **not included** in this repository due to its size (~150 MB). Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root directory.

---

## ⚙️ Installation

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fraud-detection-pipeline.git
cd fraud-detection-pipeline

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the full pipeline

```bash
jupyter notebook fraud_detection_pipeline.ipynb
```

Run all cells in order. The notebook will:
1. Load and explore the dataset (EDA)
2. Split data (stratified 80/20)
3. Build and fit the imblearn pipeline
4. Optimize hyperparameters with GridSearchCV
5. Evaluate both models (LR and XGBoost)
6. Compare results and identify the best model
7. Optimize the decision threshold
8. Serialize the best pipeline

### Real-time inference (after training)

```python
import joblib
import pandas as pd

# Load serialized model
model = joblib.load('fraud_detector_best_pipeline.pkl')

# Predict on a single transaction
transaction = pd.DataFrame([...])  # Your transaction features
pred_class = model.predict(transaction)[0]
pred_proba = model.predict_proba(transaction)[0, 1]

print(f"Predicted class: {pred_class} | Fraud probability: {pred_proba:.4f}")
print("⚠️ FRAUD DETECTED" if pred_class == 1 else "✅ Legitimate")
```

---

## 🔑 Key Design Decisions

| Decision | Justification |
|---|---|
| **RobustScaler** over StandardScaler | Resistant to extreme transaction amounts (outliers) |
| **Cyclical encoding of Time** (sin/cos) | Preserves temporal periodicity without leakage |
| **SMOTE inside the pipeline** | Prevents synthetic samples from polluting the test set |
| **f1_macro as scoring metric** | Balances performance on both classes despite 577:1 imbalance |
| **Stratified splits everywhere** | Preserves fraud ratio in train, validation, and test sets |

---

## 🔮 Future Improvements

**1. Explainability (XAI)**
- SHAP values for per-transaction feature attribution
- LIME for local decision explanation (GDPR compliance)
- Monitoring dashboard (Evidently, Grafana)

**2. Real-time Deployment**
- REST API with FastAPI or Flask
- Apache Kafka + Flink for streaming ingestion
- Adaptive decision threshold based on business cost

**3. Technical Enhancements**
- Isolation Forest / AutoEncoder for unsupervised anomaly detection
- Bayesian optimization (Optuna) for more efficient hyperparameter search
- Ensemble stacking: LR + XGBoost + RandomForest
- Feature engineering: Amount/user_median ratio, hourly frequency

**4. Business Considerations**
- Asymmetric cost integration: cost(FN) >> cost(FP)
- Periodic model retraining (concept drift)
- Human-in-the-loop validation for uncertain predictions (probability 0.4–0.6)

---

## 📄 License

This project is for educational purposes. The dataset is subject to [Kaggle's terms of use](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

*Project: Automatic Fraud Detection — Robust ML Pipeline*
