# 🩺 Diabetes Prediction System

> A machine learning system for diabetes risk prediction using XGBoost and SHAP explainability, trained on 100,000 clinical patient records and deployed as a real-time Streamlit web application.

## 🔍 Overview

Diabetes mellitus is a chronic disease affecting over 537 million adults worldwide. Early detection is critical to preventing complications such as cardiovascular disease, kidney failure, and neuropathy.

This project builds a machine learning classification system that predicts diabetes risk from patient clinical data. Five models were trained and compared. XGBoost was selected as the final model based on empirical performance. SHAP analysis provides transparent, clinically interpretable explanations for every prediction. The system is deployed as an interactive web application using Streamlit.

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — iammustafatz/diabetes-prediction-dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) |
| Total Records | 100,000 |
| Features | 8 clinical + demographic |
| Target | `diabetes` (0 = Non-Diabetic, 1 = Diabetic) |
| Class Distribution | 91.5% Non-Diabetic / 8.5% Diabetic |

### Features

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Male / Female |
| `age` | Numerical | Age in years |
| `hypertension` | Binary | 0 = No, 1 = Yes |
| `heart_disease` | Binary | 0 = No, 1 = Yes |
| `smoking_history` | Categorical | never / former / current / unknown |
| `bmi` | Numerical | Body Mass Index |
| `HbA1c_level` | Numerical | Glycated haemoglobin (%) |
| `blood_glucose_level` | Numerical | Fasting blood glucose (mg/dL) |

---

## 📁 Project Structure

```
ML MINI PROJECT - DIABETES PREDICTION/
│
├── assets/                          ← Charts and visualisations
│
├── notebook/                        ← Full Colab Notebook
│
├── app.py                           ← Streamlit web application
├── best_diabetes_model.pkl          ← Saved XGBoost model
├── diabetes_prediction_dataset.csv  ← Dataset
├── le_gender.pkl                    ← Gender encoder
├── le_smoking.pkl                   ← Smoking encoder
├── README.md                        ← Project documentation
├── requirements.txt                 ← Dependencies
└── scaler.pkl                       ← Saved scaler
```

## ⚙️ ML Pipeline

```
Raw Dataset (100,000 rows)
        │
        ▼
1. Exploratory Data Analysis
   └── Class distribution, histograms, correlation heatmap, box plots
        │
        ▼
2. Data Preprocessing
   ├── Smoking history: 6 categories → 4 clean categories
   ├── Gender: removed 18 'Other' rows
   ├── Dropped duplicate rows
   ├── Label encoding (gender, smoking_history)
   └── StandardScaler (all features)
        │
        ▼
3. Train / Test Split  (80% / 20%, stratified)
        │
        ▼
4. SMOTE — applied on training set ONLY
   └── Balanced 8.5% → 50/50 (training only, test untouched)
        │
        ▼
5. Feature Selection
   └── All 8 features retained (clinically justified)
        │
        ▼
6. Model Training — 5 models
   ├── Logistic Regression  (baseline)
   ├── Random Forest        (bagging ensemble)
   ├── XGBoost              (boosting — WINNER)
   ├── Soft Voting          (RF + XGBoost)
   └── Stacking             (RF + XGBoost → LR meta)
        │
        ▼
7. Evaluation
   └── ROC-AUC, F1, Precision, Recall, Confusion Matrix
        │
        ▼
8. SHAP Explainability
   └── Summary plot, dependence plots, waterfall plot
        │
        ▼
9. Streamlit Deployment
```

## 📈 Model Results

| Model | ROC-AUC | Train-AUC | Gap | F1 Score | Precision | Recall |
|---|---|---|---|---|---|---|
| **XGBoost** ⭐ | **0.9757** | 0.9982 | 0.0225 | 0.8003 | 0.8908 | 0.7264 |
| Soft Voting | 0.9749 | 1.0000 | 0.0250 | 0.7903 | 0.8450 | 0.7423 |
| Stacking | 0.9741 | 1.0000 | 0.0259 | 0.7775 | 0.8071 | 0.7500 |
| Random Forest | 0.9692 | 1.0000 | 0.0308 | 0.7587 | 0.7545 | 0.7630 |
| Logistic Regression | 0.9616 | 0.9631 | 0.0016 | 0.5759 | 0.4265 | 0.8862 |

**XGBoost selected as final model** — highest ROC-AUC on held-out test set.

> Ensemble models did not outperform XGBoost because all models learn the same dominant pattern (HbA1c + blood glucose). XGBoost, being a boosting ensemble of 100 trees, already captures the maximum learnable signal from the data.

---

## 🔬 SHAP Explainability

SHAP (SHapley Additive exPlanations) was applied to explain why the model makes each prediction.

**Feature Importance Ranking (SHAP):**

1. `HbA1c_level` — dominant predictor
2. `blood_glucose_level` — dominant predictor
3. `age` — moderate importance
4. `bmi` — moderate importance
5. `hypertension` — low importance
6. `heart_disease` — low importance
7. `smoking_history` — very low importance
8. `gender` — near zero importance

SHAP dependence plots confirm the model learned clinically valid thresholds — HbA1c ≥ 6.5% and blood glucose ≥ 140 mg/dL trigger strong positive predictions, consistent with WHO diagnostic criteria.

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | Preprocessing, models, metrics |
| XGBoost | Final prediction model |
| imbalanced-learn | SMOTE for class imbalance |
| SHAP | Model explainability |
| Matplotlib / Seaborn | Visualisations |
| Streamlit | Web application deployment |
| Joblib | Model serialisation |
| Google Colab | Training environment |

---

## ⚠️ Limitations

- **Unverified data source** — the dataset is community-contributed on Kaggle with no documented hospital or institution origin
- **Class imbalance** — only 8.5% diabetic records; SMOTE was used to address this on training data only
- **Clinical shortcut** — HbA1c and blood glucose are themselves the medical definition of diabetes, making the classification task inherently easier than real-world screening scenarios
- **Not generalisable** — the dataset's population demographics are not documented; model performance on specific ethnic groups or age ranges is unknown
- **Not for clinical use** — this system is for educational purposes only and must not be used for actual medical diagnosis

---

## 📄 Disclaimer

> This project is developed for academic and educational purposes. The predictions generated by this system are not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.