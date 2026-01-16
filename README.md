Loan Default Risk Analysis  
--------------------------

Overview  
--------

# Loan Risk Analysis & Default Prediction

This project builds an end-to-end loan risk analysis pipeline using machine learning and Power BI to predict loan defaults and analyze approval decisions.


## Problem Statement
Financial institutions need to assess loan applications while minimizing default risk.  
This project predicts loan default risk and analyzes model decisions using interpretable dashboards.


## Dataset
- Source: Kaggle Loan Prediction dataset
- Size: ~380 loan records
- Target variable:
  - `Loan_Status`
    - `Y` → Non-Default / Approved
    - `N` → Default / Rejected


## Exploratory Data Analysis (EDA)
- Analyzed class imbalance (≈71% non-default, 29% default)
- Examined income distributions, loan amounts, and outliers
- Identified skewness in income-related features


## Feature Engineering
- Total Income = Applicant + Coapplicant Income
- Loan-to-Income ratio
- Log transformation of skewed numerical features
- One-hot encoding of categorical variables


## Modeling
Trained and evaluated multiple models:
- Logistic Regression (scaled)
- Random Forest
- Gradient Boosting

**Evaluation Metrics:**
- ROC–AUC
- Precision, Recall, F1-score
- Emphasis on recall for default (high-risk) cases

**Final Model:** Gradient Boosting  
- ROC–AUC ≈ 0.85  
- Best recall for default detection among tested models


## Power BI Dashboard
Built an interactive Power BI dashboard to:
- Visualize loan approvals vs rejections
- Display confusion matrix (Actual vs Predicted)
- Analyze predicted default risk distribution
- Track KPIs such as total loans, default count, and default rate


Folder Structure  
----------------
Loan-Risk-Analysis/
│
├── data/
│   ├── loan_data_final.csv
│   └── gb_final_predictions.csv
│
├── notebooks/
│   └── loan_risk_model.ipynb
│
├── models/
│   └── gradient_boosting_final.pkl
│
├── powerbi/
│   └── Loan_Risk_PowerBI.pbix
│
└── README.md

Dataset Source  
--------------

The dataset is sourced from Kaggle:  
https://www.kaggle.com/datasets/nikhil1e9/loan-default
