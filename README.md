Loan Default Risk Analysis  
--------------------------

Overview  
--------

This project presents an end-to-end analysis and prediction system for loan default risk, using a combination of Python, SQL, Excel, and Power BI. The primary objective is to identify key factors that influence loan approvals and build a machine learning model capable of accurately predicting loan status.

Key Objectives  
--------------

- Build a predictive model to classify loan approval status.
- Explore and preprocess data using Excel and Python.
- Handle class imbalance using SMOTE (Synthetic Minority Oversampling Technique).
- Optimize model performance with hyperparameter tuning using GridSearchCV.
- Visualize patterns and prediction outcomes using interactive Power BI dashboards.
- Query and analyze the dataset using structured SQL queries.

Tools and Technologies  
----------------------

- Python (pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn)
- SQL (PostgreSQL)
- Excel (data cleaning, exploratory analysis)
- Power BI (reporting, visualization)

Machine Learning Summary  
-------------------------

Two classification models were developed: Logistic Regression and Random Forest.  
The data was preprocessed using label encoding and null value handling.  
To address class imbalance in the target variable, SMOTE was applied after feature selection.  
Random Forest was tuned using GridSearchCV with cross-validation to identify optimal hyperparameters.

Key model improvements:
- Balanced recall across classes
- Increased accuracy and F1-score with tuned models
- Improved minority class prediction due to SMOTE

Final model performance was evaluated using accuracy, confusion matrix, and classification report.

Power BI Dashboard  
------------------

The Power BI report highlights relationships between credit history, income, employment status, and loan approvals.  
It includes:
- Actual vs Predicted Loan Status
- Credit History vs Approval Rate
- Income vs Prediction Outcome
- Filters by Gender, Education, and Property Area

The dashboard file is available as a `.pbix` file in the repository, and a preview screenshot is included.  
If the Power BI file cannot be previewed online, a Google Drive link is provided separately.

Folder Structure  
----------------

Loan-Default-Risk-Analysis/  
├── data/  
│   ├── Loan_Default_Cleaned.csv  
│   └── loan_predictions_with_output.csv  
├── notebooks/  
│   └── model_with_smote_and_gridsearch.py  
├── sql/  
│   └── insights.sql  
├── powerbi/  
│   ├── loan_dashboard.pbix  
│   └── dashboard.png  
├── README.md  

Dataset Source  
--------------

The dataset is sourced from Kaggle:  
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

Author  
------

Kumari Archita  
Second-year undergraduate student with interests in data analytics, business intelligence, and applied machine learning.
