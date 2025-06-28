import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Loan_Default_Cleaned.csv")

df.dropna(inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X.copy(), y.copy(), test_size=0.2, random_state=42)

X_train["Set"] = "Train"
X_test["Set"] = "Test"
X_combined = pd.concat([X_train, X_test]).sort_index()
df["Data_Split"] = X_combined["Set"]

# Try two models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

results = {}

for name, model in models.items():
    model.fit(X_train.drop("Set", axis=1), y_train)
    preds = model.predict(X_test.drop("Set", axis=1))
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, preds))

# Compare model performance
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

best_model = models[max(results, key=results.get)]

df["Predicted"] = best_model.predict(X)
df["Probability_Approved"] = best_model.predict_proba(X)[:, 1]

df["Actual_Label"] = y.map({1: "Approved", 0: "Rejected"})
df["Predicted_Label"] = df["Predicted"].map({1: "Approved", 0: "Rejected"})

export_df = df[[
    "ApplicantIncome", "LoanAmount", "Credit_History", "Education", "Self_Employed",
    "Actual_Label", "Predicted_Label", "Probability_Approved", "Data_Split"
]]

export_df.to_csv("loan_predictions_full.csv", index=False)
print("✅ File saved: loan_predictions_full.csv")
