import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Loan_Default_Cleaned.csv")
df.dropna(inplace=True)
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

models = {}

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
models["Logistic Regression"] = lr

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
models["Random Forest (Tuned)"] = best_rf

results = {}
for name, model in models.items():
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, preds))

sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Accuracy Comparison (SMOTE + GridSearchCV)")
plt.ylabel("Accuracy")
plt.show()

best_model_name = max(results, key=results.get)
final_model = models[best_model_name]

df["Actual_Status"] = y
df["Predicted_Status"] = final_model.predict(X)
df["Predicted_Label"] = df["Predicted_Status"].map({1: "Approved", 0: "Rejected"})
df["Actual_Label"] = df["Actual_Status"].map({1: "Approved", 0: "Rejected"})

df.to_csv("loan_predictions_with_output.csv", index=False)
print("CSV saved as 'loan_predictions_with_output.csv'")
print(f"Best Model: {best_model_name}")
