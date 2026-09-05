import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json

# Load dataset
df = pd.read_csv("heart.csv", sep=";")

# Drop id column
df = df.drop(columns=["id"])

# Convert age from days to years
df["age"] = (df["age"] / 365).round().astype(int)

# Calculate BMI
df["bmi"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(2)

# Remove outliers in blood pressure
df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 250)]
df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 200)]
df = df[df["ap_hi"] > df["ap_lo"]]

# Remove outliers in height and weight
df = df[(df["height"] >= 100) & (df["height"] <= 220)]
df = df[(df["weight"] >= 30) & (df["weight"] <= 200)]

# Remove outliers in BMI
df = df[(df["bmi"] >= 10) & (df["bmi"] <= 60)]

print("Shape after cleaning:", df.shape)
print("Class distribution:\n", df["cardio"].value_counts())

# Features and target
X = df.drop(columns=["cardio"])
y = df["cardio"]

print("\nFeatures:", X.columns.tolist())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)

# Train Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Evaluate both
results = {
    "Logistic Regression": {
        "Accuracy": round(accuracy_score(y_test, lr_preds), 4),
        "Precision": round(precision_score(y_test, lr_preds), 4),
        "Recall": round(recall_score(y_test, lr_preds), 4),
        "F1 Score": round(f1_score(y_test, lr_preds), 4)
    },
    "Random Forest": {
        "Accuracy": round(accuracy_score(y_test, rf_preds), 4),
        "Precision": round(precision_score(y_test, rf_preds), 4),
        "Recall": round(recall_score(y_test, rf_preds), 4),
        "F1 Score": round(f1_score(y_test, rf_preds), 4)
    }
}

# Print results
for model, metrics in results.items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

# Save comparison results
with open("comparison_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("\ncomparison_results.json saved!")

# Save feature names
feature_names = X.columns.tolist()
with open("feature_names.json", "w") as f:
    json.dump(feature_names, f)
print("feature_names.json saved!")

# Save best model and scaler
joblib.dump(rf, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("heart_model.pkl and scaler.pkl saved!")