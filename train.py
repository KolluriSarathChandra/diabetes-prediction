import pandas as pd
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

df = pd.read_csv("data/diabetes.csv")

cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_to_fix:
    df[col] = df[col].replace(0, df[col].median())

FEATURE_NAMES = list(df.drop("Outcome", axis=1).columns)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
cm = confusion_matrix(y_test, preds).tolist()

cv_scores = cross_val_score(model, X, y, cv=5)

print(f"Test accuracy: {acc:.4f}")
print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Save model
joblib.dump(model, "model/diabetes_model.pkl")

# Save real metrics + feature importance so the app never has to invent numbers
metrics = {
    "accuracy": round(acc, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "cv_mean_accuracy": round(cv_scores.mean(), 4),
    "cv_std": round(cv_scores.std(), 4),
    "confusion_matrix": cm,
    "feature_names": FEATURE_NAMES,
    "feature_importance": [round(x, 4) for x in model.feature_importances_.tolist()],
    "test_set_size": len(y_test),
    "train_set_size": len(y_train),
}

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Model saved to model/diabetes_model.pkl")
print("Metrics saved to model/metrics.json")
