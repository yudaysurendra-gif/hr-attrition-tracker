"""
attrition_model.py
Core logic for the HR Attrition Tracker:
 - data loading / validation
 - preprocessing
 - model training (Random Forest)
 - prediction helpers
 - feature importance
Kept separate from the UI (app.py) so it can be reused or unit-tested on its own.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

CATEGORICAL_COLS = ["Department", "JobRole", "OverTime"]
TARGET_COL = "Attrition"
DROP_COLS = ["EmployeeID"]


def load_data(path_or_buffer) -> pd.DataFrame:
    """Load a CSV (path or file-like/uploaded buffer) into a DataFrame."""
    df = pd.read_csv(path_or_buffer)
    required = {"Attrition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {missing}")
    return df


def preprocess(df: pd.DataFrame):
    """
    Encode categorical columns and split features/target.
    Returns: X (features), y (target), encoders (dict of fitted LabelEncoders)
    """
    data = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()

    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

    target_le = LabelEncoder()
    y = target_le.fit_transform(data[TARGET_COL].astype(str))  # No=0, Yes=1 typically
    encoders[TARGET_COL] = target_le

    X = data.drop(columns=[TARGET_COL])
    return X, y, encoders


def train_model(X, y, test_size=0.25, random_state=42):
    """Train a RandomForestClassifier and return the model + evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=random_state, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return model, metrics, (X_test, y_test, y_pred)


def get_feature_importance(model, X) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances."""
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def predict_single(model, encoders, employee: dict) -> dict:
    """
    Predict attrition risk for a single employee record.
    `employee` is a dict with the same feature columns used in training.
    Returns dict with prediction label and probability of attrition ('Yes').
    """
    row = pd.DataFrame([employee])

    for col in CATEGORICAL_COLS:
        if col in row.columns and col in encoders:
            le = encoders[col]
            # handle unseen categories gracefully
            row[col] = row[col].astype(str).apply(
                lambda v: v if v in le.classes_ else le.classes_[0]
            )
            row[col] = le.transform(row[col])

    proba = model.predict_proba(row)[0]
    target_le = encoders[TARGET_COL]
    yes_index = list(target_le.classes_).index("Yes") if "Yes" in target_le.classes_ else 1
    prob_yes = proba[yes_index]
    pred_label = "Yes" if prob_yes >= 0.5 else "No"

    return {"prediction": pred_label, "attrition_probability": round(float(prob_yes), 3)}


if __name__ == "__main__":
    # Quick smoke test when run directly: python attrition_model.py
    df = load_data("data/hr_data.csv")
    X, y, encoders = preprocess(df)
    model, metrics, _ = train_model(X, y)

    print("Model performance on held-out test set:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.3f}")
    print("  confusion_matrix:")
    print(metrics["confusion_matrix"])

    print("\nTop feature importances:")
    print(get_feature_importance(model, X).head(8).to_string(index=False))

    sample_employee = {
        "Age": 29, "Department": "Sales", "JobRole": "Executive",
        "MonthlyIncome": 32000, "YearsAtCompany": 1, "DistanceFromHome": 25,
        "JobSatisfaction": 2, "WorkLifeBalance": 2, "PerformanceRating": 3,
        "OverTime": "Yes", "TrainingTimesLastYear": 1,
    }
    print("\nSample prediction:", predict_single(model, encoders, sample_employee))
