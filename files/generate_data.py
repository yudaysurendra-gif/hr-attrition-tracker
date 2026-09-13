"""
generate_data.py
Generates a synthetic HR dataset for the HR Attrition Tracker project.
Replace this with your real HR dataset (same column names) when available.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000  # number of employees

departments = ["Sales", "R&D", "HR", "IT", "Finance"]
job_roles = ["Executive", "Manager", "Analyst", "Associate", "Lead"]

df = pd.DataFrame({
    "EmployeeID": range(1, N + 1),
    "Age": np.random.randint(21, 60, N),
    "Department": np.random.choice(departments, N, p=[0.30, 0.25, 0.10, 0.20, 0.15]),
    "JobRole": np.random.choice(job_roles, N),
    "MonthlyIncome": np.random.randint(20000, 150000, N),
    "YearsAtCompany": np.random.randint(0, 25, N),
    "DistanceFromHome": np.random.randint(1, 40, N),
    "JobSatisfaction": np.random.randint(1, 5, N),       # 1 (low) - 4 (high)
    "WorkLifeBalance": np.random.randint(1, 5, N),        # 1 (bad) - 4 (great)
    "PerformanceRating": np.random.randint(1, 5, N),
    "OverTime": np.random.choice(["Yes", "No"], N, p=[0.30, 0.70]),
    "TrainingTimesLastYear": np.random.randint(0, 6, N),
})

# Build a simple, realistic-ish attrition probability from the features
risk = (
    (df["JobSatisfaction"] <= 2).astype(int) * 0.25
    + (df["WorkLifeBalance"] <= 2).astype(int) * 0.20
    + (df["OverTime"] == "Yes").astype(int) * 0.20
    + (df["YearsAtCompany"] < 2).astype(int) * 0.15
    + (df["DistanceFromHome"] > 20).astype(int) * 0.10
    + (df["MonthlyIncome"] < 35000).astype(int) * 0.15
    + np.random.normal(0, 0.08, N)
)

df["Attrition"] = np.where(risk > 0.55, "Yes", "No")

out_path = "data/hr_data.csv"
df.to_csv(out_path, index=False)
print(f"Generated {N} records -> {out_path}")
print(df["Attrition"].value_counts(normalize=True).rename("proportion"))
