# HR Attrition Tracker

A simple, self-contained project that tracks employee attrition and predicts
attrition risk using machine learning.

## What it does

- Loads HR employee data (a synthetic sample dataset is included)
- Shows KPIs: headcount, attrition rate, average tenure
- Breaks down attrition by department, overtime status, and job satisfaction
- Trains a Random Forest model to predict which employees are likely to leave
- Shows the top factors driving attrition (feature importance)
- Lets you enter a single employee's details and get a real-time attrition
  risk score

## Project structure

```
hr_attrition_tracker/
├── app.py               # Streamlit dashboard (the main app)
├── attrition_model.py   # Data loading, preprocessing, ML model, prediction
├── generate_data.py     # Creates the synthetic sample dataset
├── data/
│   └── hr_data.csv      # Sample HR dataset (1000 employees)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the dashboard

```bash
streamlit run app.py
```

This opens the dashboard in your browser (usually http://localhost:8501).

## Use your own data

Upload any CSV with these columns (case-sensitive) via the sidebar:

| Column                | Type        | Example |
|-----------------------|-------------|---------|
| Age                   | number      | 34      |
| Department            | text        | Sales   |
| JobRole               | text        | Manager |
| MonthlyIncome         | number      | 55000   |
| YearsAtCompany        | number      | 4       |
| DistanceFromHome      | number      | 12      |
| JobSatisfaction       | number 1-4  | 3       |
| WorkLifeBalance       | number 1-4  | 2       |
| PerformanceRating     | number 1-4  | 3       |
| OverTime              | Yes/No      | Yes     |
| TrainingTimesLastYear | number      | 2       |
| Attrition             | Yes/No      | No      |

(`Attrition` is the target column and is required. Extra columns are ignored.)

## Regenerate the sample dataset

```bash
python generate_data.py
```

## Extend this base project

Ideas to build on top of this starter:
- Swap Random Forest for XGBoost / Logistic Regression and compare metrics
- Add SHAP values for per-employee explainability
- Connect to a live HR database instead of CSV upload
- Add email/Slack alerts when an employee crosses a risk threshold
- Deploy on Azure App Service or Streamlit Community Cloud
