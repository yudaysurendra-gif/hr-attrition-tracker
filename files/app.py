"""
app.py
HR Attrition Tracker - Streamlit Dashboard

Run with:
    streamlit run app.py

Features:
 - Upload your own HR CSV or use the bundled sample dataset
 - KPI summary (headcount, attrition rate, avg tenure/income)
 - Attrition breakdown by department, overtime, job satisfaction
 - Trains a Random Forest attrition-prediction model on the fly
 - Feature importance chart (what's driving attrition)
 - "What-if" form to score a single employee's attrition risk
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from attrition_model import (
    load_data, preprocess, train_model, get_feature_importance, predict_single
)

st.set_page_config(page_title="HR Attrition Tracker", layout="wide")

st.title("📊 HR Attrition Tracker")
st.caption("Monitor workforce attrition risk and explore the key drivers behind it.")

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload HR CSV", type=["csv"])
    st.caption("No file? The app uses the bundled sample dataset (data/hr_data.csv).")

try:
    df = load_data(uploaded) if uploaded is not None else load_data("data/hr_data.csv")
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# 2. KPI summary
# ---------------------------------------------------------------------------
total_employees = len(df)
attrition_count = (df["Attrition"] == "Yes").sum()
attrition_rate = attrition_count / total_employees * 100
avg_tenure = df["YearsAtCompany"].mean() if "YearsAtCompany" in df.columns else None
avg_income = df["MonthlyIncome"].mean() if "MonthlyIncome" in df.columns else None

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Employees", f"{total_employees:,}")
k2.metric("Employees Left", f"{attrition_count:,}")
k3.metric("Attrition Rate", f"{attrition_rate:.1f}%")
k4.metric("Avg. Tenure (yrs)", f"{avg_tenure:.1f}" if avg_tenure is not None else "N/A")

st.divider()

# ---------------------------------------------------------------------------
# 3. Breakdown charts
# ---------------------------------------------------------------------------
st.subheader("Attrition Breakdown")
c1, c2, c3 = st.columns(3)

def attrition_rate_by(col):
    return (
        df.groupby(col)["Attrition"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

with c1:
    if "Department" in df.columns:
        st.markdown("**By Department**")
        rates = attrition_rate_by("Department")
        fig, ax = plt.subplots()
        rates.plot(kind="bar", ax=ax, color="#4C72B0")
        ax.set_ylabel("Attrition Rate (%)")
        ax.set_xlabel("")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

with c2:
    if "OverTime" in df.columns:
        st.markdown("**By OverTime**")
        rates = attrition_rate_by("OverTime")
        fig, ax = plt.subplots()
        rates.plot(kind="bar", ax=ax, color="#DD8452")
        ax.set_ylabel("Attrition Rate (%)")
        ax.set_xlabel("")
        st.pyplot(fig)

with c3:
    if "JobSatisfaction" in df.columns:
        st.markdown("**By Job Satisfaction (1-4)**")
        rates = attrition_rate_by("JobSatisfaction")
        fig, ax = plt.subplots()
        rates.sort_index().plot(kind="bar", ax=ax, color="#55A868")
        ax.set_ylabel("Attrition Rate (%)")
        ax.set_xlabel("Job Satisfaction")
        st.pyplot(fig)

with st.expander("View raw data"):
    st.dataframe(df, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 4. Model training + feature importance
# ---------------------------------------------------------------------------
st.subheader("Predictive Model: What Drives Attrition?")

try:
    X, y, encoders = preprocess(df)
    model, metrics, _ = train_model(X, y)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    m2.metric("Precision", f"{metrics['precision']*100:.1f}%")
    m3.metric("Recall", f"{metrics['recall']*100:.1f}%")
    m4.metric("F1 Score", f"{metrics['f1']*100:.1f}%")

    fi = get_feature_importance(model, X)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(fi["feature"][:8][::-1], fi["importance"][:8][::-1], color="#8172B2")
    ax.set_xlabel("Relative Importance")
    ax.set_title("Top Attrition Drivers")
    st.pyplot(fig)

except Exception as e:
    st.warning(f"Model could not be trained on this dataset: {e}")
    model, encoders = None, None

st.divider()

# ---------------------------------------------------------------------------
# 5. What-if: score a single employee
# ---------------------------------------------------------------------------
st.subheader("🔍 Check an Employee's Attrition Risk")

if model is not None:
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 65, 30)
            department = st.selectbox("Department", sorted(df["Department"].unique()))
            job_role = st.selectbox("Job Role", sorted(df["JobRole"].unique())) if "JobRole" in df.columns else "Analyst"
        with col2:
            income = st.number_input("Monthly Income", 10000, 300000, 45000, step=1000)
            years = st.number_input("Years at Company", 0, 40, 3)
            distance = st.number_input("Distance From Home (km)", 0, 100, 10)
        with col3:
            job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            wlb = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
            overtime = st.selectbox("OverTime", ["Yes", "No"])

        submitted = st.form_submit_button("Predict Attrition Risk")

    if submitted:
        employee = {
            "Age": age, "Department": department, "JobRole": job_role,
            "MonthlyIncome": income, "YearsAtCompany": years, "DistanceFromHome": distance,
            "JobSatisfaction": job_sat, "WorkLifeBalance": wlb,
            "PerformanceRating": 3, "OverTime": overtime, "TrainingTimesLastYear": 2,
        }
        # keep only columns the model was trained on
        employee = {k: v for k, v in employee.items() if k in X.columns}
        result = predict_single(model, encoders, employee)

        if result["prediction"] == "Yes":
            st.error(f"⚠️ High attrition risk — probability {result['attrition_probability']*100:.1f}%")
        else:
            st.success(f"✅ Low attrition risk — probability {result['attrition_probability']*100:.1f}%")
else:
    st.info("Upload a dataset with an 'Attrition' column to enable predictions.")
