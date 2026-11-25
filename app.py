
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Gender Bias Audit", page_icon="⚖️", layout="wide")

st.title("⚖️ AI-Based Employee Monitoring")
st.subheader("Gender Bias Audit Dashboard")
st.write("Comparative analysis of rigid and improved employee productivity monitoring.")

st.info("Upload the two CSV datasets to begin the analysis.")

col1, col2 = st.columns(2)

with col1:
    keystroke_file = st.file_uploader("Upload Fixed-Text CSV", type="csv")

with col2:
    demographics_file = st.file_uploader("Upload Demographics CSV", type="csv")

if keystroke_file and demographics_file:

    df_k = pd.read_csv(keystroke_file)
    df_d = pd.read_csv(demographics_file)

    df_k.columns = df_k.columns.str.strip()
    df_d.columns = df_d.columns.str.strip()

    if "participant" not in df_k.columns or "participant" not in df_d.columns:
        st.error("Both datasets must contain a participant column.")
        st.stop()

    if "gender" not in df_d.columns or "repetition" not in df_k.columns:
        st.error("Required gender or repetition column is missing.")
        st.stop()

    if len(df_k.columns) <= 49:
        st.error("The keystroke dataset does not contain the expected time column.")
        st.stop()

    time_col = df_k.columns[49]

    df = pd.merge(df_k, df_d, on="participant", how="inner")

    df["total_time"] = pd.to_numeric(df[time_col], errors="coerce")
    df["repetition"] = pd.to_numeric(df["repetition"], errors="coerce")

    df = df.dropna(subset=["total_time", "gender", "repetition"])
    df = df[df["total_time"] > 0].copy()

    df.loc[df["gender"].str.lower() == "female", "total_time"] += 1200

    df["task_complete_rate"] = np.where(df["repetition"] >= 1, 1.0, 0.0)
    df["true_performance_metric"] = df["task_complete_rate"].astype(int)

    # Model 1: Rigid Monitoring
    X1 = df[["total_time"]]

    y1 = np.where(
        df["total_time"] > 300,
        0,
        df["true_performance_metric"]
    )

    model1 = RandomForestClassifier(n_estimators=50, random_state=47)
    model1.fit(X1, y1)

    df["linear_verdict"] = model1.predict(X1)

    # Model 2: Improved Monitoring
    median_baseline = df["total_time"].median()

    df["adjusted_time"] = np.where(
        df["total_time"] > 300,
        median_baseline,
        df["total_time"]
    )

    df["active_typing_efficiency"] = 46 / df["adjusted_time"]

    X2 = df[["active_typing_efficiency", "task_complete_rate"]]
    y2 = df["true_performance_metric"]

    model2 = RandomForestClassifier(n_estimators=50, random_state=3)
    model2.fit(X2, y2)

    df["solved_verdict"] = model2.predict(X2)

    st.success("Analysis completed successfully!")

    st.header("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Participants", f"{len(df):,}")
    c2.metric("Male", f"{(df['gender'].str.lower() == 'male').sum():,}")
    c3.metric("Female", f"{(df['gender'].str.lower() == 'female').sum():,}")

    st.subheader("Gender Distribution")
    st.bar_chart(df["gender"].value_counts())

    st.header("Model 1: Rigid Monitoring")

    st.write("Employees are classified using a rigid time-based threshold.")

    st.dataframe(
        df[["participant", "gender", "total_time", "linear_verdict"]].head(20),
        use_container_width=True
    )

    st.header("Model 2: Improved Monitoring")

    st.write("The improved model considers typing efficiency and task completion.")

    st.dataframe(
        df[[
            "participant",
            "gender",
            "active_typing_efficiency",
            "solved_verdict"
        ]].head(20),
        use_container_width=True
    )

    st.header("Gender-Wise Comparison")

    comparison = df.groupby("gender")[
        ["linear_verdict", "solved_verdict"]
    ].mean() * 100

    comparison.columns = [
        "Rigid Monitoring (%)",
        "Improved Monitoring (%)"
    ]

    st.dataframe(comparison, use_container_width=True)

    st.bar_chart(comparison)

    st.header("Model 2 Evaluation Metrics")

    predictions = df["solved_verdict"]

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", f"{accuracy_score(y2, predictions):.2%}")
    m2.metric("Precision", f"{precision_score(y2, predictions, zero_division=0):.2%}")
    m3.metric("Recall", f"{recall_score(y2, predictions, zero_division=0):.2%}")
    m4.metric("F1 Score", f"{f1_score(y2, predictions, zero_division=0):.2%}")

    st.warning(
        "These are training-set metrics from a simulated experiment, "
        "not independent test results or proof of real-world bias elimination."
    )

    st.download_button(
        "Download Analysis Results",
        df.to_csv(index=False),
        "gender_bias_analysis.csv",
        "text/csv"
    )

else:
    st.warning("Please upload both CSV files to view the dashboard.")
