import streamlit as st
import pandas as pd

from common import apply_styling, render_header
from ml_utils import train_model

st.set_page_config(page_title="Predictions | InsightIQ", page_icon="🔮", layout="wide")
apply_styling()
render_header(
    "🔮 Predictive Analytics",
    "Upload any business CSV and let AI find patterns, predict outcomes, "
    "and rank the factors that matter most."
)

uploaded_csv = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.dataframe(df.head(), use_container_width=True)

    target_col = st.selectbox(
        "Select the column you want to predict (e.g. Attrition, Churn)",
        df.columns
    )

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model, accuracy, importances, feature_cols = train_model(df, target_col)

        col1, col2, col3 = st.columns(3)
        col1.metric("Model Accuracy", f"{accuracy:.2%}")
        col2.metric("Total Records", len(df))
        col3.metric("Features Used", len(feature_cols))

        st.subheader("Top 10 Factors")
        st.bar_chart(importances.head(10))
else:
    st.info("👆 Upload a CSV to get started.")
