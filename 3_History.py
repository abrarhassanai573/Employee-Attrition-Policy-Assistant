import streamlit as st
import pandas as pd

from common import apply_styling, render_header
from db_utils import init_db, fetch_history

st.set_page_config(page_title="History | InsightIQ", page_icon="🕓", layout="wide")
apply_styling()
render_header("🕓 Chat History", "Every question and answer asked through InsightIQ, logged automatically.")

init_db()
history = fetch_history()

if history:
    hist_df = pd.DataFrame(history, columns=["Time", "Question", "Answer", "Provider"])
    st.dataframe(hist_df, use_container_width=True)
else:
    st.info("No history yet — ask something on the Ask AI page first.")
