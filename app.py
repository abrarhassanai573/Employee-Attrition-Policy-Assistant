import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="HR Insight App", page_icon="💼", layout="wide")

# =========================================================
# LOAD ML MODEL (cached so it only loads once)
# =========================================================
@st.cache_resource
def load_model():
    model = joblib.load("attrition_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()

# =========================================================
# RAG DOCUMENTS (HR Policies)
# =========================================================
policy_1 = """
Leave Policy
Employees are entitled to 20 paid leaves per year.
Sick leave requires a medical certificate if taken for more than 2 consecutive days.
Unused leaves can be carried forward up to 5 days into the next year.
"""

policy_2 = """
Remote Work Policy
Employees may work remotely up to 2 days per week with manager approval.
Remote work requests must be submitted at least 3 days in advance.
Employees must be available on company communication tools during work hours.
"""

policy_3 = """
Overtime Policy
Overtime is compensated at 1.5x the regular hourly rate.
Overtime must be pre-approved by the department head.
Maximum overtime per month is capped at 30 hours.
"""

documents = [policy_1, policy_2, policy_3]

# =========================================================
# EMBEDDING MODEL (cached so it only loads once)
# =========================================================
@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_doc_embeddings():
    embedder = load_embedder()
    return embedder.encode(documents)

# =========================================================
# SQLITE — chat + prediction history
# =========================================================
def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    query TEXT,
                    answer TEXT,
                    provider TEXT
                 )""")
    c.execute("""CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    inputs TEXT,
                    result TEXT,
                    probability REAL
                 )""")
    conn.commit()
    conn.close()

def save_chat(query, answer, provider):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (timestamp, query, answer, provider) VALUES (?, ?, ?, ?)",
              (datetime.now().isoformat(), query, answer, provider))
    conn.commit()
    conn.close()

def save_prediction(inputs_str, result, probability):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("INSERT INTO prediction_history (timestamp, inputs, result, probability) VALUES (?, ?, ?, ?)",
              (datetime.now().isoformat(), inputs_str, result, probability))
    conn.commit()
    conn.close()

init_db()

# =========================================================
# SEMANTIC SEARCH
# =========================================================
def search_documents(query, top_k=1):
    from sklearn.metrics.pairwise import cosine_similarity
    embedder = load_embedder()
    doc_embeddings = get_doc_embeddings()
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [{"document": documents[i], "score": similarities[i]} for i in top_indices]

# =========================================================
# LLM CALL (Gemini -> Groq fallback)
# =========================================================
def ask_llm(query, context):
    prompt = f"""You are an HR assistant. Answer the employee's question using ONLY the
context below. If the answer isn't in the context, say you don't have that information.

Context:
{context}

Question: {query}

Answer clearly and briefly."""

    gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    groq_key = st.secrets.get("GROQ_API_KEY", "")

    # Try Gemini first
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            response = gemini_model.generate_content(prompt)
            return response.text, "Gemini"
        except Exception as e:
            st.session_state["last_gemini_error"] = str(e)

    # Fallback to Groq
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content, "Groq"
        except Exception as e:
            return f"Both providers failed. Groq error: {e}", "None"

    return "No API keys configured. Please add GEMINI_API_KEY and/or GROQ_API_KEY in Streamlit secrets.", "None"

def rag_chatbot(query):
    results = search_documents(query, top_k=1)
    context = results[0]["document"]
    answer, provider = ask_llm(query, context)
    return answer, provider, context

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("💼 HR Insight App")
page = st.sidebar.radio("Navigate", ["Attrition Prediction", "HR Policy Chatbot", "History"])

# =========================================================
# PAGE 1: ATTRITION PREDICTION
# =========================================================
if page == "Attrition Prediction":
    st.title("Employee Attrition Prediction")
    st.caption("Fill in employee details to predict attrition risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 65, 30)
        daily_rate = st.number_input("Daily Rate", 100, 1500, 800)
        distance = st.number_input("Distance From Home", 0, 30, 5)
        education = st.slider("Education (1-5)", 1, 5, 3)
        env_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        hourly_rate = st.number_input("Hourly Rate", 30, 100, 60)
        job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
        job_level = st.slider("Job Level (1-5)", 1, 5, 2)
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
        monthly_rate = st.number_input("Monthly Rate", 2000, 27000, 14000)
        num_companies = st.number_input("Num Companies Worked", 0, 10, 2)

    with col2:
        salary_hike = st.slider("Percent Salary Hike", 10, 25, 15)
        performance = st.slider("Performance Rating (1-4)", 1, 4, 3)
        relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
        stock_option = st.slider("Stock Option Level (0-3)", 0, 3, 1)
        total_working_years = st.number_input("Total Working Years", 0, 40, 8)
        training_times = st.slider("Training Times Last Year", 0, 6, 2)
        work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
        years_at_company = st.number_input("Years At Company", 0, 40, 5)
        years_in_role = st.number_input("Years In Current Role", 0, 20, 3)
        years_since_promotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
        years_with_manager = st.number_input("Years With Current Manager", 0, 20, 3)

    with col3:
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Frequently", "Travel_Rarely"])
        department = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
        education_field = st.selectbox("Education Field", ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        job_role = st.selectbox("Job Role", ["Healthcare Representative", "Human Resources", "Laboratory Technician",
                                              "Manager", "Manufacturing Director", "Research Director",
                                              "Research Scientist", "Sales Executive", "Sales Representative"])
        marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
        overtime = st.selectbox("OverTime", ["No", "Yes"])

    if st.button("Predict Attrition", type="primary"):
        row = {col: 0 for col in model_columns}

        numeric_values = {
            "Age": age, "DailyRate": daily_rate, "DistanceFromHome": distance,
            "Education": education, "EnvironmentSatisfaction": env_satisfaction,
            "HourlyRate": hourly_rate, "JobInvolvement": job_involvement,
            "JobLevel": job_level, "JobSatisfaction": job_satisfaction,
            "MonthlyIncome": monthly_income, "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies, "PercentSalaryHike": salary_hike,
            "PerformanceRating": performance, "RelationshipSatisfaction": relationship_satisfaction,
            "StockOptionLevel": stock_option, "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times, "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company, "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promotion, "YearsWithCurrManager": years_with_manager,
        }
        for k, v in numeric_values.items():
            if k in row:
                row[k] = v

        one_hot_selections = {
            f"BusinessTravel_{business_travel}": 1,
            f"Department_{department}": 1,
            f"EducationField_{education_field}": 1,
            f"Gender_{gender}": 1,
            f"JobRole_{job_role}": 1,
            f"MaritalStatus_{marital_status}": 1,
            f"OverTime_{overtime}": 1,
        }
        for k, v in one_hot_selections.items():
            if k in row:
                row[k] = v

        input_df = pd.DataFrame([row])[model_columns]
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        result_label = "Likely to Leave" if prediction == 1 else "Likely to Stay"

        if prediction == 1:
            st.error(f"⚠️ {result_label} — Attrition probability: {probability:.1%}")
        else:
            st.success(f"✅ {result_label} — Attrition probability: {probability:.1%}")

        save_prediction(str(numeric_values | one_hot_selections), result_label, float(probability))

# =========================================================
# PAGE 2: RAG CHATBOT
# =========================================================
elif page == "HR Policy Chatbot":
    st.title("HR Policy Chatbot")
    st.caption("Ask a question about Leave, Remote Work, or Overtime policies.")

    query = st.text_input("Your question")

    if st.button("Ask", type="primary") and query:
        with st.spinner("Thinking..."):
            answer, provider, context = rag_chatbot(query)
        st.markdown(f"**Answer** _(via {provider})_:")
        st.write(answer)
        with st.expander("Source policy used"):
            st.text(context)
        save_chat(query, answer, provider)

# =========================================================
# PAGE 3: HISTORY
# =========================================================
elif page == "History":
    st.title("History")

    tab1, tab2 = st.tabs(["Chat History", "Prediction History"])

    conn = sqlite3.connect("history.db")

    with tab1:
        chat_df = pd.read_sql_query("SELECT timestamp, query, answer, provider FROM chat_history ORDER BY id DESC", conn)
        st.dataframe(chat_df, use_container_width=True)

    with tab2:
        pred_df = pd.read_sql_query("SELECT timestamp, result, probability FROM prediction_history ORDER BY id DESC", conn)
        st.dataframe(pred_df, use_container_width=True)

    conn.close()
