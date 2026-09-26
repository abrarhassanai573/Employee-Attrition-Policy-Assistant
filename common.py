import streamlit as st
from google import genai
from groq import Groq
from sentence_transformers import SentenceTransformer


def apply_styling():
    """Applies the shared InsightIQ dashboard theme to any page."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        div[data-testid="stMetricLabel"] { color: #6B7280; font-weight: 500; }
        div[data-testid="stMetricValue"] { color: #111827; font-weight: 700; }

        h1 { color: #111827; font-weight: 700; letter-spacing: -0.5px; }
        h2, h3 { color: #1F2937; font-weight: 600; }

        .stButton>button {
            background-color: #4F46E5;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.55em 1.6em;
            font-weight: 500;
            transition: background-color 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #4338CA;
            color: white;
        }

        section[data-testid="stSidebar"] {
            background-color: #111827;
        }
        section[data-testid="stSidebar"] * { color: #F9FAFB !important; }

        .feature-card {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 22px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            height: 100%;
        }
        .feature-card h4 { margin-top: 0; color: #4F46E5; }
        .feature-card p { color: #4B5563; font-size: 0.92em; line-height: 1.5; }

        div[data-testid="stAlert"] { border-radius: 10px; }
        </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str):
    """Renders the consistent branded page header."""
    st.markdown(f"""
        <div style="padding: 4px 0 18px 0;">
            <h1 style="margin-bottom:0;">{title}</h1>
            <p style="color:#6B7280; font-size:1.05em; margin-top:4px;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_clients():
    """Initializes Gemini and Groq clients once per session (used for fallback)."""
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return gemini_client, groq_client


@st.cache_resource
def get_embed_model():
    """Loads the sentence embedding model once per session."""
    return SentenceTransformer('all-MiniLM-L6-v2')
