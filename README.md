# 📊 InsightIQ

**AI-Powered Business Intelligence — Predictions & Document Insights in One Platform**

InsightIQ is a full-stack AI platform that combines classic machine learning predictions with a Retrieval-Augmented Generation (RAG) chatbot, letting any business turn raw data and documents into instant, natural-language insights.

🔗 **Live Demo:** [Add your Streamlit Cloud link here]

---

## ✨ Features

- **🔮 Predictive Analytics** — Upload any CSV (employee data, sales, customer records), pick a target column, and get an instantly trained model with accuracy metrics and a ranked breakdown of the factors that matter most.
- **💬 Document Q&A Chatbot** — Upload company PDFs (policies, reports, manuals) and ask questions in plain English or Roman Urdu. Answers are generated strictly from your documents using semantic search, not guessed.
- **🔁 Multi-Provider AI Fallback** — Automatically switches between Google Gemini and Groq if one provider is rate-limited or unavailable, ensuring the assistant never goes down.
- **🕓 Persistent History** — Every question and answer is logged to a local SQLite database for later review.
- **🎨 Professional Dashboard** — Clean, card-based interface designed for non-technical business users.

---

## 🧠 How It Works

InsightIQ combines two distinct AI approaches:

| Module | Handles | Technique |
|---|---|---|
| Predictive Analytics | Structured data (CSV) | Random Forest classification (scikit-learn) |
| Document Q&A | Unstructured data (PDF) | RAG — embeddings + semantic search + LLM generation |

Both modules train and index data **on the fly** from whatever the user uploads — there is no hardcoded dataset or document baked into the app.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend/App:** Streamlit
- **Machine Learning:** scikit-learn, pandas, NumPy
- **LLM APIs:** Google Gemini, Groq (multi-provider fallback)
- **RAG:** Sentence-Transformers (`all-MiniLM-L6-v2`) embeddings, cosine similarity search
- **Document Parsing:** pdfplumber
- **Database:** SQLite
- **Deployment:** Streamlit Cloud

---

## 📂 Project Structure

```
insightiq-ai/
├── app.py            # Main Streamlit app — UI, navigation, styling
├── ml_utils.py        # Dynamic model training on uploaded CSVs
├── rag_utils.py        # PDF parsing, chunking, embeddings, AI fallback logic
├── db_utils.py          # SQLite history logging
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/abrarhassanai573/insightiq-ai.git
cd insightiq-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API keys
Create a `.streamlit/secrets.toml` file:
```toml
GEMINI_API_KEY = "your_gemini_api_key"
GROQ_API_KEY = "your_groq_api_key"
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## ☁️ Deployment

This app is deployed on **Streamlit Community Cloud**. To deploy your own copy:

1. Fork/upload this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set `app.py` as the entry point.
4. Add `GEMINI_API_KEY` and `GROQ_API_KEY` under **App Settings → Secrets**.

---

## 👤 Author

**Abrar Hassan**
AI/ML & Generative AI Developer | BS Information Technology Student

- LinkedIn: [linkedin.com/in/abrar-hassan-7bba443bb](https://linkedin.com/in/abrar-hassan-7bba443bb)
- GitHub: [github.com/abrarhassanai573](https://github.com/abrarhassanai573)
- Email: abrarhassanai573@gmail.com

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
