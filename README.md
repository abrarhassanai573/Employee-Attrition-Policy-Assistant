# HR Insight App

A Streamlit web application that combines a machine learning model for **employee attrition prediction** with a **Retrieval-Augmented Generation (RAG) chatbot** for answering HR policy questions.

**Live demo:** _add your Streamlit Cloud URL here once deployed_

---

## Features

### 1. Employee Attrition Prediction
- Enter an employee's profile (age, income, job role, satisfaction scores, overtime status, etc.)
- A trained **Random Forest Classifier** predicts whether the employee is likely to leave the company, along with a probability score
- Every prediction is logged to a local history table

### 2. HR Policy Chatbot (RAG)
- Ask natural-language questions about company policies (Leave, Remote Work, Overtime)
- The app performs **semantic search** over the policy documents using sentence embeddings, rather than simple keyword matching
- The most relevant policy is passed as context to a large language model, which generates a grounded answer
- **Dual-provider fallback:** the app first tries **Google Gemini**; if that request fails or is rate-limited, it automatically retries with **Groq**, so the chatbot stays available even during provider outages
- Every question and answer is logged to a local history table

### 3. History Dashboard
- View past chatbot conversations and past predictions in a simple tabbed interface

---

## How It Works

| Step | Component | Description |
|---|---|---|
| 1 | **Embedding model** | `sentence-transformers` (`all-MiniLM-L6-v2`) converts each policy document and each user query into a 384-dimensional vector |
| 2 | **Semantic search** | Cosine similarity ranks the policy documents against the query embedding to find the most relevant one |
| 3 | **LLM generation** | The matched policy text is injected into a prompt sent to Gemini (primary) or Groq (fallback), which answers strictly from that context |
| 4 | **ML prediction** | Employee inputs are one-hot encoded to match the model's expected feature set, then passed to the pre-trained Random Forest model |
| 5 | **Persistence** | SQLite stores prediction and chat history locally within the deployed app |

---

## Tech Stack

- **Frontend / App framework:** Streamlit
- **ML model:** scikit-learn (Random Forest Classifier), trained on the IBM HR Analytics Attrition dataset
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **LLM providers:** Google Gemini (`gemini-2.0-flash`), Groq (`openai/gpt-oss-20b`) as fallback
- **Storage:** SQLite (local, ephemeral on Streamlit Cloud)

---

## Project Structure

```
hr-app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Pinned Python dependencies
├── attrition_model.pkl             # Trained Random Forest model
├── model_columns.pkl               # Feature column order expected by the model
├── .streamlit/
│   └── secrets.toml.example        # Template for required API keys
├── .gitignore
└── README.md
```

---

## Setup & Deployment

### 1. Clone / upload this repository to GitHub
Push all files above to a new GitHub repository, **excluding** any real `secrets.toml` file.

### 2. Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **"Create app"**, select this repository, and set the main file to `app.py`
3. Under **Advanced settings → Python version**, select **3.11** (for compatibility with pinned dependencies)
4. Under **Advanced settings → Secrets**, add:

   ```toml
   GEMINI_API_KEY = "your_actual_gemini_key"
   GROQ_API_KEY = "your_actual_groq_key"
   ```

5. Click **Deploy**

### 3. Run locally (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Create a `.streamlit/secrets.toml` file locally (copy the format from `secrets.toml.example`) with your own API keys before running.

---

## Notes & Limitations

- Semantic search is performed over a small set of three policy documents; retrieval quality depends on how closely the query's wording matches the underlying policy content.
- SQLite history is stored on the app's local filesystem, which is ephemeral on Streamlit Cloud — history resets on app restart or redeploy. For persistent storage, replace it with an external database.
- The Random Forest model was trained with `scikit-learn==1.6.1`; this version is pinned in `requirements.txt` to ensure prediction consistency.
- API keys must be added via Streamlit's Secrets manager — never commit real keys to the repository.

---

## License

For personal/educational use. Update this section if you intend to open-source or distribute the project.
