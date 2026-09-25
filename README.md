# HR Insight App

This app does two things:
1. **Attrition Prediction** — enter employee details and predict whether they're likely to leave the company
2. **HR Policy Chatbot** — ask questions about Leave, Remote Work, and Overtime policies (RAG + Gemini/Groq)

Everything can be done from the browser — no need to install Python or VS Code.

---

## Step 1: Create a Repository on GitHub

1. Log in at [github.com](https://github.com)
2. Click the "+" icon (top-right) → **New repository**
3. Give it a name (e.g. `hr-insight-app`) → choose **Public** or **Private** → **Create repository**
4. On the new repo page, click **"uploading an existing file"**
5. Drag and drop **all the files** from this folder:
   - `app.py`
   - `requirements.txt`
   - `.gitignore`
   - `attrition_model.pkl`
   - `model_columns.pkl`
   - `.streamlit/secrets.toml.example`
   - `README.md`

   **Important:** Upload `secrets.toml.example`, but **never** upload a real `secrets.toml` file containing actual API keys.
6. Scroll down and click "Commit changes"

---

## Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Log in / sign up with your GitHub account
3. Click **"Create app"** → select your repository → set the main file to `app.py`
4. Under **"Advanced settings" → "Secrets"**, paste this (with your real keys):

   ```toml
   GEMINI_API_KEY = "your_actual_gemini_key"
   GROQ_API_KEY = "your_actual_groq_key"
   ```

5. Click **Deploy** — within 2-3 minutes the app will go live with a public URL

---

## Notes

- `history.db` (chat and prediction history) is created automatically the first time the app runs — you don't need to upload it to GitHub (it's already excluded in `.gitignore`)
- If `sentence-transformers` takes a little time on first run, that's normal (it's downloading the embedding model)
- The model was trained with `scikit-learn==1.6.1` — this exact version is pinned in `requirements.txt` to keep predictions accurate
