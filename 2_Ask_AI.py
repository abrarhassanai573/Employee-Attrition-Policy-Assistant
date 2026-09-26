import streamlit as st

from common import apply_styling, render_header, get_clients, get_embed_model
from rag_utils import extract_text_from_pdf, chunk_text, embed_texts, search_chunks, get_ai_response
from db_utils import init_db, log_chat

st.set_page_config(page_title="Ask AI | InsightIQ", page_icon="💬", layout="wide")
apply_styling()
render_header(
    "💬 Document Q&A Assistant",
    "Upload company documents (PDF) and ask questions in plain English or Roman Urdu — "
    "answers are sourced directly from your documents."
)

init_db()
gemini_client, groq_client = get_clients()
embed_model = get_embed_model()

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

uploaded_pdfs = st.file_uploader(
    "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
)

if uploaded_pdfs and st.button("Process Documents"):
    with st.spinner("Processing documents..."):
        all_chunks = []
        for pdf_file in uploaded_pdfs:
            text = extract_text_from_pdf(pdf_file)
            all_chunks.extend(chunk_text(text))

        st.session_state.chunks = all_chunks
        st.session_state.embeddings = embed_texts(embed_model, all_chunks)
    st.success(f"{len(all_chunks)} chunks indexed successfully!")

if st.session_state.chunks:
    query = st.text_input("Ask a question:")
    if query and st.button("Ask"):
        with st.spinner("Thinking..."):
            results = search_chunks(
                query, st.session_state.chunks, st.session_state.embeddings,
                embed_model, top_k=2
            )
            context = "\n\n".join([r[0] for r in results])

            prompt = f"""You are a helpful assistant. Answer using ONLY the context below.
If the answer isn't in the context, say you don't have that information.

Context:
{context}

Question: {query}

Answer in the same language as the question (English or Roman Urdu)."""

            answer, provider = get_ai_response(prompt, gemini_client, groq_client)
            log_chat(query, answer, provider)

        st.write(f"**Answer** _(via {provider})_:")
        st.info(answer)
else:
    st.info("👆 Upload and process documents first.")
