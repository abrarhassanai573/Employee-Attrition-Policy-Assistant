import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_from_pdf(file):
    """Extracts all text from an uploaded PDF file."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    """Splits long text into overlapping word chunks for embedding."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def embed_texts(embed_model, texts):
    """Converts a list of text chunks into embeddings."""
    return embed_model.encode(texts)


def search_chunks(query, chunks, embeddings, embed_model, top_k=3):
    """Finds the most relevant chunk(s) for a given query using cosine similarity."""
    query_embedding = embed_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [(chunks[i], similarities[i]) for i in top_indices]


def get_ai_response(prompt, gemini_client, groq_client):
    """
    Tries Gemini first. If it fails (rate limit, downtime, etc.),
    automatically falls back to Groq. This is the multi-provider
    fallback pattern.
    """
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )
        return response.text, "Gemini"
    except Exception:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content, "Groq"
