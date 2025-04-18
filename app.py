import streamlit as st
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import faiss
import numpy as np
import os
import textwrap
import tempfile

@st.cache_resource
def load_model():
    return SentenceTransformer('BAAI/bge-large-en-v1.5')

model = load_model()

st.title("📄 Document Q&A with BGE-large + FAISS (No OpenAI 🔓)")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
question = st.text_input("Ask a question about the document:")
submit = st.button("Get Answer")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.index = None

def load_and_chunk(file) -> list:
    text = ""
    temp_path = tempfile.mkstemp()[1]
    with open(temp_path, "wb") as f:
        f.write(file.read())

    if file.name.endswith(".pdf"):
        doc = fitz.open(temp_path)
        for page in doc:
            text += page.get_text()
    elif file.name.endswith(".txt"):
        with open(temp_path, "r", encoding="utf-8") as f:
            text += f.read()
    os.remove(temp_path)
    chunks = textwrap.wrap(text, width=500, break_long_words=False)
    return chunks

def embed_and_index(chunks: list):
    vectors = model.encode(chunks, convert_to_numpy=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

if uploaded_file:
    st.session_state.chunks = load_and_chunk(uploaded_file)
    st.session_state.index = embed_and_index(st.session_state.chunks)
    st.success(f"✅ Processed and indexed {len(st.session_state.chunks)} chunks.")

if submit and question:
    if st.session_state.index is None:
        st.warning("Please upload and process a document first.")
    else:
        q_vector = model.encode([question], convert_to_numpy=True)
        D, I = st.session_state.index.search(q_vector, k=3)
        st.subheader("Top Matching Answers:")
        for i in I[0]:
            st.markdown(f"🟢 **Match:**\n```\n{st.session_state.chunks[i]}\n```")
