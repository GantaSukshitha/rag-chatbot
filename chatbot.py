import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

DB_PATH = "vectorstore"



@st.cache_resource
def load_db():
    index = faiss.read_index(f"{DB_PATH}/index.faiss")
    chunks = json.load(open(f"{DB_PATH}/chunks.json"))
    return index, chunks


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_docs(query, index, chunks, embedder, k=3):
    q_vec = embedder.encode([query])
    dist, ids = index.search(np.array(q_vec), k)
    return [chunks[i] for i in ids[0]]


def generate_answer(context, query):
    llm = OllamaLLM(model="tinyllama")  # SMALL + LOW RAM
    prompt = (
    "Answer using ONLY the provided context below. "
    "Be concise and to the point. "
    "If the answer is not in the context, say 'Not found in PDF.' "
    "Do NOT define terms unless asked.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {query}\n\n"
    "Answer in 2–4 sentences:"
)

    return llm.invoke(prompt)


def main():
    st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📚")
    st.title("📚 RAG PDF Chatbot (Local + Offline)")

    index, chunks = load_db()
    embedder = load_embedder()

    query = st.text_input("Ask a question from your PDF:")

    if query:
        with st.spinner("Retrieving relevant chunks..."):
            docs = retrieve_docs(query, index, chunks, embedder)

        with st.spinner("Generating answer using local model..."):
            answer = generate_answer("\n\n".join(docs), query)

        st.write("### 🧾 Answer")
        st.write(answer)

        st.write("### 📚 Context Used")
        for d in docs:
            st.info(d[:300] + "...")


if __name__ == "__main__":
    main()
