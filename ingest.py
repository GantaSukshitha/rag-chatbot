import os
import json
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_PATH = "data"
DB_PATH = "vectorstore"

def load_pdfs():
    texts = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(DATA_PATH, file))
            for page in reader.pages:
                texts.append(page.extract_text())
    return texts

def chunk(text, size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest():
    raw = load_pdfs()
    print(f"Loaded {len(raw)} pages")

    all_chunks = []
    for page in raw:
        if page:
            all_chunks.extend(chunk(page))

    print(f"Created {len(all_chunks)} chunks")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(all_chunks)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))

    os.makedirs(DB_PATH, exist_ok=True)
    faiss.write_index(index, f"{DB_PATH}/index.faiss")
    json.dump(all_chunks, open(f"{DB_PATH}/chunks.json", "w"))

    print("Vectorstore built successfully!")

if __name__ == "__main__":
    ingest()
