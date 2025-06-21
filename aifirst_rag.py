import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import pandas as pd
import docx
import fitz  # PyMuPDF for PDFs
from bs4 import BeautifulSoup
import uuid

# --- PAGE SETUP ---
st.set_page_config(page_title="AIFirst RAG Assistant", page_icon="📚", layout="wide")
st.title("🔍 AIFirst RAG Assistant (Multiformat + Qdrant)")
st.markdown("Upload PDFs, DOCX, Excel, TXT, and more — then ask natural language questions.")

# --- SIDEBAR ---
st.sidebar.title("🔐 Configuration")

# OpenAI
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()

client = OpenAI(api_key=openai_key)

# Qdrant
qdrant_api = st.sidebar.text_input("Qdrant API Key", type="password")
qdrant_url = st.sidebar.text_input("Qdrant URL (e.g., https://yourhost.cloud)", value="http://localhost:6333")

if not qdrant_url:
    st.warning("Please enter your Qdrant host URL.")
    st.stop()

try:
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api if qdrant_api else None)
except Exception as e:
    st.error(f"Qdrant connection failed: {e}")
    st.stop()

COLLECTION_NAME = "rag_demo"

# --- COLLECTION SETUP ---
try:
    collections = qdrant.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
except Exception as e:
    st.error(f"Failed to check/create Qdrant collection: {e}")
    st.stop()

# --- EMBEDDING MODEL ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- FILE TEXT EXTRACTOR ---
def extract_text_from_file(uploaded_file, file_type):
    try:
        if file_type == "txt":
            return uploaded_file.read().decode("utf-8")

        elif file_type == "pdf":
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                return "\n\n".join([page.get_text() for page in doc])

        elif file_type == "docx":
            doc = docx.Document(uploaded_file)
            return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
            return df.to_string(index=False)

        elif file_type == "csv":
            df = pd.read_csv(uploaded_file)
            return df.to_string(index=False)

        elif file_type == "html":
            soup = BeautifulSoup(uploaded_file.read(), "html.parser")
            return soup.get_text()

        else:
            return ""
    except Exception as e:
        st.error(f"Failed to process file: {e}")
        return ""

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📄 Upload a document (PDF, Word, Excel, CSV, TXT, HTML)", type=["txt", "pdf", "docx", "xlsx", "xls", "csv", "html"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    extracted_text = extract_text_from_file(uploaded_file, file_type)
    if not extracted_text:
        st.warning("No readable text was extracted.")
    else:
        chunks = [chunk.strip() for chunk in extracted_text.split("\n\n") if len(chunk.strip()) > 30]

        with st.spinner("🔎 Embedding and storing chunks..."):
            vectors = embedder.encode(chunks).tolist()
            points = [
                PointStruct(id=str(uuid.uuid4()), vector=vec, payload={"text": chunk})
                for vec, chunk in zip(vectors, chunks)
            ]
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        st.success(f"✅ {len(chunks)} chunks embedded and stored.")

# --- QUERY SECTION ---
st.header("🧠 Ask a Question")
user_query = st.text_input("Type your question here:")

if st.button("Get RAG Answer", disabled=not user_query.strip()):
    try:
        # Step 1: Embed the query
        query_vec = embedder.encode([user_query])[0]

        # Step 2: Search Qdrant
        results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=query_vec, limit=5)
        retrieved_chunks = [hit.payload['text'] for hit in results]

        context = "\n\n".join(retrieved_chunks)

        st.subheader("📚 Retrieved Context")
        st.write(context)

        # Step 3: Generate Answer with OpenAI
        prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {user_query}"

        with st.spinner("💡 Generating answer..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer = response.choices[0].message.content.strip()

        st.subheader("🤖 RAG Answer")
        st.write(answer)

    except Exception as e:
        st.error(f"❌ Error during RAG: {e}")