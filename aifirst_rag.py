import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import pandas as pd
import docx
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import uuid

# --- PAGE SETUP ---
st.set_page_config(page_title="AIFirst RAG Assistant", page_icon="📚", layout="wide")
st.title("🔍 AIFirst RAG Assistant")
st.markdown("Upload PDFs, DOCX, Excel, TXT, and more — then ask natural language questions.")

# --- SIDEBAR ---
st.sidebar.title("🔐 Configuration")

# OpenAI API Key
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_key:
    st.sidebar.warning("Please enter your OpenAI API key.")

# Qdrant Config
qdrant_api = st.sidebar.text_input("Qdrant API Key", type="password")
qdrant_url = st.sidebar.text_input(
    "Qdrant URL (e.g., https://yourhost.cloud:6333)",
    value="https://6a7820c2-43e6-45f7-bd2e-6e1f73bc6906.eu-central-1-0.aws.cloud.qdrant.io:6333"
)

# Try to connect to Qdrant only if both URL and API key are present
qdrant_connected = False
qdrant = None
COLLECTION_NAME = "rag_demo"

if qdrant_url and qdrant_api:
    try:
        qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api)
        # Check or create collection
        existing = qdrant.get_collections().collections
        if not any(c.name == COLLECTION_NAME for c in existing):
            qdrant.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        st.sidebar.success("✅ Connected to Qdrant")
        qdrant_connected = True
    except Exception as e:
        st.sidebar.error(f"❌ Qdrant connection failed:\n{e}")
else:
    st.sidebar.info("Please input Qdrant URL and API Key.")

# --- STOP IF KEYS MISSING ---
if not openai_key:
    st.warning("🔑 OpenAI API Key is required.")
    st.stop()

if not qdrant_connected:
    st.warning("🧱 Qdrant must be connected before using the app.")
    st.stop()

client = OpenAI(api_key=openai_key)
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
        st.error(f"❌ Failed to extract text: {e}")
        return ""

# --- FILE UPLOAD + EMBEDDING ---
uploaded_file = st.file_uploader("📄 Upload a document (PDF, DOCX, Excel, CSV, TXT, HTML)", 
                                  type=["txt", "pdf", "docx", "xlsx", "xls", "csv", "html"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    text = extract_text_from_file(uploaded_file, file_type)

    if not text:
        st.warning("⚠️ No extractable text found.")
    else:
        chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 30]
        with st.spinner("🔎 Embedding and storing text chunks..."):
            vectors = embedder.encode(chunks).tolist()
            points = [
                PointStruct(id=str(uuid.uuid4()), vector=vec, payload={"text": chunk})
                for vec, chunk in zip(vectors, chunks)
            ]
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        st.success(f"✅ {len(chunks)} chunks embedded into Qdrant!")

# --- QUERY ---
st.header("🧠 Ask a Question")
user_query = st.text_input("Enter your question:")

if st.button("Get RAG Answer", disabled=not user_query.strip()):
    try:
        query_vec = embedder.encode([user_query])[0]
        results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=query_vec, limit=5)
        retrieved_chunks = [hit.payload['text'] for hit in results]

        context = "\n\n".join(retrieved_chunks)

        st.subheader("📚 Retrieved Context")
        st.write(context)

        prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {user_query}"
        with st.spinner("🤖 Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer = response.choices[0].message.content.strip()

        st.subheader("💬 RAG Answer")
        st.write(answer)

    except Exception as e:
        st.error(f"❌ Error during RAG answering: {e}")
