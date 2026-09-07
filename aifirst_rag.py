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
import re
import hashlib

# --- PAGE CONFIG ---
st.set_page_config(page_title="AIFirst RAG Assistant", page_icon="📚", layout="wide")
st.title("🔍 AIFirst RAG Assistant")
st.markdown("Upload PDFs, DOCX, Excel, TXT, and more — then ask natural language questions.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("🔐 Configuration")

openai_key_input = st.sidebar.text_input("OpenAI API Key", type="password")
qdrant_api_input = st.sidebar.text_input("Qdrant API Key", type="password")
qdrant_url_input = st.sidebar.text_input(
    "Qdrant URL (e.g., https://yourhost.cloud:6333)",
    value="https://6a7820c2-43e6-45f7-bd2e-6e1f73bc6906.eu-central-1-0.aws.cloud.qdrant.io:6333"
)

# Cost control: let the user pick the completion model
model_choice = st.sidebar.selectbox(
    "Completion model",
    options=["gpt-4o-mini", "gpt-4o"],
    index=0,
    help="gpt-4o-mini is far cheaper and is plenty accurate for matching a grant to a fixed target list. "
         "Switch to gpt-4o only if you see it missing nuance."
)

# Session state defaults
for key in ["openai_valid", "qdrant_valid", "qdrant_client", "openai_client", "indexed_file_hash"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.sidebar.button("🔄 Connect & Validate"):
    try:
        openai_client = OpenAI(api_key=openai_key_input)
        openai_client.models.list()
        st.session_state["openai_valid"] = True
        st.session_state["openai_client"] = openai_client
    except Exception as e:
        st.session_state["openai_valid"] = False
        st.sidebar.error(f"❌ OpenAI key error: {e}")

    try:
        qdrant_client = QdrantClient(url=qdrant_url_input, api_key=qdrant_api_input)
        qdrant_client.get_collections()
        st.session_state["qdrant_valid"] = True
        st.session_state["qdrant_client"] = qdrant_client
    except Exception as e:
        st.session_state["qdrant_valid"] = False
        st.sidebar.error(f"❌ Qdrant error: {e}")

if st.session_state["openai_valid"] is True:
    st.sidebar.success("✅ OpenAI API key is valid!")
elif st.session_state["openai_valid"] is False:
    st.sidebar.error("❌ Invalid OpenAI API key")

if st.session_state["qdrant_valid"] is True:
    st.sidebar.success("✅ Qdrant connected!")
elif st.session_state["qdrant_valid"] is False:
    st.sidebar.error("❌ Qdrant not connected")

if not st.session_state.get("openai_valid"):
    st.warning("Please validate your OpenAI API key to continue.")
    st.stop()

if not st.session_state.get("qdrant_valid"):
    st.warning("Please validate your Qdrant credentials to continue.")
    st.stop()

client = st.session_state["openai_client"]
qdrant = st.session_state["qdrant_client"]
COLLECTION_NAME = "rag_demo"

if not any(c.name == COLLECTION_NAME for c in qdrant.get_collections().collections):
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# --- EMBEDDING MODEL (cached across reruns, not re-loaded every time) ---
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = get_embedder()


# --- STRUCTURED CHUNKER ---
# Parses the PBSAP-style layout into records that keep theme + target number +
# title as explicit metadata, instead of relying on where a line break happens
# to fall. This avoids a THEME header ending up glued onto the wrong target's
# text, which is what was happening with the old regex-split approach.
TARGET_HEADER = re.compile(r"^\s*(\d{1,2})\s*[—\-–]\s*(.+)$")
THEME_HEADER = re.compile(r"^\s*THEME\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)


def structured_chunk(text, min_chars=30):
    lines = [l.strip() for l in text.split("\n")]
    records = []
    current_theme = None
    current_target_num = None
    current_title = None
    current_target_theme = None  # theme pinned at the moment THIS target started
    current_body = []

    def flush():
        if current_target_num is not None:
            body = " ".join(current_body).strip()
            body = re.sub(r"\s+", " ", body)
            if len(body) > min_chars or current_title:
                records.append({
                    "theme": current_target_theme or "",
                    "target_num": current_target_num,
                    "title": current_title or "",
                    "text": f"Theme: {current_target_theme}\nTarget {current_target_num} — {current_title}\n{body}".strip()
                })

    for line in lines:
        if not line:
            continue
        theme_match = THEME_HEADER.match(line)
        if theme_match:
            # A new THEME header always describes the block of targets that
            # FOLLOWS it, not the one just finished. Flush the target that
            # was in progress (it keeps the theme it started under) before
            # switching current_theme, otherwise the last target of a
            # section gets mislabeled with the next section's theme.
            flush()
            current_target_num = None
            current_theme = f"THEME {theme_match.group(1)}: {theme_match.group(2)}"
            continue
        target_match = TARGET_HEADER.match(line)
        if target_match:
            flush()
            current_target_num = target_match.group(1)
            current_title = target_match.group(2).strip()
            current_target_theme = current_theme
            current_body = []
            continue
        current_body.append(line)
    flush()
    return records


def sliding_window_chunk(text, max_chars=800, overlap_chars=100, min_chars=30):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks = []
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = current[-overlap_chars:] + " " + line
        else:
            current = (current + " " + line).strip()
    if current.strip():
        chunks.append(current.strip())
    return [{"theme": "", "target_num": "", "title": "", "text": c}
            for c in chunks if len(c) > min_chars]


def chunk_text(text):
    records = structured_chunk(text)
    if len(records) >= 3:
        return records
    # Fallback for documents that don't follow the THEME / N — TITLE layout
    return sliding_window_chunk(text)


# --- FILE EXTRACTOR ---
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


# --- UPLOAD + EMBED ---
uploaded_file = st.file_uploader(
    "📄 Upload a document (PDF, DOCX, Excel, CSV, TXT, HTML)",
    type=["txt", "pdf", "docx", "xlsx", "xls", "csv", "html"]
)

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Cost control: only re-embed if this is a genuinely new/changed file.
    # Without this check, every Streamlit rerun (e.g. typing in the question
    # box) would re-read, re-embed, and re-upsert the same file again.
    if st.session_state.get("indexed_file_hash") == file_hash:
        st.info(f"✅ '{uploaded_file.name}' is already indexed — skipping re-embedding.")
    else:
        qdrant.delete_collection(COLLECTION_NAME)
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        file_type = uploaded_file.name.split(".")[-1].lower()
        import io
        text = extract_text_from_file(io.BytesIO(file_bytes), file_type)

        if not text:
            st.warning("⚠️ No extractable text found.")
        else:
            records = chunk_text(text)
            with st.spinner("🔎 Embedding and storing text chunks..."):
                texts = [r["text"] for r in records]
                vectors = embedder.encode(texts).tolist()
                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec,
                        payload=r
                    )
                    for vec, r in zip(vectors, records)
                ]
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            st.session_state["indexed_file_hash"] = file_hash
            st.success(f"✅ {len(records)} chunks embedded into Qdrant!")

# --- QUERY ---
st.header("🧠 Ask a Question")
user_query = st.text_input("Enter your question:")

if st.button("Get RAG Answer", disabled=not user_query.strip()):
    try:
        # Corpus is a fixed, short list of PBSAP targets (a couple dozen
        # short records) — retrieving everything is intentional here, not a
        # bug: cosine similarity on a long grant description reliably drops
        # the correct target when scored against short target text. This
        # does NOT scale to large/multi-document corpora; if you outgrow a
        # single short reference list, switch back to top-k similarity search.
        all_points, _ = qdrant.scroll(collection_name=COLLECTION_NAME, limit=1000)
        retrieved = [p.payload for p in all_points]
        context = "\n\n".join(r.get("text", "") for r in retrieved)

        st.subheader("📚 Retrieved Context")
        # st.text (not st.write/markdown) so things like "$200 BILLION" in
        # Target 19 render as plain text instead of being parsed as LaTeX.
        st.text(context)

        prompt = (
            f"You are matching a grant to the single best-fitting PBSAP target below.\n\n"
            f"PBSAP TARGETS:\n{context}\n\n"
            f"TASK:\n{user_query}\n\n"
            f"DISAMBIGUATION RULES — apply before choosing:\n"
            f"- Target 2 (Restore 30% of degraded ecosystems) applies ONLY if the grant explicitly "
            f"describes rehabilitating land/water that is degraded, denuded, damaged, or deforested. "
            f"Mentions of 'forest,' 'biodiversity,' 'trees,' or 'conservation' alone are NOT enough to "
            f"trigger Target 2 — those are generic terms that appear across many targets.\n"
            f"- If the grant protects or manages an EXISTING intact forest/site (patrolling, monitoring, "
            f"management plans, governance), prefer Target 1 (planning/management) or Target 3 "
            f"(effective conservation of already-designated areas) over Target 2.\n"
            f"- If the grant's core activity is a sustainable livelihood, enterprise, or production model "
            f"built around forestry/agriculture/fisheries/aquaculture (e.g. agroforestry, forest-based "
            f"products, sustainable harvesting as an alternative livelihood), prefer Target 10 over Target 2, "
            f"even if it also includes some tree-planting or nursery components as a supporting activity.\n"
            f"- Only pick Target 2 when restoration/rehabilitation of degraded land IS the grant's stated "
            f"primary objective, not a minor or implied side-activity.\n\n"
            f"Internally compare the top 2-3 candidate targets against the grant's actual "
            f"activities (not just keyword overlap), applying the disambiguation rules above. Then commit "
            f"to the single best match. Do NOT print your comparison or reasoning. Output ONLY the final "
            f"answer, in exactly the format requested in the TASK above — nothing before it, nothing after it."
        )
        with st.spinner("🤖 Thinking..."):
            response = client.chat.completions.create(
                model=model_choice,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            answer = response.choices[0].message.content.strip()

        st.subheader("💬 RAG Answer")
        st.text(answer)

    except Exception as e:
        st.error(f"❌ Error during RAG answering: {e}")
