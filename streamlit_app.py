"""
Q&A Chatbot using RAG Architecture - Streamlit Web App
=======================================================
Upload documents (PDF/TXT) and ask questions.
The chatbot retrieves relevant context and generates grounded answers.

Run with: streamlit run streamlit_app.py
"""

import os
import re
import json
import time
import pickle
import tempfile
import ssl
from typing import List, Dict

# --- Fix SSL certificate issues (corporate proxy/firewall) ---
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
ssl._create_default_https_context = ssl._create_unverified_context

import httpx
_original_httpx_client_init = httpx.Client.__init__
def _patched_httpx_client_init(self, *args, **kwargs):
    kwargs['verify'] = False
    _original_httpx_client_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_httpx_client_init
# --- End SSL fix ---

import numpy as np
import streamlit as st
import nltk

# Download NLTK data safely (avoid file lock issues)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

from nltk.tokenize import sent_tokenize

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="RAG Q&A Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f7ff;
        border-left: 4px solid #1E88E5;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    .relevance-high { color: #2E7D32; font-weight: bold; }
    .relevance-medium { color: #F57F17; font-weight: bold; }
    .relevance-low { color: #C62828; font-weight: bold; }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Helper Functions
# ============================================================

def clean_text(text: str) -> str:
    """Clean and preprocess text."""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,;:!?\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def sentence_based_chunking(text: str, max_sentences: int = 5) -> List[str]:
    """Split text into chunks based on sentence boundaries."""
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = ' '.join(sentences[i:i + max_sentences])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks


def load_pdf(file) -> str:
    """Extract text from an uploaded PDF file."""
    from PyPDF2 import PdfReader
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def load_text_file(file) -> str:
    """Read text from an uploaded text file."""
    return file.read().decode('utf-8')


# ============================================================
# FAISS Vector Store
# ============================================================

class FAISSVectorStore:
    """FAISS-based vector store for semantic search."""

    def __init__(self, dimension: int):
        import faiss as _faiss
        self.dimension = dimension
        self.index = _faiss.IndexFlatIP(dimension)
        self.chunks = []
        self.metadata = []

    def add_documents(self, chunks: List[str], embeddings: np.ndarray,
                      metadata: List[Dict] = None):
        """Add documents to the FAISS index."""
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(normalized.astype(np.float32))
        self.chunks.extend(chunks)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in chunks])

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for top-k similar documents."""
        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        scores, indices = self.index.search(query_norm.astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(score),
                    'index': int(idx),
                    'metadata': self.metadata[idx]
                })
        return results


# ============================================================
# RAG Pipeline
# ============================================================

class RAGPipeline:
    """Complete RAG pipeline for the Streamlit app."""

    def __init__(self):
        self.embed_model = None
        self.reranker = None
        self.generator = None
        self.llm_tokenizer = None
        self.llm_model = None
        self.vector_store = None
        self.chunks = []
        self.is_loaded = False

    def load_models(self):
        """Load all ML models."""
        import torch
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with st.spinner("Loading embedding model (all-MiniLM-L6-v2)..."):
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        with st.spinner("Loading reranker (cross-encoder)..."):
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

        with st.spinner("Loading LLM (Flan-T5-Small)..."):
            self.llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

        self.is_loaded = True

    def process_documents(self, texts: List[str], filenames: List[str]):
        """Process and index uploaded documents."""
        all_chunks = []
        all_metadata = []

        for doc_idx, (text, fname) in enumerate(zip(texts, filenames)):
            cleaned = clean_text(text)
            chunks = sentence_based_chunking(cleaned, max_sentences=5)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'doc_idx': doc_idx,
                    'chunk_idx': chunk_idx,
                    'filename': fname
                })

        self.chunks = all_chunks

        # Generate embeddings
        with st.spinner(f"Generating embeddings for {len(all_chunks)} chunks..."):
            embeddings = self.embed_model.encode(all_chunks, show_progress_bar=False, batch_size=64)

        # Build FAISS index
        with st.spinner("Building vector index..."):
            self.vector_store = FAISSVectorStore(dimension=embeddings.shape[1])
            self.vector_store.add_documents(all_chunks, embeddings, all_metadata)

        return len(all_chunks)

    def query(self, question: str, use_reranking: bool = True,
              initial_k: int = 10, final_k: int = 5) -> Dict:
        """Process a question through the RAG pipeline."""
        start_time = time.time()

        # Retrieve
        query_embedding = self.embed_model.encode([question])
        initial_results = self.vector_store.search(query_embedding, top_k=initial_k)
        retrieval_time = time.time() - start_time

        # Rerank
        rerank_time = 0
        if use_reranking and initial_results:
            rerank_start = time.time()
            pairs = [[question, r['chunk']] for r in initial_results]
            ce_scores = self.reranker.predict(pairs)
            for i, result in enumerate(initial_results):
                result['rerank_score'] = float(ce_scores[i])
                result['original_rank'] = i + 1
            reranked = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)
            final_results = reranked[:final_k]
            rerank_time = time.time() - rerank_start
        else:
            final_results = initial_results[:final_k]

        # Build prompt
        context_text = "\n\n".join([
            f"[Source {i+1}] ({r.get('metadata', {}).get('filename', 'Document')}):\n{r['chunk']}"
            for i, r in enumerate(final_results)
        ])

        prompt = f"""Based on the following context, answer the question accurately.
If the answer cannot be found in the context, say "I cannot find the answer in the provided context."

Context:
{context_text}

Question: {question}

Answer:"""

        # Generate using model directly (more reliable than pipeline for seq2seq)
        gen_start = time.time()
        inputs = self.llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        import torch
        with torch.no_grad():
            outputs = self.llm_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        answer = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_time = time.time() - gen_start

        total_time = time.time() - start_time

        return {
            'question': question,
            'answer': answer,
            'sources': final_results,
            'timings': {
                'retrieval': retrieval_time,
                'reranking': rerank_time,
                'generation': gen_time,
                'total': total_time
            }
        }


# ============================================================
# Session State Initialization
# ============================================================

if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

if 'num_chunks' not in st.session_state:
    st.session_state.num_chunks = 0

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

if 'uploaded_filenames' not in st.session_state:
    st.session_state.uploaded_filenames = []


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## Settings")

    # Model Loading
    st.markdown("### 1. Load Models")
    if not st.session_state.models_loaded:
        if st.button("Load AI Models", type="primary", use_container_width=True):
            st.session_state.rag_pipeline.load_models()
            st.session_state.models_loaded = True
            st.success("Models loaded!")
            st.rerun()
    else:
        st.success("Models loaded")

    st.markdown("---")

    # Document Upload
    st.markdown("### 2. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or Text files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload one or more documents for the chatbot to search through."
    )

    if uploaded_files and st.session_state.models_loaded:
        if st.button("Process Documents", type="primary", use_container_width=True):
            texts = []
            filenames = []
            for file in uploaded_files:
                if file.name.endswith('.pdf'):
                    text = load_pdf(file)
                else:
                    text = load_text_file(file)
                texts.append(text)
                filenames.append(file.name)

            num_chunks = st.session_state.rag_pipeline.process_documents(texts, filenames)
            st.session_state.documents_loaded = True
            st.session_state.num_chunks = num_chunks
            st.session_state.uploaded_filenames = filenames
            st.session_state.chat_history = []  # Reset chat
            st.success(f"Processed {len(texts)} files into {num_chunks} chunks!")
            st.rerun()

    if st.session_state.documents_loaded:
        st.markdown("---")
        st.markdown("### Document Info")
        st.markdown(f"**Files:** {len(st.session_state.uploaded_filenames)}")
        for fname in st.session_state.uploaded_filenames:
            st.markdown(f"- {fname}")
        st.markdown(f"**Chunks:** {st.session_state.num_chunks}")

    st.markdown("---")

    # RAG Settings
    st.markdown("### 3. RAG Settings")
    use_reranking = st.checkbox("Use Reranking", value=True,
                                help="Enable cross-encoder reranking for better results")
    initial_k = st.slider("Initial retrieval (k)", 5, 20, 10,
                          help="Number of chunks to retrieve before reranking")
    final_k = st.slider("Final sources (k)", 1, 10, 5,
                        help="Number of sources to use for answer generation")

    st.markdown("---")

    # Clear Chat
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ============================================================
# Main Content
# ============================================================

st.markdown('<p class="main-header">RAG Q&A Chatbot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload documents and ask questions. '
    'Powered by Sentence Transformers, FAISS, and Flan-T5.</p>',
    unsafe_allow_html=True
)

# Status indicators
col1, col2, col3 = st.columns(3)
with col1:
    if st.session_state.models_loaded:
        st.metric("Models", "Loaded", delta="Ready")
    else:
        st.metric("Models", "Not Loaded", delta="Load in sidebar")
with col2:
    st.metric("Documents", len(st.session_state.uploaded_filenames))
with col3:
    st.metric("Chunks Indexed", st.session_state.num_chunks)

st.markdown("---")

# Chat Interface
if not st.session_state.models_loaded:
    st.info("Please load the AI models from the sidebar to get started.")
elif not st.session_state.documents_loaded:
    st.info("Please upload and process documents from the sidebar to start asking questions.")
else:
    # Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry['question'])

        with st.chat_message("assistant"):
            st.write(entry['answer'])

            # Expandable sources
            with st.expander(f"View Sources ({len(entry['sources'])} chunks)", expanded=False):
                for i, source in enumerate(entry['sources']):
                    score_key = 'rerank_score' if 'rerank_score' in source else 'score'
                    score = source[score_key]
                    fname = source.get('metadata', {}).get('filename', 'Unknown')

                    # Color code relevance
                    if score > 0.7 or (score_key == 'rerank_score' and score > 3):
                        score_class = "relevance-high"
                    elif score > 0.4 or (score_key == 'rerank_score' and score > 0):
                        score_class = "relevance-medium"
                    else:
                        score_class = "relevance-low"

                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i+1}</strong> | File: {fname} |
                        <span class="{score_class}">Relevance: {score:.4f}</span>
                        {'| Original Rank: ' + str(source.get('original_rank', '')) if 'original_rank' in source else ''}
                        <br><br>{source['chunk'][:300]}{'...' if len(source['chunk']) > 300 else ''}
                    </div>
                    """, unsafe_allow_html=True)

            # Timing info
            timings = entry['timings']
            cols = st.columns(4)
            cols[0].caption(f"Retrieval: {timings['retrieval']*1000:.0f}ms")
            cols[1].caption(f"Reranking: {timings['reranking']*1000:.0f}ms")
            cols[2].caption(f"Generation: {timings['generation']*1000:.0f}ms")
            cols[3].caption(f"Total: {timings['total']*1000:.0f}ms")

    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        # Display user message
        with st.chat_message("user"):
            st.write(question)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_pipeline.query(
                    question,
                    use_reranking=use_reranking,
                    initial_k=initial_k,
                    final_k=final_k
                )

            st.write(result['answer'])

            # Expandable sources
            with st.expander(f"View Sources ({len(result['sources'])} chunks)", expanded=True):
                for i, source in enumerate(result['sources']):
                    score_key = 'rerank_score' if 'rerank_score' in source else 'score'
                    score = source[score_key]
                    fname = source.get('metadata', {}).get('filename', 'Unknown')

                    if score > 0.7 or (score_key == 'rerank_score' and score > 3):
                        score_class = "relevance-high"
                    elif score > 0.4 or (score_key == 'rerank_score' and score > 0):
                        score_class = "relevance-medium"
                    else:
                        score_class = "relevance-low"

                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i+1}</strong> | File: {fname} |
                        <span class="{score_class}">Relevance: {score:.4f}</span>
                        {'| Original Rank: ' + str(source.get('original_rank', '')) if 'original_rank' in source else ''}
                        <br><br>{source['chunk'][:300]}{'...' if len(source['chunk']) > 300 else ''}
                    </div>
                    """, unsafe_allow_html=True)

            # Timing info
            timings = result['timings']
            cols = st.columns(4)
            cols[0].caption(f"Retrieval: {timings['retrieval']*1000:.0f}ms")
            cols[1].caption(f"Reranking: {timings['reranking']*1000:.0f}ms")
            cols[2].caption(f"Generation: {timings['generation']*1000:.0f}ms")
            cols[3].caption(f"Total: {timings['total']*1000:.0f}ms")

        # Save to chat history
        st.session_state.chat_history.append(result)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #999; font-size: 0.85rem;'>"
    "RAG Q&A Chatbot | Built with Sentence Transformers, FAISS, Flan-T5, and Streamlit"
    "</div>",
    unsafe_allow_html=True
)
