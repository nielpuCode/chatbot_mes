import os
from random import seed
import time
import json
import re
import pickle
import faiss
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify

from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
import spacy
from nltk.corpus import wordnet
import nltk
import logging
from collections import Counter
import math
from tqdm.auto import tqdm
import random

# 1) Fix randomness
random.seed(42)
np.random.seed(42)

# 2) Use all CPU cores in FAISS
faiss.omp_set_num_threads(os.cpu_count())

# Set up logging
logging.basicConfig(
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# by running ==> nvidia-smi -L
os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-5ade70c7-7865-b3c0-fcde-8fd5281d82d6'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Download necessary NLTK data
try:
    wordnet.synsets('test')
except LookupError:
    nltk.download('wordnet')

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.resolve()
INDEX_PATH     = BASE_DIR / "faiss_index" / "index.faiss"
CHUNKS_PATH    = BASE_DIR / "faiss_index" / "chunks.pkl"
LOG_PATH       = BASE_DIR / "chat_log.json"
BM25_PATH      = BASE_DIR / "faiss_index" / "bm25.pkl"
HYBRID_SEARCHER_PATH = BASE_DIR / "faiss_index" / "hybrid_searcher.pkl"
MEMORY_PATH    = BASE_DIR / "memory.pkl"
# Information in faiss_index consist of Corpus size: 3962, Average document length: 93.26, Vocabulary size (unique tokens): 2729,Total vectors indexed: 3962, Vector dimensionality: 1024, Index type: IndexFlatL2, Total chunks: 3962, Total context length: 3309923 characters

EMBED_MODEL     = "bge-m3:567m"
EMBED_BASE_URL = "http://127.0.0.1:11434"
LLM_MODEL      = "qwen3:1.7b"
TOP_K_FINAL    = 15 
BM25_SCALE     = 10.0  # Fixed scaling factor for BM25 scores

# ── BOOTSTRAP chat log ──────────────────────────────────────────────────────────
if not LOG_PATH.exists():
    LOG_PATH.write_text("[]")

def log_exchange(user, assistant):
    entry = {"timestamp": time.time(), "user": user, "ai": assistant}
    try:
        history = json.loads(LOG_PATH.read_text())
    except Exception:
        history = []
    history.append(entry)
    LOG_PATH.write_text(json.dumps(history, indent=2))

# ── Helper Functions ─────────────────────────────────────────────────────────────
def normalize_vectors(index):
    """Normalize all vectors in a FAISS index to unit length"""
    n = index.ntotal
    if n == 0:
        return index

    # Get all vectors
    vectors = np.zeros((n, index.d), dtype=np.float32)
    for i in range(n):
        vector = np.zeros(index.d, dtype=np.float32)
        index.reconstruct(i, vector)
        vectors[i] = vector

    # Normalize vectors
    faiss.normalize_L2(vectors)

    # Create a new index and add normalized vectors
    new_index = faiss.IndexFlatIP(index.d)  # Use Inner Product for cosine similarity
    new_index.add(vectors)
    return new_index

def warm_up_model(model, queries=["Hello, how are you?", "What is the weather today?", "Tell me about machine learning"]):
    """Warm up a model with some example queries"""
    logger.info("Warming up model...")
    for query in queries:
        try:
            if hasattr(model, 'embed_documents'):
                model.embed_documents([query])
            elif hasattr(model, 'invoke'):
                model.invoke(query)
        except Exception as e:
            logger.warning(f"Model warmup failed with error: {e}")
    logger.info("Model warmup complete")

def get_synonyms(word):
    """Get synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def preprocess_text(text):
    """Preprocess text using spaCy for tokenization, lemmatization, and stop word removal."""
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

class BM25:
    """BM25 implementation for information retrieval"""

    def __init__(self, k1=1.5, b=0.75, nlp=None):
        """Initialize BM25 with parameters

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenized_corpus = []
        self.nlp = nlp  # spaCy NLP object

    def tokenize(self, text):
        """Tokenize using spaCy with lemmatization and stop word removal."""
        if self.nlp:
            return preprocess_text(text).split()
        else:
            return re.findall(r'\w+', text.lower())

    def fit(self, corpus):
        """Fit BM25 parameters to the corpus

        Args:
            corpus: List of text documents
        """
        self.corpus_size = len(corpus)
        logger.info(f"Fitting BM25 on {self.corpus_size} documents")

        # Tokenize the corpus
        self.tokenized_corpus = [self.tokenize(doc) for doc in tqdm(corpus, desc="Tokenizing")]

        # Calculate document lengths and average document length
        self.doc_len = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size

        # Count term frequencies per document
        self.doc_freqs = []
        for doc in tqdm(self.tokenized_corpus, desc="Counting terms"):
            term_freqs = Counter(doc)
            self.doc_freqs.append(term_freqs)

        # Calculate IDF values for all terms in corpus
        df = {}
        for doc_freqs in self.doc_freqs:
            for term in doc_freqs.keys():
                if term not in df:
                    df[term] = 0
                df[term] += 1

        # Calculate IDF score for each term
        for term, freq in df.items():
            self.idf[term] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

        logger.info(f"BM25 fitted with {len(self.idf)} unique terms")
        return self

    def score(self, query, doc_idx):
        """Calculate BM25 score for a query and document

        Args:
            query: Tokenized query
            doc_idx: Document index in corpus

        Returns:
            float: BM25 score
        """
        score = 0
        doc_len = self.doc_len[doc_idx]
        doc_freqs = self.doc_freqs[doc_idx]

        for term in query:
            if term not in self.idf:
                continue

            # Term frequency in document
            f = doc_freqs.get(term, 0)

            # BM25 scoring function
            numerator = self.idf[term] * f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += numerator / denominator

        return score

    def search(self, query, top_k=5):
        """Search corpus for documents matching query

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List[Tuple[int, float]]: List of (doc_idx, score) tuples
        """
        tokenized_query = self.tokenize(query)

        scores = []
        for i in range(self.corpus_size):
            score = self.score(tokenized_query, i)
            scores.append((i, score))

        # Sort by decreasing score
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

class HybridSearcher:
    """Class to combine BM25 and vector search results"""

    def __init__(self, bm25, faiss_index, chunks, alpha=0.5, embeddings_model=None):
        """Initialize hybrid searcher

        Args:
            bm25: BM25 instance
            faiss_index: FAISS index 
            chunks: Document chunks
            alpha: Weight factor for combining scores (0.5 = equal weight)
        """
        self.bm25 = bm25
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.alpha = alpha
        self.embeddings_model = embeddings_model

    def search(self, query, top_k=5):
        """Perform hybrid search with consistent normalization

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List[Dict]: List of results with document and score
        """
        # BM25 search
        bm25_results = self.bm25.search(query, top_k=top_k*2)

        # Vector search
        query_vec = self.embeddings_model.embed_documents([query])[0]
        # Normalize query vector for consistent cosine similarity
        query_vec = query_vec / np.linalg.norm(query_vec)
        D, I = self.faiss_index.search(np.array([query_vec], dtype="float32"), top_k*2)

        # Use fixed scaling for normalization instead of dynamic max values
        all_scores = {}

        # Add BM25 scores - use a fixed normalization constant
        for doc_idx, score in bm25_results:
            if doc_idx not in all_scores:
                all_scores[doc_idx] = {"bm25": 0, "vector": 0}
            all_scores[doc_idx]["bm25"] = score / BM25_SCALE

        # Add vector scores
        for i, doc_idx in enumerate(I[0]):
            if doc_idx not in all_scores:
                all_scores[doc_idx] = {"bm25": 0, "vector": 0}
            # If using IndexFlatIP, scores are directly similarities
            # If using IndexFlatL2, convert distances to similarities
            similarity = D[0][i]
            if isinstance(self.faiss_index, faiss.IndexFlatL2):
                # Convert L2 distance to similarity (smaller = better)
                similarity = 1.0 / (1.0 + D[0][i])
            all_scores[doc_idx]["vector"] = similarity

        # Calculate combined scores
        results = []
        for doc_idx, scores in all_scores.items():
            combined_score = self.alpha * scores["bm25"] + (1 - self.alpha) * scores["vector"]
            results.append({
                "document": self.chunks[doc_idx],
                "bm25_score": scores["bm25"],
                "vector_score": scores["vector"],
                "combined_score": combined_score
            })

        # Sort by combined score
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_k]

    def __getstate__(self):
        # copy everything except the locks / sessions
        state = self.__dict__.copy()
        state['faiss_index']      = None
        state['embeddings_model'] = None
        return state

    def __setstate__(self, state):
        # restore the pickled state…
        self.__dict__.update(state)
        # …and then recreate the dropped pieces
        from langchain_ollama import OllamaEmbeddings
        import faiss

        self.embeddings_model = OllamaEmbeddings(model=EMBED_MODEL,
                                                 base_url=EMBED_BASE_URL)
        faiss_index_path = BASE_DIR / "faiss_index" / "index.faiss"
        self.faiss_index = faiss.read_index(str(faiss_index_path))

        # Check if it's an L2 index and convert to IP if needed
        if isinstance(self.faiss_index, faiss.IndexFlatL2):
            logger.info("Converting L2 index to IP index for better consistency")
            self.faiss_index = normalize_vectors(self.faiss_index)


# ── 1) LOAD FAISS & CHUNKS ──────────────────────────────────────────────────────
logger.info("▶ Loading embedding model …")
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=EMBED_BASE_URL)
warm_up_model(embeddings)  # Warm up embeddings model

logger.info(f"▶ Loading FAISS index  ({INDEX_PATH.name}) …")
faiss_index = faiss.read_index(str(INDEX_PATH))

# Convert to inner product (cosine similarity) index if not already
if isinstance(faiss_index, faiss.IndexFlatL2):
    logger.info("Converting FAISS index to normalized IndexFlatIP for consistent retrieval")
    faiss_index = normalize_vectors(faiss_index)

logger.info(f"▶ Loading chunks       ({CHUNKS_PATH.name}) …")
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

logger.info(f"▶ Loading BM25 index       ({BM25_PATH.name}) …")
with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)

# ── 1.5) Initialize spaCy ───────────────────────────────────────────────────────────────
logger.info("▶ Stage 0: Initializing spaCy")
nlp = spacy.load("en_core_web_sm", disable=["ner"])  # Disable NER for speed if not needed

# ── 2) LOAD HybridSearcher ──────────────────────────────────────────────────────────────
logger.info("▶ Loading Hybrid Searcher …")
with open(HYBRID_SEARCHER_PATH, "rb") as f:
    hybrid_searcher = pickle.load(f)


# Update the hybrid searcher's FAISS index with our normalized one
hybrid_searcher.faiss_index = faiss_index
hybrid_searcher.embeddings_model = embeddings

# ── 3) PROMPT + LLM SETUP ───────────────────────────────────────────────────────────────
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are MyBot, a helpful assistant. Answer based on the context provided below. "
     "Only use information found in the context to answer the question. "
     "If the context doesn't contain the answer, try to find it again, then if still doesnt found it, say 'I don't have that information.' "
     "\n\nContext:\n{context}\n\n"),
     "Using only the provided retrieved documents context, answer the following question. Do not add any external knowledge.",
     "Break down the following problem into logical steps and solve it step by step using the retrieved data.  Refine it based on retrieved documents to improve accuracy.\n\n",
     "Reply in the user's language. If the user ask in Indonesian language, reply it with Indonesian langauge too. If the user ask in English language, reply it in english language too.\n\n",
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{query}")
])

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Fixed parameters for deterministic output
llm = Ollama(model=LLM_MODEL, temperature=0.0, top_p=1.0, repeat_penalty=1.1)
llm_chain = prompt_template | llm

# ── 4) PREPROCESS to strip <think>…</think> ──────────────────────────────────────────
def preprocess_answer(ans: str) -> str:
    # remove any <think>…</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", ans, flags=re.DOTALL)
    return cleaned.strip()

# ── 5) MEMORY & FLASK ───────────────────────────────────────────────────────────────
memory = ConversationBufferWindowMemory(k=5,
                                        memory_key="chat_history",
                                        return_messages=True)
app = Flask(__name__)

# ── Load memory from disk if exists ───────────────────────────────────────────────────
if MEMORY_PATH.exists():
    try:
        with open(MEMORY_PATH, "rb") as f:
            loaded_memory = pickle.load(f)
            memory.chat_memory.messages = loaded_memory.chat_memory.messages
            logger.info(f"Loaded {len(memory.chat_memory.messages)} messages from persistent memory")
    except Exception as e:
        logger.error(f"Error loading memory: {e}")


# ── Updated RAG function for more consistency ───────────────────────────────────────────
def _run_rag(query: str):
    logger.info(f"\n▶ QUERY: {query!r}")

    # 1️⃣ Preprocess the query - keep it simple and deterministic
    processed_query = preprocess_text(query)

    # 2️⃣ Use the processed query directly without synonym expansion
    # This makes the retrieval more consistent between runs

    # 3️⃣ Hybrid retrieval with our improved consistency
    results = hybrid_searcher.search(processed_query, top_k=TOP_K_FINAL)

    # 4️⃣ Build context from the returned Document objects
    snippets = []
    for result in results:
        doc = result["document"]              # extract the actual Document
        page = doc.metadata.get("page", "unknown")
        text = doc.page_content.replace("\n", " ")
        snippets.append(f"[Page {page}] {text}")
    context = "\n\n---\n\n".join(snippets)

    # 5️⃣ Invoke the LLM with the assembled context
    hist = memory.chat_memory.messages[-10:] if memory.chat_memory.messages else []
    raw = llm_chain.invoke({
        "chat_history": hist,
        "context": context,
        "query": query
    })

    # 6️⃣ Strip out any <think>…</think> tags
    answer = preprocess_answer(raw)

    return answer, context

# ── Add a route to inspect retrieval results for debugging ────────────────────────────
@app.route("/debug_retrieval", methods=["POST"])
def debug_retrieval():
    data = request.get_json()
    q = data["query"]

    # Process query same as in _run_rag
    processed_query = preprocess_text(q)

    # Get results
    results = hybrid_searcher.search(processed_query, top_k=TOP_K_FINAL)

    # Return detailed scores and snippets
    return jsonify({
        "query": q,
        "processed_query": processed_query,
        "results": [
            {
                "text": r["document"].page_content[:200] + "..." if len(r["document"].page_content) > 200 else r["document"].page_content,
                "page": r["document"].metadata.get("page", "unknown"),
                "bm25_score": r["bm25_score"],
                "vector_score": r["vector_score"],
                "combined_score": r["combined_score"]
            }
            for r in results
        ]
    })

@app.route("/", methods=["GET","POST"])
def home():
    ui = [{"type":m.type,"content":m.content} for m in memory.chat_memory.messages]
    if request.method=="POST":
        q = request.form["query"]
        t0 = time.time()
        ans, _ = _run_rag(q)
        dur = time.time()-t0

        memory.chat_memory.add_user_message(q)
        memory.chat_memory.add_ai_message(ans)
        
        ui += [
            {"type":"human","content":q},
            {"type":"ai","content":ans},
            {"type":"meta","content":f"(pipeline {dur:.2f}s)"}
        ]
        log_exchange(q, ans)

    return render_template("index.html", messages=ui)

@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    q = data["query"]
    t0 = time.time()
    ans, _ = _run_rag(q)
    dur = time.time()-t0

    memory.chat_memory.add_user_message(q)
    memory.chat_memory.add_ai_message(ans)
     # Save memory after updating
    log_exchange(q, ans)
    return jsonify({"answer": ans, "duration": dur})

@app.route("/reset", methods=["POST"])
def reset_memory():
    memory.clear()
     # Save the empty memory state
    return jsonify({"status":"success","message":"Conversation reset"})

@app.route("/debug", methods=["GET"])
def debug():
    return jsonify({
        "chunks": len(chunks),
        "vectors": faiss_index.ntotal,
        "memory": len(memory.chat_memory.messages),
        "index_type": str(type(faiss_index).__name__)
    })

if __name__ == "__main__":
    os.system("cls" if os.name=="nt" else "clear")
    logger.info("🚀 Starting Enhanced RAG server with consistency improvements…")
    logger.info(f" • Chunks      : {len(chunks)}")
    logger.info(f" • FAISS vecs  : {faiss_index.ntotal}")
    logger.info(f" • FAISS type  : {type(faiss_index).__name__}")
    logger.info(f" • LLM model   : {LLM_MODEL}")
    app.run(debug=True)

