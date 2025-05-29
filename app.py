import os
import sys
import time
import json
import re
import pickle
import math
import faiss
import numpy as np

from pathlib import Path
from collections import Counter
from flask import Flask, render_template, request, jsonify

import torch
from transformers import CLIPProcessor, CLIPModel

from langchain.schema import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama

# ── CONFIG ─────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.resolve()
PERSIST_DIR     = BASE_DIR / "faiss_index"
TEXT_IDX_PATH   = PERSIST_DIR / "index_text.faiss"
VISUAL_IDX_PATH = PERSIST_DIR / "index_visual.faiss"
BM25_PATH       = PERSIST_DIR / "bm25.pkl"
CHUNKS_PATH     = PERSIST_DIR / "chunks.pkl"
PAGE_META_PATH  = PERSIST_DIR / "page_meta.pkl"
LOG_PATH        = BASE_DIR / "chat_log.json"

EMBED_MODEL     = "mxbai-embed-large"
EMBED_BASE_URL  = "http://127.0.0.1:11434"
CLIP_NAME       = "openai/clip-vit-base-patch32"
LLM_MODEL       = "qwen3:1.7b"

TOP_K_FINAL     = 12
ALPHA_BM25      = 0.3   # BM25 weight
BETA_VEC        = 0.5   # text-vector weight
GAMMA_VIS       = 0.2   # visual weight

# ── BOOTSTRAP chat log ───────────────────────────────────────────────────
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

# ── 1) LOAD EMBEDDERS & INDICES ─────────────────────────────────────────
print("▶ Loading text-embedding model …")
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=EMBED_BASE_URL)

print("▶ Loading CLIP for visual queries …")
device    = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 2.8e9 else "cpu"
clip_proc = CLIPProcessor.from_pretrained(CLIP_NAME)
clip_model= CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()

print("▶ Loading text FAISS index …")
idx_text  = faiss.read_index(str(TEXT_IDX_PATH))

print("▶ Loading visual FAISS index …")
idx_vis   = faiss.read_index(str(VISUAL_IDX_PATH))

# ── 1.5) Define BM25 so pickle.load can find it ─────────────────────────
class BM25:
    def __init__(self, k1=1.5, b=0.75):
        # these will be overwritten by pickle state
        self.k1, self.b = k1, b
    def tokenize(self, t: str):
        return re.findall(r"\w+", t.lower())
    def score(self, q: str, idx: int) -> float:
        tf, dl = self.tf[idx], self.len[idx]
        s = 0.0
        for w in self.tokenize(q):
            if w not in self.idf:
                continue
            f = tf.get(w, 0)
            s += self.idf[w] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.av))
        return s
    def search(self, q: str, top_k: int = 8):
        scores = [(i, self.score(q, i)) for i in range(self.N)]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

print("▶ Loading BM25 index …")
with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)

print("▶ Loading chunks …")
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print("▶ Loading page metadata …")
with open(PAGE_META_PATH, "rb") as f:
    page_meta = pickle.load(f)

# ── 2) HYBRID SEARCHER ────────────────────────────────────────────────────
class HybridSearcher:
    def __init__(self,
                 bm25_index,
                 idx_text: faiss.Index,
                 idx_vis: faiss.Index,
                 chunks_list: list[Document],
                 page_meta: list[dict],
                 alpha: float, beta: float, gamma: float):
        self.bm25      = bm25_index
        self.idx_text  = idx_text
        self.idx_vis   = idx_vis
        self.chunks    = chunks_list
        self.page_meta = page_meta
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.emb = embeddings

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        # 1) BM25 hits
        bm     = self.bm25.search(query, top_k*3)
        # bm_max = bm[0][1] if bm else 1.0
        top_score = bm[0][1] if bm else 0.0
        bm_max    = top_score if top_score > 1e-8 else 1.0

        # 2) text-vector hits
        qvec   = np.asarray(self.emb.embed_documents([query])[0], dtype="float32").reshape(1, -1)
        D_t, I_t = self.idx_text.search(qvec, top_k*3)
        vec_max  = max(1e-4, np.max(D_t))

        # 3) visual hits via CLIP text encoder
        with torch.no_grad():
            clip_q = clip_model.get_text_features(**clip_proc(text=query, return_tensors="pt").to(device))
        clip_q = torch.nn.functional.normalize(clip_q, p=2, dim=-1).cpu().numpy().astype("float32")
        D_v, I_v = self.idx_vis.search(clip_q, top_k*3)
        vis_max  = max(1e-4, np.max(D_v))

        # 4) fuse scores
        scores = {}
        for idx, sc in bm:
            scores[idx] = scores.get(idx, 0.0) + self.alpha * (sc / bm_max)
        for rank, idx in enumerate(I_t[0]):
            scores[idx] = scores.get(idx, 0.0) + self.beta * (1 - D_t[0][rank] / vec_max)
        for rank, pidx in enumerate(I_v[0]):
            meta = self.page_meta[pidx]
            same = next(
                (i for i,c in enumerate(self.chunks)
                 if c.metadata["source"] == meta["source"] and c.metadata["page"] == meta["page"]),
                None
            )
            if same is not None:
                scores[same] = scores.get(same, 0.0) + self.gamma * (D_v[0][rank] / vis_max)

        best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [ self.chunks[i] for i,_ in best ]

hybrid_searcher = HybridSearcher(
    bm25, idx_text, idx_vis, chunks, page_meta,
    alpha=ALPHA_BM25, beta=BETA_VEC, gamma=GAMMA_VIS
)

# ── 3) PROMPT + LLM SETUP ────────────────────────────────────────────────
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system",
#         "You are MyBot, a helpful assistant. Answer based on the context provided below. "
#         "Only use information found in the context to answer the question. Dont make up anything. "
#         "If the context doesn't contain the answer, try to find it again or summarize based what you know from everything in the context, then answer it within mentioning the disclaimer, then if still doesnt found/concluding anything, say 'I don't have that information in my knowledge.' "
#         "Be specific and cite page numbers when possible.\n\n"
#         "Always reply in the same language as the user’s latest query. If the user’s input is in English, respond in English. If it is in Bahasa Indonesia, respond in Indonesian. /no_think\n\n"
#         "Context:\n{context}\n\n"),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{query}")
# ])

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are **MyBot**, a reliable assistant that answers questions *only* from the information in **Context**.

        ### Language Policy
        - Always reply in the same language as the user’s latest query.
        - If the user’s query is in **English**, answer in **English**.
        - If the user’s query is in **Bahasa Indonesia**, answer in **Bahasa Indonesia**.
        - Never mix or switch to any other language.

        ### Answering Rules
        1. Use *only* facts present in **Context** — do **not** invent information.  
        2. Cite page numbers when possible. 
        3. If the answer is implied (not explicit), Give them a heads up. 
        4. If the information truly is not in Context, reply **exactly**:  
        • English: *“I’m sorry, I don’t have that information in my knowledge.”*  
        • Indonesian: *“Maaf, aku tidak menemukan informasi tersebut di konteks.”*  
        5. Keep responses concise and precise.

        /no_think

        Context:
        {context}
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{query}")
])

llm       = Ollama(model=LLM_MODEL)
llm_chain = prompt_template | llm

def preprocess_answer(ans: str) -> str:
    return re.sub(r"<think>.*?</think>", "", ans, flags=re.DOTALL).strip()

# ── 4) MEMORY & FLASK ─────────────────────────────────────────────────────
memory = ConversationBufferWindowMemory(k=5,
                                        memory_key="chat_history",
                                        return_messages=True)
app = Flask(__name__)

def _run_rag(query: str):
    print(f"\n▶ QUERY: {query!r}")
    docs = hybrid_searcher.search(query, top_k=TOP_K_FINAL)
    snippets = []
    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        txt  = doc.page_content.replace("\n", " ")
        snippets.append(f"[Page {page}] {txt}")
    context = "\n\n---\n\n".join(snippets)

    hist = memory.chat_memory.messages[-10:] if memory.chat_memory.messages else []
    raw  = llm_chain.invoke({"chat_history": hist, "context": context, "query": query})
    return preprocess_answer(raw), context

@app.route("/", methods=["GET","POST"])
def home():
    ui = [{"type":m.type, "content":m.content}
          for m in memory.chat_memory.messages]
    if request.method=="POST":
        q   = request.form["query"]
        t0  = time.time()
        ans, _ = _run_rag(q)
        dur = time.time() - t0

        memory.chat_memory.add_user_message(q)
        memory.chat_memory.add_ai_message(ans)
        ui += [
            {"type":"human","content":q},
            {"type":"ai",   "content":ans},
            {"type":"meta", "content":f"(pipeline {dur:.2f}s)"}
        ]
        log_exchange(q, ans)

    return render_template("index.html", messages=ui)

@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    q    = data["query"]
    t0   = time.time()
    ans, _ = _run_rag(q)
    dur  = time.time() - t0

    memory.chat_memory.add_user_message(q)
    memory.chat_memory.add_ai_message(ans)
    log_exchange(q, ans)
    return jsonify({"answer": ans, "duration": dur})

@app.route("/reset", methods=["POST"])
def reset_memory():
    memory.clear()
    return jsonify({"status":"success","message":"Conversation reset"})

@app.route("/debug", methods=["GET"])
def debug():
    return jsonify({
        "chunks":      len(chunks),
        "text_vecs":   idx_text.ntotal,
        "visual_pages":idx_vis.ntotal,
        "memory":      len(memory.chat_memory.messages)
    })

if __name__=="__main__":
    os.system("cls" if os.name=="nt" else "clear")
    print("🚀 Starting RAG server…")
    print(f" • Chunks       : {len(chunks)}")
    print(f" • Text vecs    : {idx_text.ntotal}")
    print(f" • Visual pages : {idx_vis.ntotal}")
    print(f" • LLM model    : {LLM_MODEL}")
    app.run(debug=True)
