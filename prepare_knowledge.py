"""
prepare_knowledge.py  •  2025-05-15
Visual-aware knowledge extraction for RAG chatbots
– adds CLIP page-embeddings à la VDocRAG
– keeps your BM25 + text-embedding pipeline
– zero paid APIs    
"""

# ── 0) House-keeping ──────────────────────────────────────────────
import os, io, math, pickle, logging, re, sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple

BASE_DIR    = Path(__file__).parent
PDF_DIR     = BASE_DIR / "real_pdf"
PERSIST_DIR = BASE_DIR / "faiss_index"
LOG_FILE    = BASE_DIR / "process_log.txt"
os.makedirs(PERSIST_DIR, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE), filemode='a',
    format="%(asctime)s | %(levelname)-8s | %(message)s", level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
logger = logging.getLogger(__name__)
logger.addHandler(console)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # silence oneDNN

# ── 1) Imports ────────────────────────────────────────────────────
from tqdm.auto import tqdm
import fitz                 # PyMuPDF
from PIL import Image
import pytesseract
import torch, faiss, numpy as np
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# ── 2) Config knobs ───────────────────────────────────────────────
EMBED_MODEL      = "mxbai-embed-large"
EMBED_BASE       = "http://127.0.0.1:11434"
CLIP_NAME        = "openai/clip-vit-base-patch32"
CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 300
IMAGE_DPI        = 220
BM25_K1, BM25_B  = 1.5, 0.75
ALPHA_BM25, BETA_VEC, GAMMA_VIS = 0.3, 0.5, 0.2

# ── 3) Models ─────────────────────────────────────────────────────
logger.info("▶ Loading BLIP captioner (for edge cases)")
blip_proc  = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

logger.info("▶ Loading CLIP for page-level visual embeddings")
device = (
    "cuda"
    if torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).total_memory > 2.8e9
    else "cpu"
)
clip_proc  = CLIPProcessor.from_pretrained(CLIP_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()

def embed_page_image(pil_img: Image.Image) -> np.ndarray:
    inputs = clip_proc(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        vec = clip_model.get_image_features(**inputs)
    vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
    return vec.squeeze(0).cpu().numpy().astype("float32")

def caption_image(img: Image.Image) -> str:
    img = img.convert("RGB")
    inp = blip_proc(images=img, return_tensors="pt")
    out = blip_model.generate(**inp, max_length=60)
    return blip_proc.decode(out[0], skip_special_tokens=True)

# ── 4) Harvest documents ─────────────────────────────────────────
logger.info("▶ Stage 1 – scanning PDFs")
pdf_paths = sorted(PDF_DIR.rglob("*.pdf"))
docs: List[Document] = []
page_visual_vecs: List[np.ndarray] = []
page_meta: List[Dict[str, Any]] = []

for pdf_path in tqdm(pdf_paths, desc="PDFs", unit="file"):
    try:
        pdf = fitz.open(pdf_path)
    except Exception as e:
        logger.exception(f"Cannot open {pdf_path.name}: {e}")
        continue

    for pno in tqdm(range(pdf.page_count),
                    desc=f"{pdf_path.name}", leave=False):
        page = pdf.load_page(pno)
        idx = pno + 1

        txt = page.get_text().strip()
        if txt:
            docs.append(Document(page_content=txt,
                                 metadata={"source":str(pdf_path),"page":idx,"type":"text"}))

        mat = fitz.Matrix(IMAGE_DPI/72, IMAGE_DPI/72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        ocr_txt = pytesseract.image_to_string(pil, lang="eng")
        if ocr_txt.strip():
            docs.append(Document(page_content=f"[Page-OCR] {ocr_txt}",
                                 metadata={"source":str(pdf_path),"page":idx,"type":"page-ocr"}))

        try:
            vec = embed_page_image(pil)
            page_visual_vecs.append(vec)
            page_meta.append({"source":str(pdf_path),"page":idx})
        except Exception as e:
            logger.warning(f"CLIP failed (p{idx}, {pdf_path.name}): {e}")

        if not txt and not ocr_txt.strip():
            try:
                cap = caption_image(pil)
                docs.append(Document(page_content=f"[Page-caption] {cap}",
                                     metadata={"source":str(pdf_path),"page":idx,"type":"page-caption"}))
            except Exception as e:
                logger.debug(f"BLIP caption error p{idx}: {e}")

logger.info(f"Collected {len(docs)} textual docs and {len(page_visual_vecs)} visual pages")

# ── 5) Chunking ───────────────────────────────────────────────────
logger.info("▶ Stage 2 – chunking text")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
chunks: List[Document] = []
for d in tqdm(docs, desc="Chunking"):
    chunks.extend(splitter.split_documents([d]))
logger.info(f"Produced {len(chunks)} text chunks")
if not chunks:
    logger.error("No text chunks generated; exiting.")
    sys.exit(1)

# ── 6) BM25 ───────────────────────────────────────────────────────
logger.info("▶ Stage 3 – BM25")
class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def tokenize(self, t: str) -> List[str]:
        return re.findall(r"\w+", t.lower())

    def fit(self, corpus: List[str]):
        self.N = len(corpus)
        if self.N == 0:
            logger.warning("Empty corpus for BM25; skipping.")
            self.av, self.tf, self.idf = 0.0, [], {}
            return self

        tok = [self.tokenize(doc) for doc in corpus]
        self.len = [len(t) for t in tok]
        self.av  = sum(self.len) / float(self.N)
        self.tf  = [Counter(t)    for t in tok]
        df      = Counter(w for t in tok for w in set(t))
        self.idf = {w: math.log((self.N-df[w]+0.5)/(df[w]+0.5)+1) for w in df}
        return self

    def score(self, q: str, idx: int) -> float:
        tf, dl = self.tf[idx], self.len[idx]
        s = 0.0
        for w in self.tokenize(q):
            if w not in self.idf: continue
            f = tf.get(w,0)
            s += self.idf[w] * (f*(self.k1+1)) / (f + self.k1*(1-self.b+self.b*dl/self.av))
        return s

    def search(self, q: str, top_k: int=8) -> List[Tuple[int,float]]:
        scores = [(i, self.score(q,i)) for i in range(self.N)]
        return sorted(scores, key=lambda x:x[1], reverse=True)[:top_k]

bm25 = BM25(BM25_K1, BM25_B).fit([c.page_content for c in chunks])
pickle.dump(bm25, open(PERSIST_DIR/"bm25.pkl","wb"))

# ── 7) Text embeddings + Faiss ───────────────────────────────────
logger.info("▶ Stage 4 – text embeddings")
emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=EMBED_BASE)
dim_text = len(emb.embed_documents([""])[0])
idx_text = faiss.IndexFlatL2(dim_text)
for i, ch in enumerate(tqdm(chunks,desc="Text-index"), start=1):
    try:
        v = np.asarray(emb.embed_documents([ch.page_content])[0],dtype="float32")
        idx_text.add(v.reshape(1,-1))
    except Exception as e:
        logger.warning(f"Text embed fail chunk#{i}: {e}")
faiss.write_index(idx_text, str(PERSIST_DIR/"index_text.faiss"))
pickle.dump(chunks, open(PERSIST_DIR/"chunks.pkl","wb"))

# ── 8) Visual Faiss index ────────────────────────────────────────
logger.info("▶ Stage 5 – visual index (CLIP)")
dim_vis = page_visual_vecs[0].shape[0] if page_visual_vecs else 512
idx_vis = faiss.IndexFlatIP(dim_vis)
idx_vis.add(np.vstack(page_visual_vecs))
faiss.write_index(idx_vis, str(PERSIST_DIR/"index_visual.faiss"))
pickle.dump(page_meta, open(PERSIST_DIR/"page_meta.pkl","wb"))

# ── 9) Hybrid search helper ─────────────────────────────────────
logger.info("▶ Stage 6 – hybrid search helper")
class HybridSearcher:
    def __init__(self, bm25, idx_text, idx_vis, chunks, pmeta,
                 alpha=ALPHA_BM25, beta=BETA_VEC, gamma=GAMMA_VIS):
        self.bm25, self.idx_text, self.idx_vis = bm25, idx_text, idx_vis
        self.chunks, self.pmeta = chunks, pmeta
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        # create embedding client at runtime only
        self.emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=EMBED_BASE)

    def __getstate__(self):
        state = self.__dict__.copy()
        # drop embedding client
        state.pop("emb", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # restore embedding client
        self.emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=EMBED_BASE)

    def search(self, query: str, top_k: int=5) -> List[Document]:
        bm = self.bm25.search(query, top_k*3)
        bm_max = bm[0][1] if bm else 1.0

        qv = np.asarray(self.emb.embed_documents([query])[0],dtype="float32").reshape(1,-1)
        D, I = self.idx_text.search(qv, top_k*3)
        vec_max = max(1e-4, np.max(D))

        with torch.no_grad():
            cq = clip_model.get_text_features(**clip_proc(text=query,return_tensors="pt").to(device))
        cq = torch.nn.functional.normalize(cq, p=2, dim=-1).cpu().numpy().astype("float32")
        Dv, Iv = self.idx_vis.search(cq, top_k*3)
        vis_max = max(1e-4, np.max(Dv))

        scores: Dict[int,float] = {}
        for idx, s in bm:
            scores[idx] = scores.get(idx,0.0) + self.alpha*(s/bm_max)
        for rank, idx in enumerate(I[0]):
            scores[idx] = scores.get(idx,0.0) + self.beta*(1 - D[0][rank]/vec_max)
        for rank, pidx in enumerate(Iv[0]):
            m = self.pmeta[pidx]
            same = next((i for i,c in enumerate(self.chunks)
                         if c.metadata["source"]==m["source"] and c.metadata["page"]==m["page"]), None)
            if same is not None:
                scores[same] = scores.get(same,0.0) + self.gamma*(Dv[0][rank]/vis_max)

        best = sorted(scores.items(), key=lambda x:x[1], reverse=True)[:top_k]
        return [self.chunks[i] for i,_ in best]

hybrid = HybridSearcher(bm25, idx_text, idx_vis, chunks, page_meta)
pickle.dump(hybrid, open(PERSIST_DIR/"hybrid_searcher.pkl","wb"))

logger.info("✅ Finished. Indexes & helper saved in ./faiss_index/")
