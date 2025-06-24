import io
import os
import pickle
import logging
from pathlib import Path
import time
from sklearn.preprocessing import normalize


t0 = time.time()
# ── 0) Setup ──────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PDF_DIR     = BASE_DIR / "real_pdf"
PERSIST_DIR = BASE_DIR / "faiss_index"
LOG_FILE    = BASE_DIR / "process_log.txt"

# make sure output dirs exist
os.makedirs(PERSIST_DIR, exist_ok=True)


# logging to both console & file
logging.basicConfig(
    filename=str(LOG_FILE),
    filemode='a',
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
logger = logging.getLogger()
logger.addHandler(console)

# quell oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ── Imports ─────────────────────────────────────────────────────────
import fitz                                # PyMuPDF
from PIL import Image
import pytesseract
import camelot
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_ollama import OllamaEmbeddings
import faiss
import numpy as np
from tqdm.auto import tqdm
import re
import math
from collections import Counter
from typing import List, Dict, Tuple, Any

import spacy
from nltk.corpus import wordnet
import nltk

# Download necessary NLTK data
try:
    wordnet.synsets('test')
except LookupError:
    nltk.download('wordnet')

# ── Config ─────────────────────────────────────────────────────────
EMBED_MODEL     = "bge-m3:567m"
EMBED_BASE      = "http://127.0.0.1:11434"
TOP_K           = 8           # Increased from 4 for better recall with large documents
CHUNK_SIZE      = 800         # Decreased from 1000 for more granular chunks
CHUNK_OVERLAP   = 300         # Increased from 200 to ensure context preservation
IMAGE_THRESHOLD = 50
HYBRID_ALPHA    = 0.3         # Weight for BM25 scores (lower = favor vector search)

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75

# ── 1) Initialize spaCy ────────────────────────────────────────────────────────
logger.info("▶ Stage 0: Initializing spaCy")
nlp = spacy.load("en_core_web_lg", disable=["ner"])  # Disable NER for speed if not needed

# ── 1.5) Define helper functions ──────────────────────────────────────────────

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

# ── 2) BLIP Captioner ──────────────────────────────────────────────
logger.info("▶ Stage 1: Initializing BLIP image‐captioner")
processor     = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(img: Image.Image) -> str:
    img = img.convert("RGB")
    inp = processor(images=img, return_tensors="pt")
    out = caption_model.generate(**inp)
    return processor.decode(out[0], skip_special_tokens=True)

# ── 3) Find PDFs ────────────────────────────────────────────────────
pdf_paths = sorted(PDF_DIR.rglob("*.pdf"))
logger.info(f"▶ Stage 2: Found {len(pdf_paths)} PDF(s) in '{PDF_DIR}'")

docs = []

for pdf_path in tqdm(pdf_paths, desc="PDFs", unit="file"):
    logger.info(f"── Processing PDF: {pdf_path.name}")
    try:
        pdf = fitz.open(pdf_path)
    except Exception as e:
        logger.exception(f"Failed to open '{pdf_path.name}': {e}")
        continue

    for pno in range(pdf.page_count):
        page_index = pno + 1
        page = pdf.load_page(pno)

        # 3a) pure text
        txt = page.get_text().strip()
        if txt:
            docs.append(Document(
                page_content=txt,
                metadata={"source": str(pdf_path), "page": page_index, "type":"text"}
            ))
            logger.info(f"Extracted text from page {page_index}")
        else:
            logger.info(f"No plain text on page {page_index}")

        # 3b) table extraction via Camelot
        try:
            tables = camelot.read_pdf(str(pdf_path), pages=str(page_index), flavor='stream')  # Use 'stream' flavor
            for ti, table in enumerate(tables, start=1):
                # Try different table conversion methods
                try:
                    html = table.df.to_html(index=False)
                    docs.append(Document(
                        page_content=f"[Table {ti}, in page {page_index}]\n{html}",
                        metadata={"source": str(pdf_path), "page": page_index, "type":"table", "table_index":ti}
                    ))
                    logger.info(f"Extracted table #{ti} on page {page_index} as HTML")
                except Exception as e:
                    logger.warning(f"Failed to convert table to HTML: {e}.  Trying string representation.")
                    table_string = table.df.to_string(index=False)  # Fallback to string
                    docs.append(Document(
                        page_content=f"[Table {ti}, in page {page_index}]\n{table_string}",
                        metadata={"source": str(pdf_path), "page": page_index, "type":"table", "table_index":ti}
                    ))
                    logger.info(f"Extracted table #{ti} on page {page_index} as string")
        except Exception as e:
            logger.exception(f"Table extraction failed on page {page_index}: {e}")

        # 3c) images
        images = page.get_images(full=True)
        logger.info(f"Found {len(images)} image‐XObjects on page {page_index}")

        # if too many XObjects, rasterize full page & OCR+caption it
        if len(images) > IMAGE_THRESHOLD:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
                img = Image.frombytes("RGB", [pix.width,pix.height], pix.samples)

                # Improved OCR:  Specify language and use a better engine if needed
                ocr_t = pytesseract.image_to_string(img, lang='eng', config='--oem 1 --psm 3') # OEM 1 is LSTM engine
                if ocr_t.strip():
                    docs.append(Document(
                        page_content=f"[Full‐page OCR, in page{page_index}]\n{ocr_t}",
                        metadata={"source":str(pdf_path),"page":page_index,"type":"fullpage-ocr"}
                    ))
                    logger.info(f"OCR full‐page in page{page_index}")
                # then BLIP caption
                caption = caption_image(img)
                docs.append(Document(
                    page_content=f"[Full‐page caption, in page{page_index}]: {caption}",
                    metadata={"source":str(pdf_path),"page":page_index,"type":"fullpage-caption"}
                ))
                logger.info(f"Caption full‐page in page{page_index}: {caption}")
            except Exception as e:
                logger.exception(f"Full‐page render failed on in page{page_index}: {e}")
            continue

        # otherwise per‐image
        for img_meta in images:
            xref = img_meta[0]
            try:
                imginfo = pdf.extract_image(xref)
                img = Image.open(io.BytesIO(imginfo["image"]))
            except Exception as e:
                logger.exception(f"Load image xref={xref} in page{page_index} failed: {e}")
                continue

            w, h = img.size
            if w<=5 or h<=5:
                logger.warning(f"Skipping tiny image xref={xref} ({w}×{h}) in page{page_index}")
                continue

            # OCR
            try:
                ocr_txt = pytesseract.image_to_string(img, lang='eng', config='--oem 1 --psm 3') # OEM 1 is LSTM engine
                if ocr_txt.strip():
                    docs.append(Document(
                        page_content=f"[Image OCR xref={xref}, in page{page_index}]\n{ocr_txt}",
                        metadata={"source":str(pdf_path),"page":page_index,"type":"image-ocr","xref":xref}
                    ))
                    logger.info(f"OCR image xref={xref} in page{page_index}")
                    # skip BLIP if heavy OCR
                    continue
            except Exception as e:
                logger.exception(f"OCR failed on image xref={xref} in page{page_index}: {e}")

            # BLIP caption
            try:
                cap = caption_image(img)
                docs.append(Document(
                    page_content=f"[Image caption xref={xref}, in page{page_index}]: {cap}",
                    metadata={"source":str(pdf_path),"page":page_index,"type":"image-caption","xref":xref}
                ))
                logger.info(f"Captioned image xref={xref} in page{page_index}: {cap}")
            except Exception as e:
                logger.exception(f"Caption failed on xref={xref} in page{page_index}: {e}")

    logger.info(f"── Finished {pdf_path.name}, docs so far: {len(docs)}")

# ── 4) Chunk splitting ────────────────────────────────────────────────────
logger.info("▶ Stage 3: Splitting into chunks")

# Use a sentence splitter to avoid breaking sentences
splitter = SentenceTransformersTokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

chunks = []
for i, doc in enumerate(tqdm(docs, desc="Splitting"), start=1):
    new_chunks = splitter.split_documents([doc])
    chunks.extend(new_chunks)
    logger.info(f"Doc {i}/{len(docs)} → +{len(new_chunks)} chunks")

logger.info(f"Total chunks: {len(chunks)}")

# ── 5) BM25 Implementation ────────────────────────────────────────────────
logger.info("▶ Stage 4: Implementing BM25")

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

# Create BM25 index
logger.info("Building BM25 index")
bm25 = BM25(k1=BM25_K1, b=BM25_B, nlp=nlp)  # Pass spaCy NLP object to BM25
corpus_texts = [chunk.page_content for chunk in chunks]
bm25.fit(corpus_texts)

# Save BM25 index
bm25_path = PERSIST_DIR / "bm25.pkl"
logger.info(f"Saving BM25 index to {bm25_path}")
with open(bm25_path, "wb") as f:
    pickle.dump(bm25, f)

# ── 6) Embedding & FAISS ─────────────────────────────────────────────────
logger.info("▶ Stage 5: Embedding & FAISS index build")
emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=EMBED_BASE)
dim = len(emb.embed_documents([""])[0])

# Create empty arrays to store all vectors
all_vectors = []

# Embed all chunks
for i, chunk in enumerate(tqdm(chunks, desc="Embedding"), start=1):
    try:
        vec = emb.embed_documents([chunk.page_content])[0]
        all_vectors.append(vec)
        if i%100==0 or i==len(chunks):
            logger.info(f"  → Embedded {i}/{len(chunks)}")
    except Exception as e:
        logger.exception(f"Embed failed chunk#{i}: {e}")
        # Add a zero vector as placeholder to maintain indices alignment
        all_vectors.append(np.zeros(dim, dtype=np.float32))

# Convert to numpy array and normalize
logger.info("Converting embeddings to normalized vectors")
vecs = np.array(all_vectors, dtype=np.float32)
vecs = normalize(vecs)  # L2 normalize

# Create and populate index
logger.info("Creating FAISS IndexFlatIP (inner product) for cosine similarity")
index = faiss.IndexFlatIP(dim)  # Use inner product for cosine similarity
index.add(vecs)

logger.info(f"Built FAISS with {index.ntotal} vectors")


# ── 7) Hybrid Search Helper ─────────────────────────────────────────────────
logger.info("▶ Stage 6: Creating hybrid search helper")

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
        """Perform hybrid search

        Args:
            query: Query text
            top_k: Number of results to return
            embeddings_model: Model to create embeddings

        Returns:
            List[Dict]: List of results with document and score
        """
        # BM25 search
        bm25_results = self.bm25.search(query, top_k=top_k*2)  # Get more results to improve recall

        # Vector search
        query_vec = self.embeddings_model.embed_documents([query])[0]
        D, I = self.faiss_index.search(np.array([query_vec], dtype="float32"), top_k*2)

        # Combine results - normalize scores
        all_scores = {}

        # Get max scores for normalization
        max_bm25 = max([score for _, score in bm25_results]) if bm25_results else 1.0
        max_vec = max(D[0]) if len(D) > 0 and len(D[0]) > 0 else 1.0

        # Add BM25 scores
        for doc_idx, score in bm25_results:
            if doc_idx not in all_scores:
                all_scores[doc_idx] = {"bm25": 0, "vector": 0}
            all_scores[doc_idx]["bm25"] = score / max_bm25

        # Add vector scores (invert since FAISS returns distances, not similarities)
        for i, doc_idx in enumerate(I[0]):
            if doc_idx not in all_scores:
                all_scores[doc_idx] = {"bm25": 0, "vector": 0}
            # Convert distance to similarity score (smaller distance = higher score)
            similarity = 1 - (D[0][i] / max_vec)
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
                                                 base_url=EMBED_BASE)
        self.faiss_index = faiss.read_index(str(PERSIST_DIR/"index.faiss"))

# Create and save hybrid searcher instance
hybrid_searcher = HybridSearcher(bm25, index, chunks, alpha=HYBRID_ALPHA, embeddings_model=emb)
hybrid_searcher_path = PERSIST_DIR / "hybrid_searcher.pkl"
logger.info(f"Saving hybrid searcher to {hybrid_searcher_path}")
with open(hybrid_searcher_path, "wb") as f:
    pickle.dump(hybrid_searcher, f)

# ── 8) Persist ────────────────────────────────────────────────────────────
logger.info("▶ Stage 7: Saving index + chunks")
faiss.write_index(index, str(PERSIST_DIR/"index.faiss"))
with open(PERSIST_DIR/"chunks.pkl","wb") as f:
    pickle.dump(chunks,f)
logger.info("✅ All done. Outputs in ./faiss_index/")

t1 = time.time()
logger.info(f"Total processing time: {t1 - t0:.2f} seconds")
