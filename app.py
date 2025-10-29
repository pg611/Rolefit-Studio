from flask import Flask, render_template, request
from docx import Document
from pypdf import PdfReader  # fast, pure-Python fallback
import io
import re
import math
from collections import Counter, defaultdict
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# --------- Lazy model loader (prevents OOM on small dynos) ----------
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')  # loads on first use
    return _model

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# -------------------- Helpers: file handling --------------------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_file: io.BytesIO) -> str:
    """
    Extract text from a PDF file passed as BytesIO (no disk writes).
    Try pypdf first (lightweight), then fallback to PyMuPDF (better layout).
    """
    # 1) pypdf (lightweight)
    try:
        pdf_file.seek(0)
        reader = PdfReader(pdf_file)
        txt = "".join(page.extract_text() or "" for page in reader.pages)
        if txt.strip():
            return txt
    except Exception:
        pass

    # 2) fallback to PyMuPDF (lazy import keeps memory lower at boot)
    try:
        import fitz  # PyMuPDF
        pdf_file.seek(0)
        with fitz.open(stream=pdf_file.getvalue(), filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

def extract_text_from_docx(docx_file: io.BytesIO) -> str:
    """
    Extract text from a DOCX file passed as BytesIO (no disk writes).
    """
    text = ""
    try:
        docx_file.seek(0)
        document = Document(docx_file)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        text = f"Error extracting text from DOCX: {e}"
    return text

# -------------------- Full-JD vs Full-Resume matcher --------------------
def _chunks(text: str, max_chars: int = 700) -> List[str]:
    """
    Lightweight sentence/paragraph packer to keep chunks ~max_chars for embedding.
    """
    text = re.sub(r'\r\n?', '\n', text or "").strip()
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    chunks, buf, n = [], [], 0
    for s in (p.strip() for p in parts if p and p.strip()):
        if n + len(s) + 1 > max_chars:
            if buf:
                chunks.append(" ".join(buf))
            buf, n = [s], len(s) + 1
        else:
            buf.append(s)
            n += len(s) + 1
    if buf:
        chunks.append(" ".join(buf))
    return chunks or [text[:max_chars]]

def analysis_resume(resume_text: str, job_description: str) -> Dict:
    """
    Compare the entire resume to the entire JD using semantic similarity.
    Returns an overall 0–100 score and top matching resume snippets.
    """
    resume_text = (resume_text or "").strip()
    job_description = (job_description or "").strip()
    if not resume_text or not job_description:
        return {
            "overall_score": 0.0,
            "top_matches": [],
            "raw": {"max_sim": 0.0, "avg_sim": 0.0}
        }

    resume_chunks = _chunks(resume_text, max_chars=700)
    model = get_model()  # <— lazy-load right here

    jd_emb = model.encode(job_description, convert_to_tensor=True)
    chunk_embs = model.encode(resume_chunks, convert_to_tensor=True)

    sims = util.cos_sim(chunk_embs, jd_emb).squeeze(1)  # [num_chunks]
    max_sim = float(sims.max().item())
    avg_sim = float(sims.mean().item())

    # Top-k evidence
    k = min(5, len(resume_chunks))
    top_idx = sims.topk(k).indices.tolist()
    top_matches = [
        {"score": float(sims[i].item()), "text": resume_chunks[i][:600]}
        for i in top_idx
    ]

    # Simple 0–100 scaling
    def _to_pct(x: float) -> float:
        lo, hi = 0.2, 0.8  # clamp typical cosine range to stabilize
        x = max(min(x, hi), lo)
        return round((x - lo) / (hi - lo) * 100.0, 1)

    return {
        "overall_score": _to_pct(0.6 * max_sim + 0.4 * avg_sim),
        "top_matches": top_matches,
        "raw": {"max_sim": round(max_sim, 4), "avg_sim": round(avg_sim, 4)}
    }

# -------------------- Stateless JD keyword mining --------------------
def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    return [p.strip() for p in parts if p and p.strip()]

def _tokenize(text: str) -> List[str]:
    # keep letters/digits/+/#/.- ; lowercase; strip trailing punctuation
    raw = re.findall(r"[A-Za-z0-9\+\#\.\-]+", text or "")
    toks = []
    for t in raw:
        t = t.strip().lower().rstrip(".:,;")
        if not t:
            continue
        if len(t) >= 2 or any(c.isdigit() for c in t):  # keep s3, .net, ec2
            toks.append(t)
    return toks

def _candidate_terms_from_jd(jd_text: str, max_terms: int = 60) -> List[str]:
    """
    Purely statistical keyphrase mining from the JD.
    - n-grams (1..4) within sentences
    - TF * log(S / DF) over sentences
    - filters short/filler unigrams and cleans n-gram edges
    """
    sents = _split_sentences(jd_text)
    if not sents:
        return []

    S = len(sents)
    sent_tokens = [_tokenize(s) for s in sents]

    tf = Counter()
    df = defaultdict(int)

    for toks in sent_tokens:
        seen_ngrams = set()
        for n in range(1, 5):
            if len(toks) < n:
                continue
            for i in range(len(toks) - n + 1):
                ng = " ".join(toks[i:i+n])

                # drop ultra-short unigrams early (common fillers)
                if n == 1 and len(ng) <= 3 and ng.isalpha():
                    continue

                tf[ng] += 1
                seen_ngrams.add(ng)
        for ng in seen_ngrams:
            df[ng] += 1

    # score
    scored = []
    for ng, t in tf.items():
        d = df.get(ng, 1)
        idf = math.log((S + 1) / (d + 1))
        score = t * idf
        if score <= 0:
            continue

        # drop ubiquitous short unigrams (dynamic)
        if " " not in ng:
            df_ratio = d / S
            if ng.isalpha() and len(ng) <= 7 and df_ratio >= 0.35:
                continue

        scored.append((ng, score))

    # clean n-gram edges of tiny connectors
    CONNECTORS = {"and", "or", "to", "of", "in", "on", "for", "by", "as", "with"}
    cleaned = []
    for ng, score in scored:
        if " " in ng:
            words = ng.split()
            while words and words[0] in CONNECTORS:
                words.pop(0)
            while words and words[-1] in CONNECTORS:
                words.pop()
            if not words:
                continue
            ng = " ".join(words)
        cleaned.append((ng, score))

    # rank and dedupe
    cleaned.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    out, seen = [], set()
    for ng, _ in cleaned:
        if ng and ng not in seen:
            out.append(ng)
            seen.add(ng)
        if len(out) >= max_terms:
            break
    return out

def jd_coverage_percent(resume_text: str, jd_text: str, max_terms: int = 60) -> Dict:
    """
    JD-first coverage:
      - Extract top JD terms (n-grams)
      - Check exact (case-insensitive) presence in the resume
      - Return: % coverage, matched list, missing list, totals
    """
    resume_text = (resume_text or "").lower()
    jd_text = (jd_text or "")
    terms = _candidate_terms_from_jd(jd_text, max_terms=max_terms)

    matched, missing = [], []
    for t in terms:
        if t.lower() in resume_text:
            matched.append(t)
        else:
            missing.append(t)

    total = len(terms) or 1
    coverage = round(100.0 * len(matched) / total, 1)

    return {
        "coverage_pct": coverage,
        "matched": matched,
        "missing": missing,
        "total_terms": total,
        "matched_count": len(matched),
        "max_terms": max_terms
    }

# -------------------- Route --------------------
@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    extracted_text = None
    analysis_results = None
    coverage_results = None

    if request.method == 'POST':
        resume_file = request.files.get('resume')
        job_description = request.form.get('job_description', '')

        if resume_file and allowed_file(resume_file.filename):
            ext = resume_file.filename.rsplit('.', 1)[1].lower()
            file_stream = io.BytesIO(resume_file.read())
            if ext == 'pdf':
                extracted_text = extract_text_from_pdf(file_stream)
            elif ext == 'docx':
                extracted_text = extract_text_from_docx(file_stream)

            if extracted_text and job_description:
                analysis_results = analysis_resume(extracted_text, job_description)
                coverage_results = jd_coverage_percent(extracted_text, job_description)

    return render_template(
        'index.html',
        extracted_text=extracted_text,
        analysis_results=analysis_results,
        coverage_results=coverage_results
    )

if __name__ == '__main__':
    app.run(debug=True)
