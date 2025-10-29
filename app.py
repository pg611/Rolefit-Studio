from flask import Flask, render_template, request
from docx import Document
import io, re, math, threading
from collections import Counter, defaultdict
from typing import List, Dict

# ---------------- Flask ----------------
app = Flask(__name__)

# ---------- lazy model load + warmup (prevents 502 on Render free) ----------
_model = None
_model_ready = False

def get_model():
    """Load the sentence-transformers model the first time we need it."""
    global _model
    if _model is None:
        # Import here so module import stays light
        from sentence_transformers import SentenceTransformer
        # Small, fast model; CPU only
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    return _model

def _warmup():
    """Background warmup so the first user POST doesn't time out."""
    global _model_ready
    try:
        m = get_model()
        _ = m.encode(["warmup"], show_progress_bar=False)
        _model_ready = True
    except Exception as e:
        print("Warmup failed:", e)

# Fire once at boot; does NOT block startup
threading.Thread(target=_warmup, daemon=True).start()

# ---------------- Allowed file types ----------------
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- File readers ----------------
def extract_text_from_pdf(pdf_file: io.BytesIO) -> str:
    """
    Extract text from a PDF file passed as BytesIO (no disk writes).
    """
    import fitz  # PyMuPDF (import here to keep startup light)
    text = ""
    try:
        pdf_file.seek(0)
        with fitz.open(stream=pdf_file.getvalue(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        text = f"Error extracting text from PDF: {e}"
    return text


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
        print(f"Error reading DOCX: {e}")
        text = f"Error extracting text from DOCX: {e}"
    return text

# ---------------- Resume ↔ JD similarity ----------------
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
    Returns an overall percent score and top matching resume snippets.
    """
    from sentence_transformers import util  # light import
    m = get_model()

    resume_text = (resume_text or "").strip()
    jd_text = (job_description or "").strip()
    if not resume_text or not jd_text:
        return {"overall_pct": 0.0, "top_matches": [], "raw": {"max_sim": 0.0, "avg_sim": 0.0}}

    resume_chunks = _chunks(resume_text, max_chars=700)
    jd_emb = m.encode(jd_text, convert_to_tensor=True, show_progress_bar=False)
    chunk_embs = m.encode(resume_chunks, convert_to_tensor=True, show_progress_bar=False)

    sims = util.cos_sim(chunk_embs, jd_emb).squeeze(1)  # [num_chunks]
    max_sim = float(sims.max().item())
    avg_sim = float(sims.mean().item())

    # Top-k evidence
    k = min(5, len(resume_chunks))
    top_idx = sims.topk(k).indices.tolist()
    top_matches = [{"pct": round(100 * float(sims[i].item()), 1), "text": resume_chunks[i][:800]} for i in top_idx]

    # Scale to a friendlier 0–100; clamp typical range
    def _scale(x: float) -> float:
        lo, hi = 0.2, 0.8
        x = max(min(x, hi), lo)
        return round((x - lo) / (hi - lo) * 100.0, 1)

    overall_pct = _scale(0.6 * max_sim + 0.4 * avg_sim)
    return {"overall_pct": overall_pct, "top_matches": top_matches, "raw": {"max_sim": max_sim, "avg_sim": avg_sim}}

# ---------------- JD term mining (stateless; no stored lists) ----------------
def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    return [p.strip() for p in parts if p and p.strip()]

def _tokenize(text: str) -> List[str]:
    # keep letters/digits/+/#/.- ; lowercase; strip edge punctuation
    raw = re.findall(r"[A-Za-z0-9\+\#\.\-]+", text or "")
    toks = []
    for t in raw:
        t = t.strip().lower().rstrip(".:,;")
        if not t:
            continue
        # keep tokens len>=2 OR containing a digit (keeps s3, .net, ec2)
        if len(t) >= 2 or any(c.isdigit() for c in t):
            toks.append(t)
    return toks

def _candidate_terms_from_jd(jd_text: str, max_terms: int = 60) -> List[str]:
    """
    Stateless keyphrase mining from the JD:
      - n-grams (1..4) within sentences
      - TF * log(S / DF) over sentences
      - drop ubiquitous short unigrams (so words like 'provide', 'including' disappear)
      - trim connector words at edges
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
                # drop tiny unigrams right away
                if n == 1 and len(ng) <= 3 and ng.isalpha():
                    continue
                tf[ng] += 1
                seen_ngrams.add(ng)
        for ng in seen_ngrams:
            df[ng] += 1

    scored = []
    for ng, t in tf.items():
        d = df.get(ng, 1)
        idf = math.log((S + 1) / (d + 1))
        score = t * idf
        if score <= 0:
            continue

        # drop ubiquitous short unigrams dynamically (e.g., provide, including)
        if " " not in ng:
            df_ratio = d / S
            if ng.isalpha() and len(ng) <= 7 and df_ratio >= 0.35:
                continue

        scored.append((ng, score))

    # trim edges of connector words
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
    JD-first coverage (exact contains, case-insensitive).
    Returns: coverage %, matched list, missing list.
    """
    resume_text = (resume_text or "").lower()
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

# ---------------- Routes ----------------
@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    warming_note = None
    extracted_text = None
    analysis_results = None
    coverage_results = None

    if request.method == 'POST':
        # If model still warming, return fast to avoid proxy timeout/502
        if not _model_ready:
            warming_note = "Warming the NLP model… please submit again in ~10–20 seconds."
        else:
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
        warming_note=warming_note,
        extracted_text=extracted_text,
        analysis_results=analysis_results,
        coverage_results=coverage_results
    )

if __name__ == '__main__':
    app.run(debug=True)
