from flask import Flask, render_template, request
from docx import Document
import io, fitz, re, math, os
from collections import Counter, defaultdict
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# -------- PDF / DOCX readers ----------
ALLOWED_EXTENSIONS = {'pdf','docx'}
def allowed_file(fn): return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(fileobj: io.BytesIO) -> str:
    fileobj.seek(0)
    with fitz.open(stream=fileobj.getvalue(), filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def extract_text_from_docx(fileobj: io.BytesIO) -> str:
    fileobj.seek(0)
    d = Document(fileobj)
    return "\n".join(p.text for p in d.paragraphs)

# --------- Model (lazy) ----------
_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None:
        cache = os.environ.get("TRANSFORMERS_CACHE", "/opt/render/project/.cache")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache)
    return _MODEL

# --------- Similarity & JD coverage (your earlier logic) ----------
def _chunks(text: str, max_chars: int = 700) -> List[str]:
    text = re.sub(r'\r\n?', '\n', text or "").strip()
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    chunks, buf, n = [], [], 0
    for s in (p.strip() for p in parts if p and p.strip()):
        if n + len(s) + 1 > max_chars:
            if buf: chunks.append(" ".join(buf))
            buf, n = [s], len(s) + 1
        else:
            buf.append(s); n += len(s) + 1
    if buf: chunks.append(" ".join(buf))
    return chunks or [text[:max_chars]]

def analysis_resume(resume_text: str, jd_text: str) -> Dict:
    resume_text, jd_text = (resume_text or "").strip(), (jd_text or "").strip()
    if not resume_text or not jd_text:
        return {"overall_pct": 0.0, "top_matches": []}

    chunks = _chunks(resume_text, 700)
    m = get_model()
    jd_emb = m.encode(jd_text, convert_to_tensor=True)
    ch_emb = m.encode(chunks, convert_to_tensor=True)
    sims = util.cos_sim(ch_emb, jd_emb).squeeze(1)  # [N]

    max_sim = float(sims.max().item())
    avg_sim = float(sims.mean().item())

    def to_pct(x):
        lo, hi = 0.2, 0.8
        x = max(min(x, hi), lo)
        return round((x - lo) / (hi - lo) * 100.0, 1)

    k = min(5, len(chunks))
    idx = sims.topk(k).indices.tolist()
    top = [{"pct": round(float(sims[i].item())*100, 1), "text": chunks[i][:600]} for i in idx]

    return {"overall_pct": to_pct(0.6*max_sim + 0.4*avg_sim), "top_matches": top}

def _split_sentences(t: str) -> List[str]:
    t = (t or "").strip()
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', t)
    return [p.strip() for p in parts if p and p.strip()]

def _tokenize(t: str) -> List[str]:
    raw = re.findall(r"[A-Za-z0-9\+\#\.\-]+", t or "")
    out = []
    for x in raw:
        x = x.strip().lower().rstrip(".,:;")
        if x and (len(x) >= 2 or any(c.isdigit() for c in x)):
            out.append(x)
    return out

def _candidate_terms_from_jd(jd_text: str, max_terms=60) -> List[str]:
    sents = _split_sentences(jd_text)
    if not sents: return []
    S = len(sents)
    toks_sents = [_tokenize(s) for s in sents]
    tf = Counter(); df = defaultdict(int)
    for toks in toks_sents:
        seen = set()
        for n in range(1,5):
            if len(toks) < n: continue
            for i in range(len(toks)-n+1):
                ng = " ".join(toks[i:i+n])
                if n == 1 and len(ng) <= 3 and ng.isalpha(): continue
                tf[ng]+=1; seen.add(ng)
        for ng in seen: df[ng]+=1
    scored = []
    for ng,t in tf.items():
        d = df.get(ng,1)
        idf = math.log((S+1)/(d+1))
        score = t*idf
        if score <= 0: continue
        if " " not in ng:
            if ng.isalpha() and len(ng) <= 7 and (d/S) >= 0.35:
                continue
        scored.append((ng, score))
    connectors = {"and","or","to","of","in","on","for","by","as","with"}
    cleaned=[]
    for ng,score in scored:
        if " " in ng:
            w = ng.split()
            while w and w[0] in connectors: w.pop(0)
            while w and w[-1] in connectors: w.pop()
            if not w: continue
            ng = " ".join(w)
        cleaned.append((ng,score))
    cleaned.sort(key=lambda x:(x[1],len(x[0])), reverse=True)
    out,seen=[],set()
    for ng,_ in cleaned:
        if ng not in seen:
            out.append(ng); seen.add(ng)
        if len(out) >= max_terms: break
    return out

def jd_coverage_percent(resume_text: str, jd_text: str, max_terms=60) -> Dict:
    resume_lc = (resume_text or "").lower()
    terms = _candidate_terms_from_jd(jd_text, max_terms=max_terms)
    matched = [t for t in terms if t.lower() in resume_lc]
    missing = [t for t in terms if t.lower() not in resume_lc]
    total = len(terms) or 1
    pct = round(100.0 * len(matched) / total, 1)
    return {
        "coverage_pct": pct,
        "matched": matched,
        "missing": missing,
        "matched_count": len(matched),
        "total_terms": total,
        "max_terms": max_terms
    }

# ---------- Routes ----------
@app.route("/healthz")
def healthz():  # Render checks this; quick & cheap
    return "ok", 200

@app.route("/", methods=["GET","POST"])
def index():
    extracted_text = None
    analysis_results = None
    coverage_results = None
    warming_note = None

    if request.method == "POST":
        f = request.files.get("resume")
        jd = request.form.get("job_description", "")
        if f and allowed_file(f.filename):
            ext = f.filename.rsplit(".",1)[1].lower()
            data = io.BytesIO(f.read())
            if ext == "pdf": extracted_text = extract_text_from_pdf(data)
            elif ext == "docx": extracted_text = extract_text_from_docx(data)

        if extracted_text and jd:
            # warm model quickly to avoid first-request latency
            get_model()
            analysis_results = analysis_resume(extracted_text, jd)
            coverage_results = jd_coverage_percent(extracted_text, jd)
    else:
        warming_note = "Ready to analyze. Upload a resume and paste a JD."

    return render_template(
        "index.html",
        extracted_text=extracted_text,
        analysis_results=analysis_results,
        coverage_results=coverage_results,
        warming_note=warming_note
    )

if __name__ == "__main__":
    app.run(debug=True)
