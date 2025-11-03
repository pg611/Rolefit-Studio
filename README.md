# Rolefit-Studio
![Rolefit-Studio](https://github.com/pg611/Rolefit-Studio/blob/main/rolefit-studio-banner.png)

# 🤖 RoleFit-Studio — Resume ↔ JD Analyzer (Flask + Sentence-BERT)

![RoleFit Studio Banner](https://raw.githubusercontent.com/pg611/Rolefit-Studio/main/rolefit-studio-banner.png)

AI-powered web application that analyzes how well your **resume** matches a **job description**, helping you optimize for **ATS (Applicant Tracking Systems)** and **semantic relevance**.

🔗 **Live App:** [https://rolefit-studio.onrender.com](https://rolefit-studio.onrender.com)

---

## 🎥 Project Demo
<video controls width="720">
  <source src="https://raw.githubusercontent.com/pg611/Rolefit-Studio/main/Rolefit%20Studio%20LinkedIn.mp4" type="video/mp4">
  Your browser doesn’t support HTML5 video.
  <a href="https://github.com/pg611/Rolefit-Studio/blob/main/Rolefit%20Studio%20LinkedIn.mp4">Watch the demo here.</a>
</video>

[![🎬 Watch Full Demo](https://img.shields.io/badge/🎥-Watch%20Full%20Demo-blue)](https://github.com/pg611/Rolefit-Studio/blob/main/Rolefit%20Studio%20LinkedIn.mp4)

---

## 🧠 About RoleFit-Studio

**RoleFit-Studio** is an AI-driven Flask application that helps job seekers instantly assess how well their resumes align with specific job descriptions.  
It combines **semantic similarity** (using Sentence-BERT embeddings) and **keyword coverage** (using TF–IDF-style term extraction) to give actionable feedback and a transparent “RoleFit Score.”

---

### 🎯 Objective

Most ATS tools reject resumes based on missing or mismatched keywords.  
Even qualified candidates can get filtered out due to wording differences.

**RoleFit-Studio** helps bridge this gap by:
- Measuring **semantic alignment** between your resume and JD.  
- Highlighting **key terms missing** from your resume.  
- Showing **the strongest-matching sections** of your resume.  

---

### 🧩 Key Features

| Feature | Description |
|----------|-------------|
| **🔍 Resume–JD Matching** | Upload a `.pdf` or `.docx` resume and paste a job description to get an **Overall RoleFit Score (0–100)**. |
| **🧠 Semantic Similarity** | Uses **SentenceTransformer (all-MiniLM-L6-v2)** for contextual matching. |
| **✨ Highlighted Evidence** | Displays the top 5 best-matching resume snippets that contribute most to the score. |
| **📋 JD Coverage Analysis** | Extracts and compares key n-grams (1–4 words) between JD and resume to find what’s **matched** and **missing**. |
| **💾 File Support** | Handles `.pdf` and `.docx` formats using in-memory extraction — no disk writes. |
| **🎨 Clean UI** | Tabbed interface with modern dark theme: Highlights • Missing Terms • Matched Terms • Extracted Resume. |

---

### 🧠 How It Works

1. **Upload Resume + JD** → User uploads a resume file and pastes the full job description.  
2. **Text Extraction** →  
   - PDFs via `fitz` (PyMuPDF)  
   - DOCX via `python-docx`  
3. **Semantic Comparison** →  
   - Splits resume into 700-character chunks.  
   - Computes embeddings for both resume and JD.  
   - Calculates cosine similarity (`util.cos_sim`).  
   - Aggregates to an Overall RoleFit Score (0–100).  
4. **Keyword Coverage** →  
   - Extracts statistically weighted JD terms using TF × log((S+1)/(DF+1)).  
   - Reports % coverage, matched vs. missing.  
5. **Display Results** →  
   - Interactive tabs show Highlights, Missing, Matched, and Raw Extracted Resume.

---

### 🧰 Tech Stack

| Layer | Tools |
|-------|-------|
| **Frontend** | HTML5, CSS3 (custom dark UI) |
| **Backend** | Flask |
| **NLP / ML** | Sentence-BERT (`all-MiniLM-L6-v2`), PyTorch |
| **Parsing** | PyMuPDF, python-docx |
| **Deployment** | Render.com (Gunicorn WSGI server) |


