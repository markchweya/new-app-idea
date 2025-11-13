# app.py
# -----------------------------------------------------------
# NOTES SUMMARIZER — upload PDF/TXT → click Summarize
# - Robust PDF extraction (PyMuPDF), drops headers/footers
# - Structured, topic-wise extractive summary (TF-IDF)
# - Clean Markdown output
# Deps: streamlit, pymupdf, scikit-learn
# Run:  streamlit run app.py
# -----------------------------------------------------------

import re
import io
import sys
import numpy as np
import streamlit as st

# Prefer PyMuPDF for strong PDF extraction
try:
    import fitz  # PyMuPDF
    PYMUPDF_OK = True
except Exception:
    PYMUPDF_OK = False

# Lightweight extractive scoring
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------- Utilities ----------------------

def clean_text(t: str) -> str:
    """Collapse whitespace and trim."""
    t = re.sub(r"\s+", " ", t or "")
    return t.strip()

def split_sentences(text: str):
    """Simple sentence splitter without extra deps."""
    # keep periods inside decimals/initials intact reasonably
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text.strip())
    return [s.strip() for s in sents if s and len(s.strip()) > 0]

HEADER_PATTERNS = [
    r"^\s*course\s*:\s*", r"^\s*topic\s*:\s*", r"\bweek\s*\d+\b",
    r"\bmth\s*\d+\b", r"^page\s*\d+\s*$"
]
HEADER_REGEXES = [re.compile(p, re.I) for p in HEADER_PATTERNS]

def is_boilerplate_line(line: str) -> bool:
    """Filter common lecture headers/footers."""
    ln = (line or "").strip()
    if not ln:
        return True
    for rgx in HEADER_REGEXES:
        if rgx.search(ln):
            return True
    # very short all-caps often noise
    if len(ln) <= 5 and ln.isupper():
        return True
    return False

def extract_text_pymupdf(pdf_bytes: bytes, margin_frac: float = 0.14) -> str:
    """
    Read PDF with PyMuPDF, drop top/bottom margins (headers/footers),
    rebuild lines left→right, filter boilerplate, return joined text.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_pages = []
    for page in doc:
        H = page.rect.height
        top_cut = H * margin_frac
        bot_cut = H * (1 - margin_frac)

        words = page.get_text("words")  # (x0,y0,x1,y1,word, block, line, word_no)
        lines_map = {}
        for x0, y0, x1, y1, w, bno, lno, wno in words:
            if y0 < top_cut or y1 > bot_cut:
                continue
            lines_map.setdefault((bno, lno), []).append((x0, w))

        page_lines = []
        for _, items in sorted(lines_map.items(), key=lambda kv: kv[0]):
            items.sort(key=lambda t: t[0])
            line_text = " ".join(tok for _, tok in items).strip()
            if line_text and not is_boilerplate_line(line_text):
                page_lines.append(line_text)

        all_pages.append(" ".join(page_lines))

    doc.close()
    return clean_text(" ".join(all_pages))

def read_any(file) -> str:
    """Read PDF (preferred) or TXT."""
    name = file.name.lower()
    if name.endswith(".pdf"):
        if not PYMUPDF_OK:
            st.error("PyMuPDF not installed. Run: pip install pymupdf")
            return ""
        data = file.read()
        return extract_text_pymupdf(data)
    else:
        try:
            return clean_text(file.read().decode("utf-8", "ignore"))
        except Exception:
            return ""

# ---------------------- Sectioning & Summarization ----------------------

SECTION_TITLE_HINTS = [
    "differential", "equation", "separable", "homogeneous", "non-homogeneous",
    "exact", "laplace", "fourier", "power series", "series solution",
    "initial value", "boundary", "variation of parameters", "equi-dimensional",
    "systems", "first order", "second order", "operator", "examples", "exercise",
]

def detect_sections(raw_text: str):
    """
    Build sections from lines: a new section starts at probable headings:
    - ALL CAPS short lines
    - Lines ending with ':' (e.g., 'Definitions and Basic Concepts:')
    - Lines containing key hints (Laplace, Power Series, etc.) with title-ish shape
    Always keep content; return list of (title, content).
    """
    # Work in lines first (before we collapsed whitespace too much)
    # Recreate rough lines from the text by forcing breaks on common delimiters
    rough_lines = re.split(r"(?:\n|\r|\s{2,})", raw_text)
    lines = [l.strip() for l in rough_lines if l and len(l.strip()) > 0]

    sections = []
    current_title = "General"
    current_buf = []

    def is_heading(line: str) -> bool:
        if len(line) <= 80 and line.isupper() and re.search(r"[A-Z]", line):
            return True
        if line.endswith(":") and len(line) <= 120:
            return True
        lower = line.lower()
        for hint in SECTION_TITLE_HINTS:
            if hint in lower and len(line) <= 140:
                # avoid obvious boilerplate
                if not is_boilerplate_line(line):
                    return True
        return False

    for ln in lines:
        if is_boilerplate_line(ln):
            continue
        if is_heading(ln):
            # push previous
            if current_buf:
                sections.append((current_title, " ".join(current_buf).strip()))
                current_buf = []
            # new title (strip trailing colon)
            current_title = ln.rstrip(":").strip()
        else:
            current_buf.append(ln)

    # flush last
    if current_buf:
        sections.append((current_title, " ".join(current_buf).strip()))

    # If nothing detected, return one general section
    if not sections:
        return [("Notes", raw_text)]
    return sections

def summarize_sentences(text: str, n_sentences: int = 3) -> str:
    sents = split_sentences(text)
    if not sents:
        return ""
    if len(sents) <= n_sentences:
        return " ".join(sents)

    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sents)  # (num_sents, vocab)
    salience = X.sum(axis=1).A.ravel()
    top_idx = np.argsort(-salience)[:n_sentences]
    # keep original order
    top_idx = sorted(top_idx.tolist())
    return " ".join(sents[i] for i in top_idx)

def structured_summary(text: str, max_sections: int = 8, n_sentences_per_section: int = 3) -> str:
    """
    Create clean Markdown summary by sections.
    """
    sections = detect_sections(text)
    # Light dedup: drop empty or near-identical titles
    cleaned = []
    seen_titles = set()
    for title, content in sections:
        tkey = re.sub(r"\W+", " ", title.lower()).strip()
        if not content or len(content) < 40:
            continue
        if tkey in seen_titles:
            continue
        seen_titles.add(tkey)
        cleaned.append((title, content))

    if not cleaned:
        # fallback: global summary
        return "## Summary\n\n" + summarize_sentences(text, n_sentences_per_section * 2)

    cleaned = cleaned[:max_sections]
    out = ["## Summary of Notes\n"]
    for title, content in cleaned:
        summ = summarize_sentences(content, n_sentences_per_section)
        if not summ:
            continue
        out.append(f"### {title}\n- {summ}\n")
    return "\n".join(out).strip()

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="Notes Summarizer", layout="centered")
st.title("📝 Notes Summarizer")
st.caption("Upload PDF/TXT → click Summarize. Clean, structured output from your notes only.")

# Controls
colA, colB = st.columns(2)
with colA:
    n_per = st.slider("Sentences per section", 1, 6, 3)
with colB:
    max_sec = st.slider("Max sections", 3, 15, 8)

# File upload
if "corpus_text" not in st.session_state:
    st.session_state.corpus_text = ""

files = st.file_uploader(
    "Upload your notes (PDF/TXT). You can select multiple files.",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if files:
    texts = []
    for f in files:
        t = read_any(f)
        if t:
            texts.append(t)
    st.session_state.corpus_text = clean_text(" ".join(texts))
    if st.session_state.corpus_text:
        st.success(f"Loaded {len(files)} file(s).")

st.markdown("---")

if st.button("Summarize"):
    if not st.session_state.corpus_text:
        st.warning("Upload at least one file first.")
    else:
        summary_md = structured_summary(
            st.session_state.corpus_text,
            max_sections=max_sec,
            n_sentences_per_section=n_per
        )
        if not summary_md or len(summary_md.strip()) == 0:
            st.info("Couldn’t extract readable text. If your PDF is fully scanned (images), enable OCR in a future version.")
        else:
            st.markdown(summary_md)
            with st.expander("Show extracted raw text (debug)"):
                st.text(st.session_state.corpus_text[:20000])  # preview
