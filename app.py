# app.py — Math Notes Summarizer (Upload + Paste, ChatGPT-like UI, no sidebar)
# ---------------------------------------------------------------------------
# - Left panel: Upload PDFs/TXTs and/or paste notes
# - Right panel: Clean, math-aware summary with Definitions/Theorems/Examples/Formulas
# - Robust PDF text extraction (PyMuPDF) with header/footer stripping
# - No external APIs. No questions hardcoded in code.
# Deps: streamlit, pymupdf, scikit-learn, numpy
# Run:  streamlit run app.py
# ---------------------------------------------------------------------------

import re
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- PDF extraction ----------------
try:
    import fitz  # PyMuPDF
    PYMUPDF_OK = True
except Exception:
    PYMUPDF_OK = False

# ---------------- UI (no sidebar, two-pane) ----------------
st.set_page_config(page_title="Study Notes AI — Math Summarizer", layout="wide")

st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu, header, footer {visibility:hidden;}
/* Page shell */
.block-container {max-width: 1240px; margin: 0 auto; padding-top: 12px;}
/* Two-pane layout */
.shell {display:grid; grid-template-columns: 1.05fr 1fr; gap: 18px;}
.card {
  border: 1px solid rgba(255,255,255,.12);
  background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));
  border-radius: 18px; padding: 16px; box-shadow: 0 10px 28px rgba(0,0,0,.12);
}
.hero {
  border: 1px solid rgba(255,255,255,.12);
  background:
    radial-gradient(1000px 520px at 20% -10%, rgba(110,130,255,.10), rgba(0,0,0,0)) ,
    linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.01));
  border-radius: 22px; padding: 24px 22px; margin: 0 0 14px 0;
}
.hero h1 {font-size: 1.9rem; font-weight: 800; margin: 0 0 4px 0;}
.hero p {margin: 2px 0 0 0; opacity:.9;}
.badge {display:inline-block; padding:6px 10px; border-radius:999px; border:1px solid rgba(255,255,255,.15);
        background: rgba(255,255,255,.04); font-size:.78rem; letter-spacing:.06em;}
/* Inputs */
.stTextArea textarea {border-radius: 14px !important;}
.stFileUploader label {font-weight: 700;}
/* Buttons */
.stButton>button {
  border-radius: 12px; padding: 10px 14px; border:1px solid rgba(255,255,255,.18);
  background: linear-gradient(180deg, rgba(90,120,255,.22), rgba(90,120,255,.08));
  font-weight: 800;
}
/* Output sections */
.section {margin: 8px 0 12px 0;}
.section h3 {margin: 6px 0 6px 0; font-weight: 800;}
.kcard {border:1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.03);
        border-radius:14px; padding:12px; margin:8px 0;}
ul { margin: 6px 0 0 1.1rem; }
li { margin: .26rem 0; line-height: 1.55; }
hr { border:none; height:1px; background:rgba(255,255,255,.14); margin:10px 0;}
.hint {opacity:.75; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="badge">Study Notes AI • Math-Aware Summarizer</div>
  <h1>Upload your math notes — get clean summaries with definitions, theorems & formulas.</h1>
  <p>No sidebars, no noise. Works with Discrete Math, Differential Equations, Calculus, Linear Algebra, etc.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="shell">', unsafe_allow_html=True)

# ========== LEFT PANEL (INPUTS) ==========
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Notes input")
uploaded = st.file_uploader("Upload PDF/TXT (you can select multiple files)", type=["pdf","txt"], accept_multiple_files=True)
pasted = st.text_area("Or paste notes here", height=220, placeholder="Paste lecture notes, handouts or copied slide text…")

# Options row
c1, c2, c3 = st.columns([1,1,1])
with c1:
    per_section = st.slider("Sentences / section", 2, 8, 4)
with c2:
    max_sections = st.slider("Max sections", 3, 15, 8)
with c3:
    max_formulas = st.slider("Max formulas", 5, 30, 18)

generate = st.button("⚡ Generate Summary")
st.markdown('<div class="hint">Tip: You can upload files and paste extra notes — both are merged automatically.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # end left card

# ========== RIGHT PANEL (OUTPUT) ==========
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Summary & Key Points")
out_placeholder = st.empty()
dl_placeholder = st.empty()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # end shell

# ---------------- Helpers ----------------
HEADER_PATTERNS = [
    r"^\s*course\s*:\s*", r"^\s*topic\s*:\s*", r"\bweek\s*\d+\b", r"\bmodule\s*\d+\b",
    r"\bmth\s*\d+\b", r"^page\s*\d+\s*$", r"^\s*lecture\s*\d+\s*$"
]
HEADER_REGEXES = [re.compile(p, re.I) for p in HEADER_PATTERNS]

def is_boilerplate_line(line: str) -> bool:
    ln = (line or "").strip()
    if not ln: return True
    for rgx in HEADER_REGEXES:
        if rgx.search(ln):
            return True
    if len(ln) <= 3 and ln.isupper():  # lonely artifacts
        return True
    return False

def clean_whitespace(t: str) -> str:
    # normalize bullets & whitespace (preserve newlines)
    t = t.replace("•", "\n• ").replace("◦", "\n• ")
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def extract_text_pymupdf(pdf_bytes: bytes, margin_frac: float = 0.16) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        H = page.rect.height
        top_cut = H * margin_frac
        bot_cut = H * (1 - margin_frac)
        words = page.get_text("words")  # x0,y0,x1,y1,word,block,line,word_no
        lines_map = {}
        for x0, y0, x1, y1, w, bno, lno, wno in words:
            if y0 < top_cut or y1 > bot_cut:  # drop header/footer
                continue
            lines_map.setdefault((bno, lno), []).append((x0, w))
        page_lines = []
        for _, items in sorted(lines_map.items(), key=lambda kv: kv[0]):
            items.sort(key=lambda t: t[0])
            line_text = " ".join(tok for _, tok in items).strip()
            if line_text and not is_boilerplate_line(line_text):
                page_lines.append(line_text)
        pages.append("\n".join(page_lines))
    doc.close()
    return clean_whitespace("\n\n".join(pages))

def read_uploaded(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        if not PYMUPDF_OK:
            st.error("PyMuPDF not installed. Run: pip install pymupdf")
            return ""
        return extract_text_pymupdf(file.read())
    # txt
    try:
        return clean_whitespace(file.read().decode("utf-8", "ignore"))
    except Exception:
        return ""

def split_sentences(text: str):
    # first respect line breaks (so headings don’t glue), then sentences
    parts = []
    for block in text.split("\n"):
        block = block.strip()
        if not block: continue
        parts.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", block))
    return [p.strip() for p in parts if p.strip()]

# ---- math-aware bits ----
HEAD_TAGS = [
    "definition", "theorem", "lemma", "corollary", "proposition",
    "example", "proof", "remark", "note", "algorithm", "axiom"
]
MATH_CHARS = r"[=±×÷∑∏∫∂∇∞≈≃≅≡≤≥→←↦αβγδϵεθλμνπρστυφχψωΩΔ∈∉⊂⊆⊇∪∩√^_]"
MATH_WORDS = r"(?:sin|cos|tan|cot|sec|csc|exp|ln|log|dx|dy|dz|dt|d/dx|d/dt|∂/∂x|lim|det|rank)"
MATH_LINE_RX = re.compile(rf"({MATH_CHARS}|{MATH_WORDS})", re.I)

def is_heading(line: str) -> bool:
    if len(line) <= 80 and line.isupper() and re.search(r"[A-Z]", line): return True
    if line.endswith(":") and len(line) <= 120: return True
    low = line.lower()
    return any(tag in low for tag in HEAD_TAGS) and len(line) <= 140

def detect_sections(raw: str):
    lines = [l.strip(" -•") for l in raw.split("\n") if l.strip()]
    sections, title, buf = [], "General", []
    for ln in lines:
        if is_boilerplate_line(ln): 
            continue
        if is_heading(ln):
            if buf:
                sections.append((title, "\n".join(buf).strip()))
                buf = []
            title = ln.rstrip(":").strip()
        else:
            buf.append(ln)
    if buf:
        sections.append((title, "\n".join(buf).strip()))
    return sections if sections else [("Notes", raw)]

def extract_blocks_by_tag(text: str, tag: str, cap: int):
    """
    Extract blocks starting with a tag word until next heading/tag/blank run.
    """
    rx = re.compile(rf"(^|\n)\s*({tag}\s*\d*\.?:?)\s*", re.I)
    heads = list(rx.finditer(text))
    out = []
    for i, m in enumerate(heads):
        start = m.end()
        end = heads[i+1].start() if i+1 < len(heads) else len(text)
        chunk = text[start:end].strip()
        header = m.group(2).strip().rstrip(":")
        if chunk and len(chunk) > 20:
            out.append((header, chunk))
        if len(out) >= cap:
            break
    return out

def extract_formulas(text: str, max_items: int = 20):
    """
    Pull math-looking lines. Streamlit renders LaTeX via st.latex.
    Keep unique top-N lines.
    """
    candidates = []
    for line in text.split("\n"):
        ln = line.strip()
        if len(ln) < 2: continue
        if MATH_LINE_RX.search(ln):
            candidates.append(ln)
    # de-duplicate by normalized form
    uniq, seen = [], set()
    for ln in candidates:
        key = re.sub(r"\s+", " ", ln).lower()
        if key in seen: continue
        seen.add(key)
        uniq.append(ln)
        if len(uniq) >= max_items: break
    return uniq

def tfidf_sentences(text: str, n: int) -> list:
    sents = split_sentences(text)
    if not sents: return []
    if len(sents) <= n: return sents
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sents)
    sal = X.sum(axis=1).A.ravel()
    idx = np.argsort(-sal)[:n]
    idx = sorted(idx.tolist())  # keep original order
    return [sents[i] for i in idx]

# ---------------- Combine corpus ----------------
if "corpus" not in st.session_state:
    st.session_state.corpus = ""

if uploaded:
    texts = []
    for f in uploaded:
        t = read_uploaded(f)
        if t: texts.append(t)
    if texts:
        st.session_state.corpus = "\n\n".join(texts)

if pasted and pasted.strip():
    st.session_state.corpus = (st.session_state.corpus + "\n\n" + pasted.strip()) if st.session_state.corpus else pasted.strip()

# ---------------- Generate ----------------
if generate:
    if not st.session_state.corpus:
        out_placeholder.info("Add notes first (upload or paste).")
    else:
        raw = st.session_state.corpus

        # Sections overview
        sections = detect_sections(raw)
        seen, cleaned = set(), []
        for title, content in sections:
            key = re.sub(r"\W+"," ", title.lower()).strip()
            if not content or len(content) < 40: continue
            if key in seen: continue
            seen.add(key)
            cleaned.append((title, content))
        cleaned = cleaned[:max_sections]

        # Tagged blocks
        defs   = extract_blocks_by_tag(raw, "Definition", cap=16)
        thms   = extract_blocks_by_tag(raw, "Theorem",    cap=12)
        lems   = extract_blocks_by_tag(raw, "Lemma",      cap=8)
        corols = extract_blocks_by_tag(raw, "Corollary",  cap=8)
        exs    = extract_blocks_by_tag(raw, "Example",    cap=12)

        # Formulas
        formulas = extract_formulas(raw, max_items=max_formulas)

        # Render right pane
        with out_placeholder.container():
            # Overview
            st.markdown('<div class="section"><h3>Overview</h3>', unsafe_allow_html=True)
            if not cleaned:
                st.markdown('<div class="kcard">No clear sections detected — showing top sentences.</div>', unsafe_allow_html=True)
                pts = tfidf_sentences(raw, per_section * 3)
                if pts:
                    st.markdown('<div class="kcard">', unsafe_allow_html=True)
                    st.markdown("\n".join([f"- {p}" for p in pts]))
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                for title, content in cleaned:
                    st.markdown(f'**{title}**')
                    points = tfidf_sentences(content, per_section)
                    if points:
                        st.markdown('<div class="kcard">', unsafe_allow_html=True)
                        st.markdown("\n".join([f"- {p}" for p in points]))
                        st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Definitions
            if defs:
                st.markdown('<div class="section"><h3>Definitions</h3>', unsafe_allow_html=True)
                st.markdown('<div class="kcard">', unsafe_allow_html=True)
                for h, b in defs:
                    st.markdown(f"- **{h}** — {b.splitlines()[0]}")
                st.markdown('</div></div>', unsafe_allow_html=True)

            # Results
            if thms or lems or corols:
                st.markdown('<div class="section"><h3>Results (Theorems, Lemmas, Corollaries)</h3>', unsafe_allow_html=True)
                st.markdown('<div class="kcard">', unsafe_allow_html=True)
                for coll in (thms, lems, corols):
                    for h, b in coll:
                        st.markdown(f"- **{h}** — {b.splitlines()[0]}")
                st.markdown('</div></div>', unsafe_allow_html=True)

            # Examples
            if exs:
                st.markdown('<div class="section"><h3>Examples (high-level)</h3>', unsafe_allow_html=True)
                st.markdown('<div class="kcard">', unsafe_allow_html=True)
                for h, b in exs:
                    st.markdown(f"- **{h}** — {b.splitlines()[0]}")
                st.markdown('</div></div>', unsafe_allow_html=True)

            # Key Formulas
            if formulas:
                st.markdown('<div class="section"><h3>Key Formulas</h3>', unsafe_allow_html=True)
                st.markdown('<div class="kcard">', unsafe_allow_html=True)
                for fx in formulas:
                    # try LaTeX rendering; if it fails, show as text
                    try:
                        st.latex(fx)
                    except Exception:
                        st.markdown(f"- {fx}")
                st.markdown('</div></div>', unsafe_allow_html=True)

        # Build Markdown export
        md = ["# Summary & Key Points\n"]
        if cleaned:
            md.append("## Overview\n")
            for title, content in cleaned:
                points = tfidf_sentences(content, per_section)
                if points:
                    md.append(f"### {title}\n" + "\n".join([f"- {p}" for p in points]) + "\n")
        if defs:
            md.append("## Definitions\n" + "\n".join([f"- **{h}** — {b.splitlines()[0]}" for h,b in defs]) + "\n")
        if thms or lems or corols:
            md.append("## Results\n")
            for coll in (thms, lems, corols):
                for h,b in coll:
                    md.append(f"- **{h}** — {b.splitlines()[0]}")
            md.append("\n")
        if formulas:
            md.append("## Key Formulas\n" + "\n".join([f"- {fx}" for fx in formulas]) + "\n")
        md_text = "\n".join(md)

        dl_placeholder.download_button(
            "⬇️ Download summary (.md)",
            data=md_text.encode("utf-8"),
            file_name="study-notes-summary.md",
            mime="text/markdown",
            type="secondary",
        )
