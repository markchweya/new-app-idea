# app.py — Kuki Docs Assistant (Ultra-Minimal Document Summarizer)
# Run: streamlit run app.py

import re
from datetime import datetime

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional PDF support
try:
    import fitz  # PyMuPDF
    PYMUPDF_OK = True
except Exception:
    PYMUPDF_OK = False

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title="Kuki Docs Assistant", layout="wide")

# -------------------------------------------------------------------
# GLOBAL CSS — ultra minimal, single centered component
# -------------------------------------------------------------------
st.markdown(
    """
<style>
/* Hide default chrome */
#MainMenu, header, footer {visibility: hidden;}

/* Background */
html, body {
    background:
      radial-gradient(circle at 0% 0%, #1d244a 0, #050516 40%, #02030a 100%);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
}

/* Center everything */
.block-container {
    max-width: 760px;
    margin: 0 auto;
    padding-top: 10vh;
    padding-bottom: 8vh;
}

/* Outer shell to vertically center */
.app-shell {
    min-height: 65vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Single glass card */
.app-card {
    width: 100%;
    border-radius: 26px;
    padding: 24px 26px 22px 26px;
    background: rgba(15,23,42,0.92);
    border: 1px solid rgba(148,163,184,0.55);
    box-shadow: 0 26px 60px rgba(0,0,0,0.95);
    backdrop-filter: blur(22px);
}

/* Title + subtitle */
.app-title {
    text-align: center;
    margin-bottom: 18px;
}
.app-title h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(135deg, #f97316, #ec4899, #6366f1, #22c1c3);
    -webkit-background-clip: text;
    color: transparent;
}
.app-title p {
    margin-top: 4px;
    font-size: 0.95rem;
    opacity: 0.78;
}

/* Prompt / upload bar wrapper */
.prompt-wrapper {
    margin-top: 6px;
}

/* File uploader styled as big bar */
.stFileUploader div[data-testid="stFileDropzone"] {
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.7);
    background: radial-gradient(circle at 0% 0%, rgba(129,140,248,0.28), rgba(15,23,42,0.98));
    padding: 10px 14px;
}
.stFileUploader:hover div[data-testid="stFileDropzone"] {
    border-color: rgba(129,140,248,1);
}

/* Main CTA button */
.stButton>button {
    margin-top: 10px;
    width: 100%;
    border-radius: 999px;
    padding: 10px 16px;
    border: none;
    background: linear-gradient(135deg, #ec4899, #6366f1, #22c1c3);
    color: #f9fafb;
    font-weight: 800;
    font-size: 0.92rem;
    letter-spacing: .04em;
    box-shadow: 0 18px 40px rgba(0,0,0,0.9);
    transition: transform .12s ease, filter .1s ease, box-shadow .12s ease;
}
.stButton>button:hover {
    transform: translateY(-1px);
    filter: brightness(1.08);
    box-shadow: 0 22px 55px rgba(0,0,0,1);
}

/* Hint text */
.hint {
    font-size: 0.8rem;
    opacity: 0.6;
    margin-top: 5px;
    text-align: center;
}

/* Summary card */
.summary-card {
    margin-top: 24px;
    border-radius: 22px;
    padding: 16px 18px 16px 18px;
    background: radial-gradient(circle at 0% 0%, rgba(129,140,248,0.25), transparent 55%),
                rgba(15,23,42,0.96);
    border: 1px solid rgba(129,140,248,0.82);
    box-shadow: 0 24px 55px rgba(0,0,0,1);
    animation: fadeIn .27s ease-out;
}
.summary-card h3 {
    margin-top: 0;
}

/* Sections inside summary */
.section-title {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: .16em;
    opacity: 0.75;
    margin: 6px 0 2px 0;
}
.kcard {
    border-radius: 15px;
    padding: 9px 11px;
    background: rgba(15,23,42,0.94);
    border: 1px solid rgba(148,163,184,0.6);
    margin-bottom: 6px;
}
ul {
    margin: 4px 0 0 1.15rem;
    padding-left: 0;
}
li {
    margin: 0.16rem 0;
    line-height: 1.5;
}

/* Follow-up input bar */
.followup-wrapper {
    margin-top: 18px;
}
.followup-label {
    font-size: 0.86rem;
    opacity: 0.75;
    margin-bottom: 4px;
}
.followup-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 8px;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.85);
    background: radial-gradient(circle at 0% 0%, rgba(129,140,248,0.22), rgba(15,23,42,0.98));
}
.followup-input textarea {
    border-radius: 999px !important;
    padding: 6px 12px !important;
    background: rgba(15,23,42,0.98) !important;
    font-size: 0.9rem !important;
    border: 1px solid transparent !important;
}
.followup-input textarea:focus {
    border-color: rgba(129,140,248,0.95) !important;
    box-shadow: 0 0 0 1px rgba(129,140,248,0.7);
}
.followup-send {
    width: 32px;
    height: 32px;
    border-radius: 999px;
    background: linear-gradient(135deg, #ec4899, #6366f1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.86rem;
    box-shadow: 0 10px 22px rgba(0,0,0,0.95);
}

/* Follow-up answer bubble */
.followup-answer {
    margin-top: 10px;
    border-radius: 18px;
    padding: 9px 11px;
    background: rgba(15,23,42,0.96);
    border: 1px solid rgba(148,163,184,0.7);
    animation: fadeIn .25s ease-out;
    font-size: 0.9rem;
}

/* Animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(8px);}
    to   {opacity: 1; transform: translateY(0);}
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# STATE
# -------------------------------------------------------------------
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "has_summary" not in st.session_state:
    st.session_state.has_summary = False
if "followup_answer" not in st.session_state:
    st.session_state.followup_answer = ""
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# Static summarization knobs (no sliders in UI)
PER_SECTION = 4
MAX_SECTIONS = 8
MAX_FORMULAS = 20

# -------------------------------------------------------------------
# HELPER FUNCTIONS (same core summariser logic)
# -------------------------------------------------------------------
HEADER_PATTERNS = [
    r"^\s*course\s*:\s*",
    r"^\s*topic\s*:\s*",
    r"\bweek\s*\d+\b",
    r"\bmodule\s*\d+\b",
    r"\bmth\s*\d+\b",
    r"^page\s*\d+\s*$",
    r"^\s*lecture\s*\d+\s*$",
]
HEADER_REGEXES = [re.compile(p, re.I) for p in HEADER_PATTERNS]

MATH_CHARS = r"[=±×÷∑∏∫∂∇∞≈≃≅≡≤≥→←↦αβγδϵεθλμνπρστυφχψωΩΔ∈⊂⊆⊇∪∩√^_]"
MATH_WORDS = r"(?:sin|cos|tan|cot|sec|csc|exp|ln|log|dx|dy|dz|dt|d/dx|d/dt|∂/∂x|lim|det|rank)"
MATH_LINE_RX = re.compile(rf"({MATH_CHARS}|{MATH_WORDS})", re.I)

HEAD_TAGS = [
    "definition",
    "theorem",
    "lemma",
    "corollary",
    "proposition",
    "example",
    "proof",
    "remark",
    "note",
    "algorithm",
    "axiom",
]


def is_boilerplate_line(line: str) -> bool:
    ln = (line or "").strip()
    if not ln:
        return True
    if ln in {"•", "-", "–", "·"}:
        return True
    for rgx in HEADER_REGEXES:
        if rgx.search(ln):
            return True
    if len(ln) <= 3 and ln.isupper():
        return True
    return False


def clean_whitespace(t: str) -> str:
    t = t.replace("•", "\n• ").replace("◦", "\n• ")
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"(?m)^\s*[•·-]\s*$", "", t)
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
        words = page.get_text("words")
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
    try:
        return clean_whitespace(file.read().decode("utf-8", "ignore"))
    except Exception:
        return ""


def split_sentences(text: str):
    parts = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        parts.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", block))
    out = []
    for p in parts:
        p = p.strip(" -•")
        if not p:
            continue
        out.append(p)
    return out


def is_heading(line: str) -> bool:
    if len(line) <= 80 and line.isupper() and re.search(r"[A-Z]", line):
        return True
    if line.endswith(":") and len(line) <= 120:
        return True
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
    rx = re.compile(rf"(^|\n)\s*({tag}\s*\d*\.?:?)\s*", re.I)
    heads = list(rx.finditer(text))
    out = []
    for i, m in enumerate(heads):
        start = m.end()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        chunk = text[start:end].strip()
        header = m.group(2).strip().rstrip(":")
        if chunk and len(chunk) > 20:
            out.append((header, chunk))
        if len(out) >= cap:
            break
    return out


def extract_formulas(text: str, max_items: int = 20):
    candidates = []
    for line in text.split("\n"):
        ln = line.strip()
        if len(ln) < 2:
            continue
        if MATH_LINE_RX.search(ln):
            candidates.append(ln)
    uniq, seen = [], set()
    for ln in candidates:
        key = re.sub(r"\s+", " ", ln).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ln)
        if len(uniq) >= max_items:
            break
    return uniq


def is_mostly_math(line: str, threshold: float = 0.4) -> bool:
    if not line:
        return False
    total = len(line)
    math_chars = sum(1 for ch in line if re.match(MATH_CHARS, ch))
    return (math_chars / max(total, 1)) >= threshold


def tfidf_sentences(text: str, n: int) -> list:
    sents = split_sentences(text)
    sents = [s for s in sents if not is_mostly_math(s)]
    if not sents:
        return []
    if len(sents) <= n:
        return sents
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sents)
    sal = X.sum(axis=1).A.ravel()
    idx = np.argsort(-sal)[:n]
    idx = sorted(idx.tolist())
    return [sents[i] for i in idx]


def build_summary(raw: str, per_section: int, max_sections: int, max_formulas: int):
    sections = detect_sections(raw)
    seen, cleaned = set(), []
    for title, content in sections:
        key = re.sub(r"\W+", " ", title.lower()).strip()
        if not content or len(content) < 40:
            continue
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((title, content))
    cleaned = cleaned[:max_sections]

    defs = extract_blocks_by_tag(raw, "Definition", cap=16)
    thms = extract_blocks_by_tag(raw, "Theorem", cap=12)
    lems = extract_blocks_by_tag(raw, "Lemma", cap=8)
    corols = extract_blocks_by_tag(raw, "Corollary", cap=8)
    exs = extract_blocks_by_tag(raw, "Example", cap=12)
    formulas = extract_formulas(raw, max_items=max_formulas)

    return {
        "sections": cleaned,
        "defs": defs,
        "thms": thms,
        "lems": lems,
        "corols": corols,
        "exs": exs,
        "formulas": formulas,
    }


def answer_followup(question: str, raw: str, top_k: int = 4) -> str:
    sentences = split_sentences(raw)
    if not sentences:
        return "I couldn't find enough content in the document to answer that."
    vec = TfidfVectorizer(stop_words="english")
    corpus = [question] + sentences
    X = vec.fit_transform(corpus)
    q_vec = X[0]
    sent_vecs = X[1:]
    scores = (sent_vecs @ q_vec.T).toarray().ravel()
    if np.all(scores == 0):
        return "That question doesn't seem to match anything in the document."
    idx = np.argsort(-scores)[:top_k]
    chosen = [sentences[i] for i in idx if scores[i] > 0]
    if not chosen:
        return "That question doesn't seem to match anything in the document."
    return " ".join(chosen)


# -------------------------------------------------------------------
# MAIN ULTRA-MINIMAL UI
# -------------------------------------------------------------------
st.markdown('<div class="app-shell"><div class="app-card">', unsafe_allow_html=True)

# Title + subtitle
st.markdown(
    """
<div class="app-title">
  <h1>Kuki Docs Assistant</h1>
  <p>Summarize any document instantly.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Upload + Summarize controls
st.markdown('<div class="prompt-wrapper">', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)

summarize = st.button("Summarize Document")

st.markdown(
    '<div class="hint">Upload one file, then tap “Summarize Document”.</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Handle summarization
if summarize:
    if not uploaded:
        st.warning("Please upload a PDF or TXT file first.")
    else:
        text = read_uploaded(uploaded)
        if not text.strip():
            st.error("I couldn't extract any readable text from this file.")
        else:
            st.session_state.doc_text = text
            st.session_state.has_summary = True
            st.session_state.followup_answer = ""
            st.session_state.last_question = ""

# -------------------------------------------------------------------
# SUMMARY (IF AVAILABLE)
# -------------------------------------------------------------------
if st.session_state.has_summary and st.session_state.doc_text.strip():
    summary = build_summary(
        st.session_state.doc_text,
        per_section=PER_SECTION,
        max_sections=MAX_SECTIONS,
        max_formulas=MAX_FORMULAS,
    )

    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown("### Summary", unsafe_allow_html=True)

    sections = summary["sections"]
    if not sections:
        st.markdown(
            '<div class="kcard">No clear sections detected — showing top sentences.</div>',
            unsafe_allow_html=True,
        )
        pts = tfidf_sentences(st.session_state.doc_text, PER_SECTION * 3)
        if pts:
            st.markdown('<div class="kcard">', unsafe_allow_html=True)
            st.markdown("\n".join(f"- {p}" for p in pts))
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="section-title">Overview</div>', unsafe_allow_html=True
        )
        for title, content in sections:
            st.markdown(f"**{title}**")
            points = tfidf_sentences(content, PER_SECTION)
            if points:
                st.markdown('<div class="kcard">', unsafe_allow_html=True)
                st.markdown("\n".join(f"- {p}" for p in points))
                st.markdown("</div>", unsafe_allow_html=True)

    defs = summary["defs"]
    thms = summary["thms"]
    lems = summary["lems"]
    corols = summary["corols"]
    exs = summary["exs"]
    formulas = summary["formulas"]

    if defs:
        st.markdown(
            '<div class="section-title">Definitions</div>', unsafe_allow_html=True
        )
        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        for h, b in defs:
            st.markdown(f"- **{h}** — {b.splitlines()[0]}")
        st.markdown("</div>", unsafe_allow_html=True)

    if thms or lems or corols:
        st.markdown(
            '<div class="section-title">Results</div>', unsafe_allow_html=True
        )
        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        for coll in (thms, lems, corols):
            for h, b in coll:
                st.markdown(f"- **{h}** — {b.splitlines()[0]}")
        st.markdown("</div>", unsafe_allow_html=True)

    if exs:
        st.markdown(
            '<div class="section-title">Examples</div>', unsafe_allow_html=True
        )
        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        for h, b in exs:
            st.markdown(f"- **{h}** — {b.splitlines()[0]}")
        st.markdown("</div>", unsafe_allow_html=True)

    if formulas:
        st.markdown(
            '<div class="section-title">Key Formulas</div>', unsafe_allow_html=True
        )
        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        for fx in formulas:
            try:
                st.latex(fx)
            except Exception:
                st.markdown(f"- {fx}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- FOLLOW-UP INPUT ----------------
    st.markdown('<div class="followup-wrapper">', unsafe_allow_html=True)
    st.markdown(
        '<div class="followup-label">Ask a follow-up question…</div>',
        unsafe_allow_html=True,
    )

    with st.form("followup-form", clear_on_submit=False):
        st.markdown('<div class="followup-bar">', unsafe_allow_html=True)
        st.markdown('<div class="followup-input">', unsafe_allow_html=True)
        question = st.text_area(
            "Ask a follow-up question…",
            value=st.session_state.last_question,
            height=40,
            label_visibility="collapsed",
            key="followup_q",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        send = st.form_submit_button(" ")
        st.markdown('<div class="followup-send">➤</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if send and question.strip():
        st.session_state.last_question = question.strip()
        st.session_state.followup_answer = answer_followup(
            question.strip(), st.session_state.doc_text
        )

    if st.session_state.followup_answer:
        st.markdown(
            '<div class="followup-answer">', unsafe_allow_html=True
        )
        st.markdown(
            f"**Answer:** {st.session_state.followup_answer}",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end summary-card

st.markdown("</div></div>", unsafe_allow_html=True)  # end app-card + app-shell
