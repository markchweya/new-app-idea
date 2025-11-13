import re
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# PDF extraction
try:
    import fitz
    PYMUPDF_OK = True
except:
    PYMUPDF_OK = False

# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(
    page_title="Math Notes Summarizer • Kuki AI",
    layout="wide",
)

# ----------------------------------------
# GLOBAL CSS — Modern, animated UI
# ----------------------------------------
st.markdown("""
<style>
/* ---------- PAGE RESET ---------- */
#MainMenu, header, footer {visibility: hidden;}
html, body {
    background: #05060f;
    font-family: 'Inter', sans-serif;
    scroll-behavior: smooth;
}

/* ---------- LAYOUT ---------- */
.container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 22px;
}

.card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(15px);
    border-radius: 22px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.08);
    animation: fadeIn 0.8s ease;
}

/* Floating header */
.topbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(5,6,15,0.85);
    backdrop-filter: blur(14px);
    padding: 14px 0 18px 0;
    margin-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ---------- HERO ---------- */
.hero {
    background: radial-gradient(circle at 20% -10%, rgba(120,150,255,0.17), transparent),
                linear-gradient(180deg, rgba(255,255,255,0.05), transparent);
    border-radius: 26px;
    padding: 28px;
    animation: slideUp 1s ease;
}
.hero h1 {
    font-size: 2.1rem;
    font-weight: 800;
}

/* ---------- BUTTONS ---------- */
.stButton>button {
    background: linear-gradient(135deg, #687bff40, #4450ff20);
    border: 1px solid rgba(120,140,255,0.25);
    color: white;
    font-weight: 700;
    border-radius: 14px;
    padding: 12px 18px;
    transition: 0.2s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #8091ff60, #5565ff30);
    box-shadow: 0 4px 14px rgba(80,110,255,0.35);
}

/* ---------- OUTPUT AREA ---------- */
.output-section {
    animation: fadeIn 1s ease;
}
.kcard {
    background: rgba(255,255,255,0.05);
    padding: 14px;
    border-radius: 14px;
    margin: 8px 0;
    border: 1px solid rgba(255,255,255,0.1);
}

/* ---------- ANIMATIONS ---------- */
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(10px);}
  to {opacity: 1; transform: translateY(0);}
}
@keyframes slideUp {
  from {opacity: 0; transform: translateY(20px);}
  to {opacity: 1; transform: translateY(0);}
}

/* Make text readable */
h3, h2, p, label, textarea, .stSlider {color: #f3f3f3 !important;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# HERO SECTION
# ----------------------------------------
st.markdown("""
<div class="topbar">
    <div class="hero">
        <h1>📘 Math Notes Summarizer — Modern UI Edition</h1>
        <p>Upload or paste your math notes. Get clean, structured summaries with formula detection & theorem extraction.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------
# LAYOUT
# ----------------------------------------
st.markdown('<div class="container">', unsafe_allow_html=True)

# LEFT PANEL -------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Input Area")

uploaded = st.file_uploader("Upload PDFs or Text Files", type=["pdf", "txt"], accept_multiple_files=True)
pasted = st.text_area("Paste Notes Here", height=240)

col1, col2, col3 = st.columns(3)
with col1:
    per_section = st.slider("Sentences per section", 2, 10, 5)
with col2:
    max_sections = st.slider("Max sections", 3, 20, 10)
with col3:
    max_formulas = st.slider("Max formulas", 5, 40, 20)

generate = st.button("⚡ Generate Summary")
st.markdown('</div>', unsafe_allow_html=True)

# RIGHT PANEL ------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Summary Output")
out_placeholder = st.empty()
download_placeholder = st.empty()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# 🧠 TEXT CLEANING + MATH DETECTOR IMPROVED
# ==========================================================

MATH_CHARS = r"[=+\-*/×÷∑∏∫∂∇∞≈≃≅≡≤≥→←↦αβγδϵεθλμνπρστυφχψωΩΔ∈⊂⊆⊇∪∩√^_]"
LATEX_MATH = r"(\$.*?\$|\$\$.*?\$\$|\\begin\{equation\}.*?\\end\{equation\})"

MATH_RX = re.compile(rf"{LATEX_MATH}|{MATH_CHARS}", re.I)

def extract_formulas(text):
    lines = text.split("\n")
    formulas = []

    # detect real math blocks
    block = []
    in_block = False

    for ln in lines:
        if "$$" in ln:
            if not in_block:
                in_block = True
                block = [ln]
            else:
                block.append(ln)
                formulas.append("\n".join(block))
                in_block = False
            continue

        if in_block:
            block.append(ln)
            continue

        # single-line math
        if MATH_RX.search(ln):
            formulas.append(ln)

    # remove duplicates
    cleaned = []
    seen = set()
    for f in formulas:
        key = f.strip().lower()
        if key not in seen:
            cleaned.append(f)
            seen.add(key)

    return cleaned[:max_formulas]

# ==========================================================
# GENERATE SUMMARY
# ==========================================================

if "corpus" not in st.session_state:
    st.session_state.corpus = ""

# Combine uploads
if uploaded:
    texts = []
    for f in uploaded:
        if f.name.endswith(".txt"):
            texts.append(f.read().decode("utf-8"))
        else:
            if PYMUPDF_OK:
                doc = fitz.open(stream=f.read(), filetype="pdf")
                txt = "\n".join(page.get_text() for page in doc)
                texts.append(txt)
    st.session_state.corpus = "\n\n".join(texts)

if pasted.strip():
    st.session_state.corpus += "\n\n" + pasted.strip()

if generate:
    raw = st.session_state.corpus

    if not raw.strip():
        out_placeholder.warning("Please upload or paste notes first.")
    else:
        # Basic split
        sections = raw.split("\n\n")
        sections = [s.strip() for s in sections if len(s.strip()) > 40]
        sections = sections[:max_sections]

        formulas = extract_formulas(raw)

        with out_placeholder.container():
            st.markdown('<div class="output-section">', unsafe_allow_html=True)

            st.markdown("## 📌 Overview")
            for sc in sections:
                sentences = re.split(r"\. |\n", sc)
                pts = sentences[:per_section]
                st.markdown('<div class="kcard">', unsafe_allow_html=True)
                for p in pts:
                    st.markdown(f"- {p}")
                st.markdown('</div>', unsafe_allow_html=True)

            if formulas:
                st.markdown("## 🔢 Key Formulas")
                st.markdown('<div class="kcard">', unsafe_allow_html=True)
                for f in formulas:
                    try:
                        st.latex(f)
                    except:
                        st.markdown(f"```\n{f}\n```")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        download_placeholder.download_button(
            "⬇️ Download Summary (.txt)",
            "\n\n".join(sections).encode("utf-8"),
            file_name="summary.txt",
        )
