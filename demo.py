import io
import os
import re
from datetime import datetime

import docx
import requests
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Official, stable Google GenAI client
from google import genai
from google.genai import types

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-2"

PII_MODEL_DIR = os.path.join(os.path.dirname(__file__), "pii_model")

# --- ENTERPRISE COLOR PALETTE (Ejada / ehub) ---
NAVY_DEEP = "#06154D"
NAVY_ACCENT = "#001C7F"
MINT = "#00E676"
MINT_ALT = "#00D27A"
TEXT_LIGHT = "#F5F7FA"
TEXT_GRAY = "#AAB4D4"


def extract_text_from_upload(uploaded_file) -> str:
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    raw = uploaded_file.read()
    if ext == "pdf":
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == "docx":
        document = docx.Document(io.BytesIO(raw))
        return "\n".join(p.text for p in document.paragraphs)
    return raw.decode("utf-8", errors="ignore")


def build_masking_receipt_pdf(clean_text, masking_reason, timestamp, findings, masking_purpose) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleNavy", parent=styles["Title"], textColor=HexColor(NAVY_DEEP), fontSize=18)
    heading_style = ParagraphStyle("HeadingNavy", parent=styles["Heading2"], textColor=HexColor(NAVY_ACCENT))
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], textColor=HexColor("#1A1A1A"), fontSize=10, leading=14)

    type_counts = {}
    for f in findings:
        type_counts[f["type"]] = type_counts.get(f["type"], 0) + 1
    breakdown_lines = [f"{k}: {v} redacted" for k, v in type_counts.items()] or ["No PII detected"]

    story = [
        Paragraph("PII Masking — Documentation Receipt", title_style),
        Spacer(1, 12),
        Paragraph(f"<b>Timestamp:</b> {timestamp}", body_style),
        Paragraph(f"<b>Masking Justification:</b> {masking_reason}", body_style),
        Paragraph(f"<b>Masking Purpose:</b> {masking_purpose}", body_style),
        Paragraph(f"<b>Total Items Redacted:</b> {len(findings)}", body_style),
        Spacer(1, 8),
        Paragraph("Breakdown by PII Type:", heading_style),
    ]
    for line in breakdown_lines:
        story.append(Paragraph(f"— {line}", body_style))
    story.append(Spacer(1, 16))
    story.append(Paragraph("Masked Document:", heading_style))
    story.append(Spacer(1, 6))
    escaped = clean_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
    story.append(Paragraph(escaped, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# --- ENVIRONMENT & API SECURITY ---
# API keys are read only from the local .env file (GOOGLE_API_KEY, GROK_API_KEY).
# They are never collected via the UI, so they can't leak into screenshots,
# logs, or session state.
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROK_API_KEY = os.environ.get("GROK_API_KEY")

# --- PAGE LAYOUT & CONFIG ---
st.set_page_config(page_title="Ejada | Data Privacy Dashboard", page_icon="🔒", layout="wide")

# --- CUSTOM ENTERPRISE THEME (Navy / Mint) ---
st.markdown(
    f"""
    <style>
    :root {{
        --navy-deep: {NAVY_DEEP};
        --navy-accent: {NAVY_ACCENT};
        --mint: {MINT};
        --mint-alt: {MINT_ALT};
        --text-light: {TEXT_LIGHT};
        --text-gray: {TEXT_GRAY};
    }}

    .stApp {{
        background: linear-gradient(160deg, var(--navy-deep) 0%, var(--navy-accent) 100%);
    }}

    [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}

    [data-testid="stSidebar"] {{
        background: var(--navy-deep);
        border-right: 1px solid rgba(0,230,118,0.25);
    }}

    .brand-header {{
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        border-radius: 12px;
        background: linear-gradient(90deg, var(--navy-accent), var(--navy-deep));
        border-left: 4px solid var(--mint);
    }}
    .brand-title {{
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-light);
    }}
    .brand-subtitle {{
        font-size: 0.95rem;
        color: var(--mint);
        margin-top: 0.25rem;
        letter-spacing: 0.03em;
    }}

    h1, h2, h3, h4 {{ color: var(--text-light) !important; }}
    p, li, span, label {{ color: var(--text-gray); }}

    [data-testid="stTabs"] [role="tablist"] {{
        gap: 4px;
        border-bottom: 1px solid rgba(0,230,118,0.2);
    }}
    [data-testid="stTabs"] button[role="tab"] {{
        color: var(--text-gray);
        background: transparent;
        border-radius: 8px 8px 0 0;
    }}
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        color: var(--mint) !important;
        border-bottom: 3px solid var(--mint) !important;
        background: rgba(0,230,118,0.08);
        font-weight: 600;
    }}

    div.stButton > button, div.stDownloadButton > button {{
        background: linear-gradient(90deg, var(--mint), var(--mint-alt));
        color: var(--navy-deep);
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.55rem 1.2rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    div.stButton > button:hover, div.stDownloadButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(0,230,118,0.35);
        color: var(--navy-deep);
    }}

    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stFileUploaderDropzone"] {{
        background: rgba(255,255,255,0.05) !important;
        color: var(--text-light) !important;
        border: 1px solid rgba(0,230,118,0.3) !important;
        border-radius: 8px !important;
    }}
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stTextInput"] input:focus {{
        border-color: var(--mint) !important;
        box-shadow: 0 0 0 1px var(--mint) !important;
    }}

    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(0,230,118,0.3) !important;
        color: var(--text-light) !important;
        border-radius: 8px !important;
    }}

    [data-testid="stAlert"] {{
        border-radius: 10px !important;
        border-left: 4px solid var(--mint) !important;
        background: rgba(255,255,255,0.05) !important;
    }}

    [data-testid="stCodeBlock"] pre {{
        background: rgba(0,0,0,0.35) !important;
        border: 1px solid rgba(0,230,118,0.25) !important;
        border-radius: 8px !important;
        color: #E4F8ED !important;
    }}

    [data-testid="stCaptionContainer"] {{ color: var(--text-gray) !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="brand-header">
        <div class="brand-title">🔒 Data Privacy & Prompt Engineering Dashboard</div>
        <div class="brand-subtitle">Enterprise PII Governance • Ejada / ehub Design System</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not GOOGLE_API_KEY:
    st.error(
        "No `GOOGLE_API_KEY` found. Create a `.env` file in the project root with:\n\n"
        "```\nGOOGLE_API_KEY=your_key_here\n```"
    )

client = genai.Client() if GOOGLE_API_KEY else None


def _call_gemini(prompt: str, system_prompt: str):
    config = types.GenerateContentConfig(system_instruction=system_prompt) if system_prompt.strip() else None
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=config)
    return response.text, "Gemini (gemini-2.5-flash)"


def _call_grok(prompt: str, system_prompt: str):
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(
        GROK_API_URL,
        headers={"Authorization": f"Bearer {GROK_API_KEY}"},
        json={"model": GROK_MODEL, "messages": messages},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"], f"Grok ({GROK_MODEL})"


def generate_with_fallback(prompt: str, system_prompt: str = "", preferred: str = "gemini"):
    """Try the user-selected backend first; on any error (or if it's not
    configured), fall back to the other backend if available.
    Returns (response_text, model_used)."""
    backends = [
        ("gemini", _call_gemini, client is not None),
        ("grok", _call_grok, bool(GROK_API_KEY)),
    ]
    if preferred == "grok":
        backends.reverse()

    first_error = None
    for name, call_fn, available in backends:
        if not available:
            continue
        try:
            text, label = call_fn(prompt, system_prompt)
            if first_error is not None:
                label = f"{label} — fallback (preferred backend failed: {first_error})"
            return text, label
        except Exception as e:
            first_error = e

    if first_error is not None:
        raise first_error
    raise RuntimeError("No API key configured for Gemini or Grok.")


# ==========================================
# PII CLASSIFICATION & MASKING MODEL
# ==========================================
class PIIMaskingModel:
    """
    Wraps the fine-tuned BertForTokenClassification model in `pii_model/`
    (36 BIO labels: EMAIL, GIVENNAME, TELEPHONENUM, SOCIALNUM, etc.).
    detect(text) -> list of findings, mask(text) -> (clean_text, findings).
    """

    # Structurally-unambiguous PII formats where the token-classifier's raw
    # per-token BIO tags tend to leave gaps mid-entity (e.g. only the local
    # part of an email, or the area code of a phone number). Used as a
    # safety net so a real email/phone is never left partially exposed.
    SAFETY_PATTERNS = {
        "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "CREDITCARDNUMBER": re.compile(r"\b(?:\d[ -]?){13,16}\d\b"),
        "TELEPHONENUM": re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "SOCIALNUM": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    }

    def __init__(self, model_dir: str):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.nlp = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )

    @staticmethod
    def _overlaps(a_start, a_end, b_start, b_end):
        return not (a_end <= b_start or a_start >= b_end)

    def _apply_safety_net(self, text: str, findings: list):
        for pii_type, pattern in self.SAFETY_PATTERNS.items():
            for m in pattern.finditer(text):
                span_start, span_end = m.span()
                overlapping = [f for f in findings if self._overlaps(f["start"], f["end"], span_start, span_end)]
                if overlapping:
                    start = min([span_start] + [f["start"] for f in overlapping])
                    end = max([span_end] + [f["end"] for f in overlapping])
                    best_type = max(overlapping, key=lambda f: f["score"])["type"]
                    findings = [f for f in findings if f not in overlapping]
                    findings.append({"type": best_type, "start": start, "end": end, "score": 0.99})
                else:
                    findings.append({"type": pii_type, "start": span_start, "end": span_end, "score": 0.99})
        return sorted(findings, key=lambda f: f["start"])

    def detect(self, text: str, gap: int = 2):
        # The BIO tagger sometimes flips label within one PII instance on
        # punctuation/no-space runs (e.g. a phone number); bridge small gaps
        # between adjacent entity spans into a single finding.
        raw = sorted(self.nlp(text), key=lambda e: e["start"])
        merged = []
        for e in raw:
            if merged and e["start"] - merged[-1]["end"] <= gap:
                prev = merged[-1]
                prev["end"] = max(prev["end"], e["end"])
                if e["score"] > prev["score"]:
                    prev["type"] = e["entity_group"]
                    prev["score"] = e["score"]
            else:
                merged.append({
                    "type": e["entity_group"],
                    "start": e["start"],
                    "end": e["end"],
                    "score": e["score"],
                    "value": text[e["start"]:e["end"]],
                })
        merged = self._apply_safety_net(text, merged)
        for f in merged:
            f["value"] = text[f["start"]:f["end"]]
        return merged

    def mask(self, text: str, strict: bool = False):
        """strict=True (vendor sharing) uses a generic [REDACTED] tag that hides
        the PII category entirely. strict=False (LLM ingestion) keeps the typed
        tag (e.g. [REDACTED_EMAIL]) so a downstream model retains useful context."""
        findings = self.detect(text)
        clean_text = text
        for f in sorted(findings, key=lambda f: -f["start"]):
            tag = "[REDACTED]" if strict else f"[REDACTED_{f['type']}]"
            clean_text = clean_text[: f["start"]] + tag + clean_text[f["end"] :]
        return clean_text, findings


@st.cache_resource(show_spinner="Loading PII classification model...")
def load_pii_model():
    return PIIMaskingModel(PII_MODEL_DIR)


pii_model = load_pii_model()

# --- TABS ---
tab1, tab2 = st.tabs(["🔒 PII Masking & Documentation", "💡 Prompt Engineering Techniques"])

# ==========================================
# TAB 1: PII MASKING & DOCUMENTATION
# ==========================================
with tab1:
    st.header("PII Masking & Masking-Log Documentation")
    st.write(
        "Upload or paste a sensitive document, log the reason it needs to be shared, "
        "then run it through the PII masking pipeline before handing it off."
    )

    masking_purpose = st.selectbox(
        "Masking Purpose",
        [
            "AI Tool Ingestion (Optimized for Large Language Models)",
            "External Vendor Sharing (Strict Data Privacy Redaction)",
        ],
    )
    st.caption(
        "🔒 Masking always executes locally via the fine-tuned PII model below — "
        "no document text is ever sent to an external API."
    )

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "Upload a document (.txt, .csv, .log, .md, .json, .pdf, .docx)",
            type=["txt", "csv", "log", "md", "json", "pdf", "docx"],
        )
        pasted_text = st.text_area(
            "...or paste the document text directly",
            height=200,
            placeholder="e.g. Please contact Basil at basil@example.com or (555) 123-4567 regarding the ticket.",
        )

    with col2:
        masking_reason = st.text_area(
            "Reason for masking (documentation log)",
            height=100,
            placeholder="e.g. Sharing logs with vendor for troubleshooting",
        )

    run_pipeline = st.button("🚀 Run PII Masking Pipeline", type="primary")

    if run_pipeline:
        source_text = ""
        if uploaded_file is not None:
            source_text = extract_text_from_upload(uploaded_file)
        elif pasted_text.strip():
            source_text = pasted_text

        if not source_text.strip():
            st.warning("Please upload a file or paste some text before running the pipeline.")
        elif not masking_reason.strip():
            st.warning("Please document a reason for masking before running the pipeline.")
        else:
            strict = masking_purpose.startswith("External Vendor")

            with st.spinner("Running PII classification & masking model..."):
                clean_text, findings = pii_model.mask(source_text, strict=strict)

            st.success("✅ Document successfully masked. Clean text is safe to share:")
            st.text_area("Masked Output", value=clean_text, height=200, disabled=True)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            type_counts = {}
            for f in findings:
                type_counts[f["type"]] = type_counts.get(f["type"], 0) + 1
            breakdown = (
                "\n".join(f"- **{k}**: {v} redacted" for k, v in type_counts.items())
                if type_counts
                else "- No PII detected"
            )

            st.markdown("---")
            st.markdown("### 📋 Masking Log / Documentation Receipt")
            st.markdown(
                f"""
- **Timestamp:** {timestamp}
- **Masking Justification:** {masking_reason}
- **Masking Purpose:** {masking_purpose}
- **Total Items Redacted:** {len(findings)}

**Breakdown by PII Type:**
{breakdown}
"""
            )

            pdf_bytes = build_masking_receipt_pdf(
                clean_text, masking_reason, timestamp, findings, masking_purpose
            )
            st.download_button(
                label="📄 Download PDF Receipt",
                data=pdf_bytes,
                file_name=f"masking_receipt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )

# ==========================================
# TAB 2: PROMPT ENGINEERING TECHNIQUES
# ==========================================
with tab2:
    st.header("Prompt Engineering Techniques")
    st.write("Compare how Zero-Shot, Few-Shot, and Role-Prompting strategies shape a live Gemini response.")

    technique = st.selectbox(
        "Select a prompt engineering strategy",
        ["Zero-Shot", "Few-Shot", "Role-Prompting"],
    )
    model_router = st.selectbox(
        "Target Backend Transformer Model",
        [
            "Google Gemini (gemini-2.5-flash)",
            "xAI Grok (grok-2)",
        ],
    )
    topic = st.text_input("Core topic:", "Why data privacy matters in AI systems")
    system_prompt = st.text_area(
        "System prompt (optional)",
        height=80,
        placeholder="e.g. You are a concise technical writer. Always answer in plain English, no jargon.",
        help="Sent separately from the user prompt via the model's system_instruction / system role.",
    )

    if technique == "Zero-Shot":
        engineered_prompt = f"Explain the following topic clearly and concisely: {topic}"
    elif technique == "Few-Shot":
        engineered_prompt = (
            "Answer in exactly one sentence, following the style of these examples.\n\n"
            "Topic: Encryption\nAnswer: Encryption scrambles data so only authorized parties can read it.\n\n"
            "Topic: Firewalls\nAnswer: Firewalls filter network traffic to block unauthorized access.\n\n"
            f"Topic: {topic}\nAnswer:"
        )
    else:  # Role-Prompting
        engineered_prompt = (
            "You are a senior data privacy officer briefing a company's executive team. "
            f"In a formal, authoritative tone, explain: {topic}"
        )

    if system_prompt.strip():
        st.markdown("**System prompt:**")
        st.code(system_prompt, language="text")
    st.markdown("**Engineered prompt sent to the transformer model:**")
    st.code(engineered_prompt, language="text")

    generate = st.button("✨ Generate AI Response")

    if generate:
        if client is None and not GROK_API_KEY:
            st.error("Cannot generate: neither `GOOGLE_API_KEY` nor `GROK_API_KEY` is set in the environment.")
        else:
            preferred = "grok" if "Grok" in model_router else "gemini"
            with st.spinner(f"Querying {model_router} (falls back to the other backend if unavailable)..."):
                try:
                    text, model_used = generate_with_fallback(engineered_prompt, system_prompt, preferred=preferred)
                    st.success(f"Model Response — via {model_used}:")
                    st.write(text)
                except Exception as e:
                    st.error(f"API Error: {e}")
