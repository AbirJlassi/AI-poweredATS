"""
CV Parser Benchmark Dashboard
Comparaison côte-à-côte : Classic (Regex+NER) | Groq LLM | Ollama Local
"""

import streamlit as st
import time
import json
import tempfile
import os
import sys
from pathlib import Path
from dataclasses import asdict

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV Parser Benchmark",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0e0f14;
    --surface:   #16181f;
    --surface2:  #1e2029;
    --border:    #2a2d3a;
    --classic:   #4ade80;
    --groq:      #60a5fa;
    --local:     #f472b6;
    --text:      #e2e4ed;
    --muted:     #6b7280;
    --accent:    #a78bfa;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3, h4 { font-family: 'Space Mono', monospace; }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Metric cards */
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.metric-card .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    margin-bottom: 4px;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
}

/* Parser header badges */
.parser-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}
.badge-classic { background: #052e16; color: var(--classic); border: 1px solid var(--classic); }
.badge-groq    { background: #1e3a5f; color: var(--groq);    border: 1px solid var(--groq); }
.badge-local   { background: #4a044e; color: var(--local);   border: 1px solid var(--local); }

/* Field rows */
.field-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    gap: 8px;
}
.field-row:last-child { border-bottom: none; }
.field-key {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    min-width: 80px;
    padding-top: 2px;
}
.field-val {
    font-size: 14px;
    color: var(--text);
    text-align: right;
    word-break: break-word;
    max-width: 200px;
}
.field-null { color: var(--muted); font-style: italic; font-size: 12px; }

/* Tag pills */
.tag-wrap { display: flex; flex-wrap: wrap; gap: 5px; justify-content: flex-end; }
.tag {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-family: 'Space Mono', monospace;
}
.tag-classic { background: #052e16; color: var(--classic); }
.tag-groq    { background: #1e3a5f; color: var(--groq); }
.tag-local   { background: #4a044e; color: var(--local); }
.tag-neutral { background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }

/* Section titles */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    padding: 12px 0 6px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}

/* Experience / Education cards */
.exp-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
}
.exp-title { font-weight: 600; color: var(--text); }
.exp-sub   { color: var(--muted); font-size: 12px; margin-top: 2px; }
.exp-period{ font-family: 'Space Mono', monospace; font-size: 11px; color: var(--accent); margin-top: 4px; }
.exp-desc  { color: var(--muted); font-size: 12px; margin-top: 6px; line-height: 1.5; }

/* Timer display */
.timer-display {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    margin: 4px 0;
}
.timer-classic { color: var(--classic); }
.timer-groq    { color: var(--groq); }
.timer-local   { color: var(--local); }

/* Status badges */
.status-ok  { color: #4ade80; }
.status-err { color: #f87171; }
.status-skip{ color: #6b7280; }

/* Diff highlight */
.diff-same { }
.diff-only { opacity: 0.5; }

/* Scrollable section */
.scroll-box {
    max-height: 320px;
    overflow-y: auto;
    padding-right: 4px;
}

/* Column headers */
.col-header {
    text-align: center;
    padding: 14px;
    border-radius: 10px 10px 0 0;
    margin-bottom: 12px;
    font-family: 'Space Mono', monospace;
}
.col-classic { background: linear-gradient(135deg, #052e16, #0a3d20); border: 1px solid var(--classic); }
.col-groq    { background: linear-gradient(135deg, #1e3a5f, #1e40af20); border: 1px solid var(--groq); }
.col-local   { background: linear-gradient(135deg, #4a044e, #6b21a820); border: 1px solid var(--local); }

/* Coverage bar */
.cov-bar-bg {
    background: var(--border);
    border-radius: 4px;
    height: 6px;
    margin-top: 6px;
    overflow: hidden;
}
.cov-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

/* JSON viewer */
.json-block {
    background: #0a0c10;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #a0aec0;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}

/* Tabs override */
[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.5px;
}

button[data-testid="baseButton-secondary"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
}

/* Stickers for sidebar */
.sticker {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def cv_to_dict(cv) -> dict:
    """Convert CVData (dataclass or dict) to plain dict."""
    if cv is None:
        return {}
    if hasattr(cv, '__dataclass_fields__'):
        d = asdict(cv)
        d.pop("_confidence", None)
        return d
    if isinstance(cv, dict):
        return cv
    return {}

def coverage_score(d: dict) -> float:
    """% of non-empty fields."""
    if not d:
        return 0.0
    scalar_fields = ["full_name", "email", "phone", "location", "linkedin", "github", "summary"]
    list_fields   = ["skills", "languages", "experiences", "education", "certifications"]
    total = len(scalar_fields) + len(list_fields)
    filled = sum(1 for f in scalar_fields if d.get(f)) + \
             sum(1 for f in list_fields   if d.get(f))
    return round(filled / total * 100, 1)

def fmt_time(t: float | None) -> str:
    if t is None: return "—"
    if t < 1:     return f"{t*1000:.0f} ms"
    return f"{t:.2f} s"

def render_tag(text: str, color_class: str) -> str:
    return f'<span class="tag {color_class}">{text}</span>'

def render_tags(items: list, color_class: str) -> str:
    if not items:
        return '<span class="field-null">none</span>'
    tags = "".join(render_tag(str(i), color_class) for i in items)
    return f'<div class="tag-wrap">{tags}</div>'

def render_scalar(val) -> str:
    if val is None or val == "":
        return '<span class="field-null">—</span>'
    return f'<span class="field-val">{val}</span>'

def render_exp_card(exp: dict) -> str:
    title   = exp.get("title") or exp.get("degree") or "—"
    company = exp.get("company") or exp.get("institution") or exp.get("school") or ""
    period  = exp.get("period") or exp.get("date_range") or ""
    desc    = exp.get("description") or ""
    return f"""
    <div class="exp-card">
        <div class="exp-title">{title}</div>
        {"<div class='exp-sub'>"+company+"</div>" if company else ""}
        {"<div class='exp-period'>"+period+"</div>" if period else ""}
        {"<div class='exp-desc'>"+str(desc)[:200]+"</div>" if desc else ""}
    </div>"""

def field_row(key: str, val_html: str) -> str:
    return f"""
    <div class="field-row">
        <span class="field-key">{key}</span>
        <span>{val_html}</span>
    </div>"""


# ── Parser runners ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_classic_parser():
    sys.path.insert(0, "/home/claude")
    from cv_parser_classic import ClassicCVParser
    return ClassicCVParser()

def run_classic(file_path: str):
    try:
        parser = load_classic_parser()
        t0 = time.perf_counter()
        result = parser.parse(file_path)
        elapsed = time.perf_counter() - t0
        return cv_to_dict(result), elapsed, None
    except Exception as e:
        return {}, None, str(e)

def run_groq(file_path: str, model: str, api_key: str):
    try:
        sys.path.insert(0, "/home/claude")
        os.environ["GROQ_API_KEY"] = api_key
        # Re-import to pick up env var
        import importlib
        if "cv_parser_llm_groq" in sys.modules:
            mod = importlib.reload(sys.modules["cv_parser_llm_groq"])
        else:
            import cv_parser_llm_groq as mod
        parser = mod.LLMCVParser(model=model)
        t0 = time.perf_counter()
        result = parser.parse(file_path)
        elapsed = time.perf_counter() - t0
        return cv_to_dict(result), elapsed, None
    except Exception as e:
        return {}, None, str(e)

def run_local(file_path: str, model: str):
    try:
        sys.path.insert(0, "/home/claude")
        import importlib
        if "cv_parser_llm_local" in sys.modules:
            mod = importlib.reload(sys.modules["cv_parser_llm_local"])
        else:
            import cv_parser_llm_local as mod
        parser = mod.LLMCVParser(model=model)
        t0 = time.perf_counter()
        result = parser.parse(file_path)
        elapsed = time.perf_counter() - t0
        return cv_to_dict(result), elapsed, None
    except Exception as e:
        return {}, None, str(e)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📄 CV Parser\n### Benchmark")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload CV", type=["pdf", "docx", "txt"],
        help="PDF, DOCX ou TXT"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Parsers actifs")

    use_classic = st.checkbox("Classic NLP (Regex + SpaCy NER)", value=True)
    use_groq    = st.checkbox("Groq LLM (cloud)", value=False)
    use_local   = st.checkbox("Ollama (local)", value=False)

    if use_groq:
        st.markdown('<div class="sticker">🔑 <b>Groq API Key</b></div>', unsafe_allow_html=True)
        groq_key = st.text_input("API Key", type="password", placeholder="gsk_...")
        groq_model = st.selectbox("Modèle Groq", [
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "llama-3.1-8b-instant",
            "qwen/qwen3-32b",
        ])
    else:
        groq_key = ""
        groq_model = "llama-3.1-8b-instant"

    if use_local:
        st.markdown('<div class="sticker">🖥️ <b>Ollama</b> doit tourner en local</div>', unsafe_allow_html=True)
        local_model = st.selectbox("Modèle Ollama", [
            "mistral", "llama3", "gemma2", "phi3", "mixtral"
        ])
    else:
        local_model = "mistral"

    st.markdown("---")
    run_btn = st.button("▶ Lancer l'analyse", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
<div style="font-size:11px; color: #6b7280; line-height:1.6">
<b>Classic</b> — Regex + SpaCy NER<br>
<b>Groq</b> — LLaMA3 / Qwen via API<br>
<b>Local</b> — Mistral / LLaMA3 via Ollama<br><br>
Même schéma CVData pour les 3.
</div>
""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 24px 0 16px 0">
    <h1 style="margin:0; font-size: 28px; letter-spacing: -0.5px">
        CV Parser <span style="color: #a78bfa">Benchmark</span>
    </h1>
    <p style="color: #6b7280; margin: 6px 0 0 0; font-size: 14px">
        Comparaison côte-à-côte — Classic · Groq · Ollama
    </p>
</div>
""", unsafe_allow_html=True)

if not uploaded:
    st.markdown("""
    <div style="
        border: 2px dashed #2a2d3a;
        border-radius: 16px;
        padding: 60px;
        text-align: center;
        margin-top: 40px;
    ">
        <div style="font-size: 48px; margin-bottom: 16px">📋</div>
        <div style="font-family: 'Space Mono', monospace; font-size: 16px; color: #6b7280">
            Uploadez un CV dans la sidebar pour commencer
        </div>
        <div style="font-size: 13px; color: #4b5563; margin-top: 8px">
            PDF · DOCX · TXT — FR & EN
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Run parsers ────────────────────────────────────────────────────────────────

if run_btn:
    # Save uploaded file to temp
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    results   = {}   # parser_name -> dict
    timings   = {}   # parser_name -> float
    errors    = {}   # parser_name -> str | None

    parsers_to_run = []
    if use_classic: parsers_to_run.append("classic")
    if use_groq:    parsers_to_run.append("groq")
    if use_local:   parsers_to_run.append("local")

    progress = st.progress(0, text="Analyse en cours…")

    for i, name in enumerate(parsers_to_run):
        progress.progress((i) / len(parsers_to_run), text=f"⏳ {name}…")
        if name == "classic":
            r, t, e = run_classic(tmp_path)
        elif name == "groq":
            if not groq_key:
                r, t, e = {}, None, "GROQ_API_KEY manquante"
            else:
                r, t, e = run_groq(tmp_path, groq_model, groq_key)
        elif name == "local":
            r, t, e = run_local(tmp_path, local_model)

        results[name] = r
        timings[name] = t
        errors[name]  = e

    progress.progress(1.0, text="✅ Terminé")
    time.sleep(0.3)
    progress.empty()

    os.unlink(tmp_path)
    st.session_state["results"] = results
    st.session_state["timings"] = timings
    st.session_state["errors"]  = errors
    st.session_state["parsers"] = parsers_to_run
    st.session_state["filename"] = uploaded.name

# ── Display results ────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.stop()

results  = st.session_state["results"]
timings  = st.session_state["timings"]
errors   = st.session_state["errors"]
parsers  = st.session_state["parsers"]
filename = st.session_state.get("filename", "")

LABEL = {
    "classic": "Classic · Regex+NER",
    "groq":    "Groq · " + groq_model,
    "local":   "Ollama · " + local_model,
}
BADGE_STYLE = {
    "classic": "background:#052e16; color:#4ade80; border:1px solid #4ade80;",
    "groq":    "background:#1e3a5f; color:#60a5fa; border:1px solid #60a5fa;",
    "local":   "background:#4a044e; color:#f472b6; border:1px solid #f472b6;",
}
TIMER_COLOR = {"classic": "#4ade80", "groq": "#60a5fa", "local": "#f472b6"}
BAR_COLOR   = {"classic": "#4ade80", "groq": "#60a5fa", "local": "#f472b6"}

# ── Filename ───────────────────────────────────────────────────────────────────
st.caption("📁 " + filename)

# ── Top metrics strip ──────────────────────────────────────────────────────────
metric_cols = st.columns(len(parsers))
for col, name in zip(metric_cols, parsers):
    cov = coverage_score(results[name])
    t   = timings[name]
    err = errors[name]
    with col:
        badge_style = BADGE_STYLE[name]
        timer_color = TIMER_COLOR[name]
        bar_color   = BAR_COLOR[name]
        ok_label    = "OK" if not err else "Erreur"
        ok_icon     = "✅" if not err else "❌"

        # Badge + timer block — plain HTML, no dynamic values inside nested quotes
        badge_html = (
            "<div style='text-align:center; padding:14px; border-radius:10px;"
            " border:1px solid #2a2d3a; background:#16181f; margin-bottom:4px'>"
            "<span style='display:inline-block; padding:4px 14px; border-radius:20px;"
            " font-family:Space Mono,monospace; font-size:12px; font-weight:700; "
            + badge_style
            + "'>" + LABEL[name] + "</span>"
            "<div style='font-family:Space Mono,monospace; font-size:30px; font-weight:700;"
            " margin:10px 0 4px; color:" + timer_color + "'>" + fmt_time(t) + "</div>"
            "<div style='font-size:12px; color:#9ca3af'>"
            + ok_icon + " " + ok_label
            + " &nbsp;&middot;&nbsp; Couverture : <b style='color:#e2e4ed'>"
            + str(cov) + "%</b></div>"
            "<div style='background:#2a2d3a; border-radius:4px; height:5px;"
            " margin-top:8px; overflow:hidden'>"
            "<div style='height:100%; border-radius:4px; background:" + bar_color
            + "; width:" + str(cov) + "%'></div></div>"
            "</div>"
        )
        st.markdown(badge_html, unsafe_allow_html=True)
        if err:
            st.error("⚠️ " + err[:200])

st.markdown("---")

# ── 2 Tabs ─────────────────────────────────────────────────────────────────────
tab_compare, tab_json = st.tabs(["📊  Comparaison", "{ }  JSON brut"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Comparaison complète
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    cols = st.columns(len(parsers))

    for col, name in zip(cols, parsers):
        d = results[name]

        with col:
            # ── Parser badge ──────────────────────────────────────────────
            st.markdown(
                "<span style='display:inline-block; padding:3px 12px; border-radius:20px;"
                " font-family:Space Mono,monospace; font-size:11px; font-weight:700; "
                + BADGE_STYLE[name] + "'>" + LABEL[name] + "</span>",
                unsafe_allow_html=True,
            )

            # ── CONTACT ───────────────────────────────────────────────────
            st.markdown("**Contact**")
            contact_rows = [
                ("Nom",      d.get("full_name")),
                ("Email",    d.get("email")),
                ("Tél",      d.get("phone")),
                ("Lieu",     d.get("location")),
                ("LinkedIn", d.get("linkedin")),
                ("GitHub",   d.get("github")),
            ]
            for label, val in contact_rows:
                left, right = st.columns([1, 2])
                left.caption(label)
                right.write(val if val else "—")

            # ── LANGUE + RÉSUMÉ ───────────────────────────────────────────
            if d.get("detected_lang"):
                st.markdown("**Langue détectée**")
                st.write(d["detected_lang"].upper())

            summary = (d.get("summary") or "").strip()
            if summary:
                st.markdown("**Résumé**")
                st.caption(summary[:400])

            # ── LANGUES PARLÉES ───────────────────────────────────────────
            st.markdown("**Langues**")
            langs_raw = d.get("languages") or []
            if langs_raw:
                lang_strs = []
                for l in langs_raw:
                    if isinstance(l, str):
                        lang_strs.append(l)
                    elif isinstance(l, dict):
                        lang_strs.append(
                            l.get("language", "") + (" (" + l.get("level", "") + ")" if l.get("level") else "")
                        )
                st.write(" · ".join(lang_strs))
            else:
                st.write("—")

            # ── CERTIFICATIONS ────────────────────────────────────────────
            st.markdown("**Certifications**")
            certs = d.get("certifications") or []
            if certs:
                for cert in certs:
                    st.write("• " + str(cert))
            else:
                st.write("—")

            st.divider()

            # ── EXPÉRIENCES ───────────────────────────────────────────────
            experiences = d.get("experiences") or []
            st.markdown("**💼 Expériences** — " + str(len(experiences)) + " trouvée(s)")
            if not experiences:
                st.caption("Aucune expérience extraite")
            else:
                for exp in experiences:
                    title   = exp.get("title")   or exp.get("degree")      or ""
                    company = exp.get("company")  or exp.get("institution") or exp.get("school") or ""
                    period  = exp.get("period")   or exp.get("date_range")  or ""
                    desc    = exp.get("description") or ""
                    with st.container(border=True):
                        if title:
                            st.write("**" + title + "**")
                        if company:
                            st.caption(company)
                        if period:
                            st.caption("🗓 " + period)
                        if desc:
                            st.caption(str(desc)[:250])

            st.divider()

            # ── FORMATION ─────────────────────────────────────────────────
            education = d.get("education") or []
            st.markdown("**🎓 Formation** — " + str(len(education)) + " trouvée(s)")
            if not education:
                st.caption("Aucune formation extraite")
            else:
                for edu in education:
                    degree  = edu.get("degree")      or edu.get("title")   or ""
                    school  = edu.get("institution")  or edu.get("school")  or edu.get("company") or ""
                    period  = edu.get("period")       or edu.get("date_range") or ""
                    with st.container(border=True):
                        if degree:
                            st.write("**" + degree + "**")
                        if school:
                            st.caption(school)
                        if period:
                            st.caption("🗓 " + period)

            st.divider()

            # ── COMPÉTENCES ───────────────────────────────────────────────
            skills_raw = d.get("skills") or []
            skill_strs = [s if isinstance(s, str) else str(s) for s in skills_raw]
            st.markdown("**🔧 Compétences** — " + str(len(skill_strs)) + " trouvée(s)")
            if not skill_strs:
                st.caption("Aucune compétence extraite")
            else:
                # Display as a wrapped text block — no HTML needed
                st.write("  ".join(skill_strs))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — JSON brut
# ═══════════════════════════════════════════════════════════════════════════════
with tab_json:
    cols = st.columns(len(parsers))
    for col, name in zip(cols, parsers):
        with col:
            st.markdown(
                "<span style='display:inline-block; padding:3px 12px; border-radius:20px;"
                " font-family:Space Mono,monospace; font-size:11px; font-weight:700; "
                + BADGE_STYLE[name] + "'>" + LABEL[name] + "</span>",
                unsafe_allow_html=True,
            )
            json_str = json.dumps(results[name], ensure_ascii=False, indent=2)
            st.code(json_str, language="json")
            st.download_button(
                label="⬇ Télécharger JSON",
                data=json_str,
                file_name="cv_" + name + ".json",
                mime="application/json",
                key="dl_" + name,
            )