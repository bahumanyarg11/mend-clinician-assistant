"""
app_integrated.py  (Part 1)
============================
Mend Clinician Assistant — a decision-support tool for psychologists.
The psychologist enters what the client says (or their own notes/questions);
EnhancedChatbot → GraphRAG KG → LLM returns a clinical brief for the psychologist.
The client does not see or interact with this tool.

Run:
    streamlit run app_integrated.py

The sidebar now shows:
  - Affect + Intent (from emotion/behaviour models)
  - Clinical tags from the KG (Disorders, Techniques, Contexts)
  - KG confidence score
  - Session turn count

To change the LLM: edit BACKEND and MODEL at the top of this file.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import streamlit as st


def _pretty(label: str) -> str:
    """'shame_guilt' / 'FeelingTenseOrOnEdge' -> 'Shame Guilt' / 'Feeling Tense Or On Edge'."""
    label = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", str(label).replace("_", " "))
    return " ".join(w[:1].upper() + w[1:] for w in label.split())

# ── Caching Heavy Models ─────────────────────────────────────────────────────────
@st.cache_resource
def get_pii_scrubber():
    from pii_scrubber import PIIScrubber
    return PIIScrubber()

@st.cache_resource
def get_transcriber():
    from transcriber import Transcriber
    # Use 'base' or 'tiny' for responsive frontend, or 'large-v3' if quality > speed
    return Transcriber(model_name="base", device="auto")

# ── LLM selection — change these to switch models ───────────────────────────────
# Start-Mend.bat sets these (lab gemma4:26b over an SSH tunnel, or a local model); the defaults are the old hard-coded values.
BACKEND     = os.environ.get("LLM_BACKEND", "ollama")            # "ollama" | "openai" | "gemini" | "anthropic"
MODEL       = os.environ.get("LLM_MODEL", "gemma4:26b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
API_KEY     = os.environ.get(f"{BACKEND.upper()}_API_KEY", "")   # Required for openai / gemini / anthropic

# Session log directory (for EHR review later)
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sessions")

from enhanced_chatbot import EnhancedChatbot, make_adapter

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mend | Clinician Assistant",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root { --ink:#132b27; --muted:#647873; --cream:#f5f7f5; --accent:#167a68; --line:#dfe8e3; }
.stApp { background:var(--cream); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:rgba(245,247,245,.82); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#102f2a 0%,#18463c 100%); border-right:0; }
[data-testid="stSidebar"] * { color:#effaf5 !important; }
[data-testid="stSidebar"] .stCaption { color:#a6c6ba !important; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.13); }
[data-testid="stSidebar"] button { border:1px solid rgba(255,255,255,.16); background:rgba(255,255,255,.08); }
[data-testid="stSidebar"] button:hover { border-color:#8de1c2; background:rgba(141,225,194,.16); }
[data-testid="stSidebar"] [data-testid="stMetric"] { background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.1);
  border-radius:14px; padding:.65rem .7rem; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color:#a6c6ba !important; font-size:.68rem; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#f4fffa !important; font-size:.95rem; }
.brand { display:flex; align-items:center; gap:.65rem; margin:.15rem 0 1.5rem; }
.brand-mark { display:grid; place-items:center; width:34px; height:34px; border-radius:11px;
  background:#8de1c2; color:#10392f !important; font-weight:700; font-size:1.1rem; }
.brand-name { font-size:1.35rem; font-weight:700; letter-spacing:-.04em; }
.hero { padding:2.2rem 0 1.25rem; }
.eyebrow { color:var(--accent); font-weight:700; letter-spacing:.15em; text-transform:uppercase; font-size:.68rem; }
h1, h2, h3 { font-family:'Playfair Display',serif !important; color:var(--ink) !important; letter-spacing:-.02em; }
.hero h1 { font-size:3.35rem; line-height:1.04; margin:.45rem 0 .75rem; }
.hero p { color:var(--muted); font-size:1.05rem; line-height:1.6; max-width:660px; }
.status { display:inline-flex; align-items:center; gap:.45rem; background:#e3f6ed; color:#216b55;
  border:1px solid #c9e9db; border-radius:999px; padding:.42rem .8rem; font-size:.76rem; font-weight:700; }
.dot { width:8px; height:8px; border-radius:50%; background:#25a875; display:inline-block; box-shadow:0 0 0 4px #c8efdf; }
.welcome { background:linear-gradient(135deg,#e3f6ed,#f0f8ee); border:1px solid #cce6d9;
  border-radius:18px; padding:1.1rem 1.25rem; margin-bottom:1.2rem; color:#36594f; line-height:1.55; }
.welcome strong { color:var(--ink); }
.kg-tag { display:inline-block; background:rgba(22,122,104,.12); color:#0e5a4a; border:1px solid rgba(22,122,104,.2);
  border-radius:999px; padding:.2rem .55rem; font-size:.67rem; font-weight:600; margin:.15rem .1rem; }
.metric { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem;
  box-shadow:0 8px 24px rgba(20,55,45,.04); }
.metric-label { color:var(--muted); font-size:.68rem; text-transform:uppercase; letter-spacing:.1em; font-weight:700; }
.metric-value { color:var(--ink); font-size:1.3rem; font-weight:700; margin-top:.25rem; }
.disclaimer { color:#a9c7bb; font-size:.73rem; line-height:1.55; margin-top:1.4rem; }
[data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:18px; padding:.9rem 1rem; margin:.7rem 0;
  background:#fff; box-shadow:0 5px 18px rgba(20,55,45,.035); }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { color:#173a32 !important; font-size:.96rem; line-height:1.65; }
div[data-testid="stChatInput"] { padding:1rem 0 1.3rem; }
div[data-testid="stChatInput"] > div { border:1px solid #bfd8cc; border-radius:18px; background:#fff; box-shadow:0 8px 25px rgba(20,55,45,.08); }
div[data-testid="stChatInput"] textarea { color:#173a32 !important; }
</style>
""", unsafe_allow_html=True)

WELCOME_MSG = (
    "Ready. Enter what the client says (or your own observations / questions). "
    "For each entry I'll return observations, working hypotheses, risk indicators, "
    "suggested interventions and possible next questions, grounded in the knowledge graph."
)

# ── Session state initialisation ────────────────────────────────────────────────
# Bump this when EnhancedChatbot API changes to force session_state recreation.
BOT_VERSION = "2"   # streaming support added

def _new_bot() -> EnhancedChatbot:
    import uuid
    adapter = make_adapter(BACKEND, model=MODEL, api_key=API_KEY, host=OLLAMA_HOST)
    sid = f"session_{uuid.uuid4().hex[:8]}"
    return EnhancedChatbot(llm=adapter, session_id=sid, log_dir=LOG_DIR)

# Force recreation if the class has been updated
if (
    "bot" not in st.session_state
    or st.session_state.get("_bot_version") != BOT_VERSION
    or not hasattr(st.session_state.bot, "generate_response_stream")
):
    st.session_state.bot = _new_bot()
    st.session_state._bot_version = BOT_VERSION

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": WELCOME_MSG
    }]
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {
        "emotion": "neutral", "behaviour": "Conversational_Act",
        "disorders": [], "techniques": [], "confidence": 0.0,
        "crisis_flag": "false",
    }
if "last_response" not in st.session_state:
    st.session_state.last_response = st.session_state.messages[0]["content"]


def submit_message(text: str) -> None:
    """Non-streaming fallback (used by demo prompt buttons in sidebar)."""
    text = text.strip()
    if not text:
        return
    response, meta = st.session_state.bot.generate_response(text)
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "crisis": meta.get("crisis_flag") == "true",
    })
    st.session_state.last_meta     = meta
    st.session_state.last_response = response


# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">✦</div><div class="brand-name">Mend</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Clinician assistant · GraphRAG grounded")
    st.markdown("---")

    # Affect + Intent
    st.markdown("### Client signals")
    meta = st.session_state.last_meta
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Affect", _pretty(meta.get("emotion", "neutral")))
    with col2:
        st.metric("Intent", _pretty(meta.get("behaviour", "—")))

    # Everything else the KG behaviour model picked up (behaviours, thinking patterns, symptoms)
    signals = meta.get("signals") or {}
    for key, title in (("behaviours", "Behaviours"), ("cognitive_patterns", "Thinking patterns"), ("symptoms", "Symptoms in the text")):
        items = signals.get(key) or []
        if items:
            st.caption(title)
            st.markdown(" ".join(f'<span class="kg-tag">{_pretty(n)} {p:.0%}</span>' for n, p in items[:6]),
                        unsafe_allow_html=True)

    # KG confidence
    conf = meta.get("confidence", 0.0)
    st.metric("KG Confidence", f"{conf:.0%}" if conf else "—")

    # Clinical tags from KG
    disorders  = meta.get("disorders", [])
    techniques = meta.get("techniques", [])
    contexts   = meta.get("contexts", [])

    if disorders or techniques or contexts:
        st.markdown("### KG clinical context")
        if disorders:
            st.caption("Candidate presentations")
            tags_html = " ".join(f'<span class="kg-tag">{d}</span>' for d in disorders)
            st.markdown(tags_html, unsafe_allow_html=True)
        if techniques:
            st.caption("Suggested techniques")
            tags_html = " ".join(f'<span class="kg-tag">{t}</span>' for t in techniques)
            st.markdown(tags_html, unsafe_allow_html=True)
        if contexts:
            st.caption("Contextual factors")
            tags_html = " ".join(f'<span class="kg-tag">{c}</span>' for c in contexts)
            st.markdown(tags_html, unsafe_allow_html=True)

    # ── Overall assessment (aggregated across the whole session) ────────────────
    profile = meta.get("session_profile") or {}
    if profile.get("overall_theme") or profile.get("dominant_emotions"):
        st.markdown("### Session formulation so far")
        if profile.get("overall_theme"):
            st.caption("Working theme")
            st.markdown(f'<span class="kg-tag">{profile["overall_theme"]}</span>',
                        unsafe_allow_html=True)
        if profile.get("dominant_emotions"):
            st.caption("Emotional profile")
            emo_html = " ".join(
                f'<span class="kg-tag">{_pretty(e)}</span>'
                for e in profile["dominant_emotions"][:3])
            st.markdown(emo_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Example client statements")
    demo_prompts = [
        "I've been feeling overwhelmed at work and can't switch off.",
        "I keep telling myself I'm a complete failure at everything.",
        "I went for a walk today and it actually helped a little.",
    ]
    for prompt in demo_prompts:
        if st.button(prompt, use_container_width=True):
            submit_message(prompt)
            st.rerun()

    st.markdown("---")
    if st.button("Start new client session", use_container_width=True):
        st.session_state.bot = _new_bot()
        st.session_state.messages = [{
            "role": "assistant",
            "content": WELCOME_MSG
        }]
        st.session_state.last_meta = {
            "emotion": "neutral", "behaviour": "Conversational_Act",
            "disorders": [], "techniques": [], "confidence": 0.0, "crisis_flag": "false",
        }
        st.session_state.last_response = st.session_state.messages[0]["content"]
        st.rerun()

    export = st.session_state.bot.export_chat_session()
    st.download_button(
        "Download session (EHR export)",
        data=json.dumps(export, indent=2, ensure_ascii=False),
        file_name=f"mend-session-{export.get('session_id','')}.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown(
        '<div class="disclaimer">Mend is a decision-support tool for qualified psychologists. '
        'Its output is AI-generated, grounded in a clinical knowledge graph, and is not a '
        'diagnosis. Clinical judgement and responsibility remain with the psychologist. '
        'Session logs are stored locally for clinical review only.</div>',
        unsafe_allow_html=True,
    )


# ── MAIN PANEL ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero"><div class="eyebrow">Clinician Assistant · GraphRAG grounded</div>'
    '<h1>Your in-session<br>clinical co-pilot.</h1>'
    '<p>Enter what your client says. Mend returns observations, working hypotheses, risk '
    'flags and evidence-based next steps — grounded in a clinical knowledge graph of '
    '123,926 real therapy sessions. For the psychologist only; not client-facing.</p>'
    '<span class="status"><span class="dot"></span>Clinician view · local processing</span></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.7, 1], gap="large")

with left:
    st.markdown(
        '<div class="welcome"><strong>How to use</strong><br>'
        'Type or dictate the client\'s statement as they said it. You can also add your own '
        'notes (e.g. "Note: flat affect, poor eye contact") or ask a direct question '
        '(e.g. "What differentials should I rule out?").</div>',
        unsafe_allow_html=True,
    )
    # ── Render all existing messages ─────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            avatar = "🩺" if message["role"] == "assistant" else "🗒️"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                if message.get("crisis"):
                    st.error("Risk indicators detected — conduct a structured risk assessment before continuing.")

    # ── Chat input & Audio input ─────────────────────────────────────────────────
    user_prompt = st.chat_input("Enter client statement, your notes, or a question...")
    audio_val = st.audio_input("Or dictate / record the client's statement...")

    # A recording stays in audio_val across reruns: only treat it as input once, and let typed text win.
    audio_key = hash(audio_val.getvalue()) if audio_val else None
    new_audio = audio_key is not None and audio_key != st.session_state.get("_last_audio")

    if user_prompt or new_audio:
        final_text = ""
        is_audio = False

        if new_audio and not user_prompt:
            st.session_state["_last_audio"] = audio_key
            is_audio = True
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_val.getvalue())
                tmp_path = f.name
            
            with chat_container:
                with st.chat_message("user", avatar="🗒️"):
                    with st.spinner("Transcribing audio..."):
                        transcriber = get_transcriber()
                        trans_res = transcriber.transcribe(tmp_path)


                        final_text = trans_res.get("text", "").strip()
            os.remove(tmp_path)
        elif user_prompt:
            final_text = user_prompt.strip()

        if final_text:
            # 1. Show user message in the left column container IMMEDIATELY
            with chat_container:
                if not is_audio:
                    with st.chat_message("user", avatar="🗒️"):
                        st.markdown(final_text)
                else:
                    with st.chat_message("user", avatar="🗒️"):
                        st.markdown(f"*(🎙️ Transcribed)*: {final_text}")

            # 2. Scrub PII
            scrubber = get_pii_scrubber()
            scrubbed_text, audit = scrubber.scrub_text(final_text)   # scrub() returns a ScrubbedText object; scrub_text() the (text, audit) pair

            # Inform user if PII was redacted
            if len(audit) > 0:
                with chat_container:
                    st.info(f"🔒 **Privacy Lock**: Redacted {len(audit)} client identifier(s) before processing.")

            st.session_state.messages.append({"role": "user", "content": scrubbed_text if len(audit) > 0 else final_text})

            # 3. Stream assistant reply token by token in the left column
            collected_tokens: list[tuple] = []

            with chat_container:
                with st.chat_message("assistant", avatar="🩺"):
                    def _token_generator():
                        for event in st.session_state.bot.generate_response_stream(scrubbed_text):
                            if event["type"] == "meta":
                                collected_tokens.append(("\x00META\x00", event["data"]))
                            elif event["type"] == "token":
                                collected_tokens.append(("TOKEN", event["data"]))
                                yield event["data"]

                    response_text = st.write_stream(_token_generator())

                    final_meta = {}
                    tokens_only: list[str] = []
                    for kind, val in collected_tokens:
                        if kind == "\x00META\x00":
                            final_meta = val
                        else:
                            tokens_only.append(val)

                    if final_meta.get("crisis_flag") == "true":
                        st.error("Risk indicators detected — conduct a structured risk assessment before continuing.")

            # 4. Commit to session state after streaming completes
            full_response = response_text if isinstance(response_text, str) else "".join(tokens_only)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "crisis": final_meta.get("crisis_flag") == "true",
            })
            st.session_state.last_meta     = final_meta
            st.session_state.last_response = full_response
            st.rerun()

with right:
    st.markdown("### How this works")
    st.markdown(
        "Each client statement is analysed for emotion and behaviour, then matched against a "
        "**6-layer Mental Health Knowledge Graph** (123,926 real therapy sessions). Relevant "
        "disorders, symptoms and techniques are retrieved and summarised into a clinical brief "
        "for you to accept, adapt or ignore."
    )
    turns = max(0, (len(st.session_state.messages) - 1) // 2)
    st.markdown(
        f'<div class="metric"><div class="metric-label">Entries this session</div>'
        f'<div class="metric-value">{turns}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown("### Latest brief")
    st.markdown(
        f'<div class="welcome"><strong>Mend · clinical brief</strong><br><br>{st.session_state.last_response}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Safety")
    st.info(
        "Crisis language in a client statement raises an immediate risk alert with "
        "recommended assessment steps. All sessions are logged locally for clinical review."
    )
    st.markdown("#### LLM backend")
    bot_meta = f"`{BACKEND} / {MODEL}`"
    st.code(bot_meta, language=None)
