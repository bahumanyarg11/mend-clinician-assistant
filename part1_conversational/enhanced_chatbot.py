"""
enhanced_chatbot.py  (Part 1)
==============================
Upgraded therapy chatbot where the LLM IS the therapist.

Pipeline per turn:
  1. Emotion + Behaviour detection (Part 1 models / keyword fallback)
  2. GraphRAG query  → clinical context from MLKG
  3. LLM call       → therapist response grounded in KG context
  4. Session log    → for later EHR review

The rule-based template engine in chatbot_demo.py is fully replaced by this.
app_integrated.py swaps in EnhancedChatbot with ZERO changes to the UI.

LLM is model-agnostic via the adapter from finalmlkg/api/therapy_assistant.py:
  from enhanced_chatbot import EnhancedChatbot, make_adapter
  bot = EnhancedChatbot(llm=make_adapter("ollama", model="llama3.2"))
  response, meta = bot.generate_response("I haven't slept in weeks")
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Path resolution ─────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
_KG_DIR   = os.path.join(_ROOT_DIR, "part2_graphrag")
for p in [_ROOT_DIR, _KG_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from graphrag_client import GraphRAGClient, build_request
from api.therapy_assistant import (
    BaseLLMAdapter, OllamaAdapter, OpenAIAdapter, GeminiAdapter, AnthropicAdapter
)
from api.session_memory import SessionMemory

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EnhancedChatbot")


# ═══════════════════════════════════════════════════════════════════════════════
# EMOTION & BEHAVIOUR DETECTION  (keyword fallback — real models optional)
# ═══════════════════════════════════════════════════════════════════════════════

EMOTION_KEYWORDS: dict[str, list[str]] = {
    "anxiety":          ["anxious","worried","panic","nervous","dread","terrified","stressed","restless","overwhelmed","tense"],
    "sadness":          ["sad","depressed","hopeless","crying","unhappy","lonely","miserable","down","empty","numb"],
    "anger":            ["angry","furious","mad","irritated","pissed","hate","resentful","rage","bitter"],
    "frustration":      ["frustrated","stuck","annoyed","exhausted","tired of","fed up"],
    "shame_guilt":      ["guilty","ashamed","my fault","embarrassed","worthless","blame myself","i ruined"],
    "fear":             ["scared","fearful","frightened","paralyzed","afraid","terrifying"],
    "grief":            ["lost","passed away","mourning","miss them","grieving","death","bereavement"],
    "relief":           ["relieved","calmer","better now","glad that's over","feeling eased"],
    "positive_progress":["better","improving","hopeful","managed to","accomplished","proud","made progress"],
}

BEHAVIOUR_KEYWORDS: dict[str, list[str]] = {
    "Social_Withdrawal":  ["avoiding","isolating","shut inside","don't want to see","pulling away","withdrawing"],
    "Rumination":         ["can't stop thinking","keeps replaying","over and over","obsessing","stuck in my head","overthinking"],
    "Avoidance":          ["procrastinating","putting off","ignoring","running away from","not facing"],
    "Negative_Self_Talk": ["i am stupid","i'm a failure","nobody likes me","i always mess up","i ruin everything","i'm worthless"],
    "Emotional_Expression":["i feel","i am experiencing","it hurts","i sense","my emotions"],
    "Help_Seeking":       ["what should i do","can you help","how do i cope","i need advice","give me strategies"],
    "Active_Coping":      ["i tried","i went for a walk","i practiced","i spoke to","journaled","exercised"],
    "Conversational_Act": ["hello","hi","good morning","thank you","bye","okay","yes"],
}

def detect_emotion_keyword(text: str) -> tuple[str, dict[str, float]]:
    """Keyword-based emotion detection. Returns (top_label, distribution_dict)."""
    lower = text.lower()
    scores: dict[str, float] = {}
    for emo, kws in EMOTION_KEYWORDS.items():
        matches = sum(1 for kw in kws if kw in lower)
        if matches:
            scores[emo] = matches / len(kws)
    if not scores:
        return "neutral", {"neutral": 1.0}
    top = max(scores, key=scores.get)
    return top, scores

def detect_behaviour_keyword(text: str) -> str:
    """Keyword-based behaviour detection. Returns top label."""
    lower = text.lower()
    for beh, kws in BEHAVIOUR_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            return beh
    return "Emotional_Expression"


# ── Singleton ModelServer ──────────────────────────────────────────────────────
# Loaded once at startup, shared across all chatbot instances.
_MODEL_SERVER = None

def get_model_server():
    global _MODEL_SERVER
    if _MODEL_SERVER is None:
        try:
            from model_server import ModelServer
            logger.info("Loading emotion + behaviour models (first-time startup)...")
            _MODEL_SERVER = ModelServer()
            logger.info(f"Models ready: {_MODEL_SERVER.status()}")
        except Exception as e:
            logger.warning(f"ModelServer unavailable ({e}). Using keyword detection fallback.")
            _MODEL_SERVER = None
    return _MODEL_SERVER


# ═══════════════════════════════════════════════════════════════════════════════
# THERAPIST SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

CLINICIAN_ASSISTANT_SYSTEM_PROMPT = """\
You are Mend, a clinical decision-support assistant for a licensed psychologist. The
psychologist is conducting the session; the client NEVER sees your output. Each message you
receive is the psychologist's input — usually what the client just said, sometimes the
psychologist's own notes or a direct question to you. Your job is to help the psychologist
think, not to talk to the client.

CLINICAL CONTEXT (retrieved from the knowledge graph for this turn — use it as evidence;
you MAY name disorders, symptoms and techniques, since the reader is a clinician):
{kg_context}

RESPOND AS A CONCISE CLINICAL BRIEF using these Markdown headings (skip a section only if
there is genuinely nothing to say):

**Observations** — affect, themes, cognitive patterns or distortions, behaviours and notable
language in what the client said. Quote short phrases where useful.

**Working hypotheses** — candidate presentations / differentials suggested by the material
and the knowledge graph, with the evidence for each and what is still missing to confirm or
rule them out. Frame as hypotheses for the clinician's judgement, never as a diagnosis.

**Risk** — any indicators of risk to self or others (state "No risk indicators noted" if
none). If any are present, put this section FIRST and recommend a structured risk
assessment.

**Suggested interventions** — 1–3 evidence-based options (CBT, DBT, ACT, MI, etc.) suited
to this moment and session stage, each with a one-line rationale and how to introduce it.

**Possible next questions / responses** — 2–3 options the psychologist could use next,
written as example wording they can adapt (e.g. a reflection, an open question, an
assessment probe).

**Notes for the record** — one or two lines suitable for session notes (optional).

GUIDELINES:
- Be specific to THIS client and build on earlier turns; track how the picture is evolving.
- Be brief and scannable — bullets are fine. No filler, no reassurance aimed at the client.
- Separate what the client actually said from your inferences, and state uncertainty.
- Do not recommend specific medications or dosages; if pharmacological questions arise,
  note that a prescriber/psychiatrist review may be indicated.
- If the psychologist asks you a direct question, answer it directly first, then add the
  brief only if relevant.
- The final clinical judgement always rests with the psychologist.
"""

# Backwards-compatible alias
THERAPIST_SYSTEM_PROMPT = CLINICIAN_ASSISTANT_SYSTEM_PROMPT

CLINICIAN_CRISIS_ALERT = """\
**⚠️ RISK ALERT — crisis indicators detected in the client's statement.**

**Recommended immediate steps for the psychologist:**
- Pause the current agenda and conduct a structured suicide/self-harm risk assessment
  (ideation, intent, plan, means, timeline, protective factors, history).
- Ask directly and calmly about safety; do not leave the client alone if risk is imminent.
- Collaboratively build or review a safety plan and identify trusted supports.
- If risk is imminent, follow your service's emergency protocol (emergency services: 112;
  Tele-MANAS: 14416).
- Document the assessment, decisions and follow-up plan.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# THERAPEUTIC SCAFFOLDING (phase policy, plain-language, micro-interventions)
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical KG disorder node → plain, non-diagnostic phrase for grounding.
DISORDER_PLAIN = {
    "SingleEpisodeDepressiveDisorder": "low mood / depression",
    "RecurrentDepressiveDisorder": "recurring low mood / depression",
    "DysthymicDisorder": "persistent low mood",
    "MixedDepressiveAndAnxietyDisorder": "mixed low mood and anxiety",
    "GeneralizedAnxietyDisorder": "ongoing worry / anxiety",
    "PanicDisorder": "panic and sudden fear",
    "SocialAnxietyDisorder": "social anxiety",
    "Agoraphobia": "fear of situations that feel hard to escape",
    "SpecificPhobia": "a specific fear",
    "SeparationAnxietyDisorder": "anxiety about separation",
    "PostTraumaticStressDisorder": "trauma-related distress",
    "ComplexPTSD": "complex trauma-related distress",
    "ProlongedGriefDisorder": "intense, prolonged grief",
    "AdjustmentDisorder": "difficulty adjusting to a life stressor",
    "ObsessiveCompulsiveDisorder": "obsessive thoughts and compulsions",
    "BodyDysmorphicDisorder": "distress about appearance",
    "InsomniaDisorder": "sleep difficulty",
    "BipolarTypeIDisorder": "big mood swings",
    "BipolarTypeIIDisorder": "mood swings with low periods",
    "AnorexiaNervosa": "a difficult relationship with eating",
    "BulimiaNervosa": "a difficult relationship with eating",
    "BingeEatingDisorder": "a difficult relationship with eating",
    "AlcoholDependence": "difficulty with alcohol use",
    "AttentionDeficitHyperactivityDisorder": "focus and attention difficulty",
}

# KG technique node (or keyword) → a short skill the therapist can gently guide,
# in plain language. Never name the technique to the patient.
TECHNIQUE_SCRIPTS = {
    "grounding": "Offer a brief grounding moment: gently invite them to notice 5 things they can see, 4 they can hear, 3 they can feel — to come back into the present.",
    "mindfulness": "Invite a small mindful pause: notice the breath or the feeling of their feet on the floor, without trying to change anything.",
    "breathing": "Guide slow paced breathing: in for a count of 4, out for a count of 6, a few gentle rounds, noticing the body settle.",
    "relaxation": "Guide slow paced breathing or a gentle release of tension in the shoulders and jaw.",
    "behaviouralactivation": "Help them choose ONE tiny, doable, meaningful or pleasant activity for today, and make it concrete — when, where, how small.",
    "behavioralactivation": "Help them choose ONE tiny, doable, meaningful or pleasant activity for today, and make it concrete — when, where, how small.",
    "cognitiverestructuring": "Gently explore one harsh thought: name it, look together at what supports it and what doesn't, then find a kinder, more balanced way to see it.",
    "thoughtrecord": "Invite them to name the specific harsh thought, then gently weigh the evidence for and against it and a fairer alternative.",
    "journalling": "Suggest writing the worry or thought down, so it can be looked at from the outside rather than swirling inside.",
    "problemsolving": "Break the problem into one small, controllable next step, and focus only on that first step.",
    "exposure": "Explore the smallest, least frightening version of the feared situation they could gently approach first.",
    "selfcompassion": "Invite them to speak to themselves as they would to a good friend facing the same thing — with the same warmth.",
    "distresstolerance": "Offer a quick way to lower the intensity first — cold water on the face, or paced breathing — before trying to solve anything.",
    "emotionregulation": "Help them name the feeling and its intensity, then choose one small soothing action before deciding what to do next.",
}

def _pick_technique_script(techniques: list[str]) -> Optional[str]:
    """Return a micro-intervention script matching the first recognised technique."""
    for tech in techniques or []:
        key = re.sub(r"[^a-z]", "", tech.lower())
        for name, script in TECHNIQUE_SCRIPTS.items():
            if name in key or key in name:
                return script
    return None


def _phase_and_goal(turn: int, asked_for_help: bool) -> tuple[str, str]:
    """
    Session arc: early turns build understanding; later turns actively help.
    If the person explicitly asks for help, move to helping sooner.
    """
    if asked_for_help and turn >= 2:
        stage = "ACTIVE HELP"
    elif turn <= 2:
        stage = "OPENING & UNDERSTANDING"
    elif turn <= 4:
        stage = "FOCUSING"
    else:
        stage = "ACTIVE HELP"

    goals = {
        "OPENING & UNDERSTANDING": (
            "Early session: prioritise assessment and rapport. Suggest exploratory questions "
            "and reflections that help the psychologist understand the presenting problem, "
            "history and context. Interventions should be light at this stage."
        ),
        "FOCUSING": (
            "Formulation stage: summarise the emerging pattern (triggers, thoughts, feelings, "
            "behaviours, maintaining factors), note strengths, and suggest how the psychologist "
            "might share a brief formulation and move toward a focus for intervention."
        ),
        "ACTIVE HELP": (
            "Intervention stage: recommend concrete, evidence-based interventions the "
            "psychologist can deliver now, with step-by-step guidance, plus a possible "
            "between-session task and what to review next session."
        ),
    }
    return stage, goals[stage]


_HELP_REQUEST_RE = re.compile(
    r"\b(what (can|should|do) i|what do i do|how (can|do) i|help me|any (advice|tips|ideas)|"
    r"what now|where (do|should) i start|how to|fix this|get better|feel better)\b", re.I)

# Signals that the person is in an ACUTE moment RIGHT NOW (panic/anxiety attack,
# acute distress) — as opposed to calmly recounting their experience.
_ACUTE_RE = re.compile(
    r"\b(right now|can'?t breathe|can'?t catch my breath|cannot breathe|hyperventilat|"
    r"heart is (pounding|racing|beating so fast)|chest (is |feels )?(tight|tightening|crushing)|"
    r"panic attack|anxiety attack|having an attack|it'?s happening (again|now)|"
    r"i'?m panicking|i am panicking|freaking out|can'?t calm|i'?m shaking|i am shaking|"
    r"feel like i'?m dying|feel like i'?m going to die|about to pass out|going to faint|"
    r"i'?m so scared right now|i can'?t stop crying)\b", re.I)

_STABILISE_GOAL = (
    "IMPORTANT: the client appears to be in ACUTE panic or anxiety RIGHT NOW. Advise the "
    "psychologist to pause exploration and stabilise first. Lead with short, calm wording "
    "the psychologist can use, and a simple grounding or paced-breathing script to guide the "
    "client through step by step. Keep the brief short; formulation can wait."
)

def _is_acute(user_input: str, emotion_label: str, emotion_dist: dict) -> bool:
    if _ACUTE_RE.search(user_input or ""):
        return True
    el = (emotion_label or "").lower()
    if el in ("fear", "panic", "nervousness"):
        probs = {k.lower(): v for k, v in emotion_dist.items()} if isinstance(emotion_dist, dict) else {}   # KG model uses "Fear"
        peak = max([probs.get(k, 0.0) for k in ("fear", "panic", "nervousness", el)] or [0.0])
        if peak >= 0.75 and re.search(
                r"\b(now|can'?t|racing|pounding|breathe|shaking|scared|panic)\b",
                user_input or "", re.I):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION LOG (for EHR review)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SessionTurn:
    turn:           int
    patient_text:   str
    response:       str
    emotion:        str
    behaviour:      str
    disorders:      list[str]
    techniques:     list[str]
    was_crisis:     bool
    timestamp:      str = field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedChatbot:
    """
    Full therapy chatbot where the LLM IS the therapist.
    Drop-in replacement for PsychotherapyChatbot — same generate_response() signature.

    Parameters
    ----------
    llm              : Any BaseLLMAdapter (Ollama, OpenAI, Gemini, Anthropic)
    session_id       : Unique patient session identifier
    emotion_model    : Optional callable(text) → (label, distribution_dict)
                       If None, falls back to keyword detection
    behaviour_model  : Optional callable(text) → label
                       If None, falls back to keyword detection
    log_dir          : Directory to save session logs (for EHR review). None = no save
    """

    def __init__(
        self,
        llm:            BaseLLMAdapter,
        session_id:     str = "session_001",
        emotion_model   = None,
        behaviour_model = None,
        log_dir:        Optional[str] = None,
    ):
        self.llm            = llm
        self.session_id     = session_id
        self._emotion_model  = emotion_model
        self._behaviour_model= behaviour_model
        self._kg_client     = GraphRAGClient()
        self._memory        = SessionMemory(max_turns=8, session_id=session_id)
        self._turn          = 0
        self._log: list[SessionTurn] = []
        self._log_dir       = log_dir
        # ── Session profile (aggregated across the whole session) ────────────────
        self._disorder_tally: Counter = Counter()   # confidence-weighted disorder votes
        self._emotion_tally:  defaultdict = defaultdict(float)  # summed emotion probs
        self._asked_for_help = False
        logger.info(f"EnhancedChatbot ready | LLM: {llm.model_name} | session: {session_id}")

    # ── Session profile + prompt assembly ───────────────────────────────────────
    def _update_profile(self, emotion_label, emotion_dist, kg_response) -> None:
        """Aggregate disorder + emotion signals across the whole session."""
        conf = float(getattr(kg_response, "confidence", 0.0) or 0.0) or 1.0
        for rank, d in enumerate(kg_response.disorders[:3]):
            self._disorder_tally[d] += conf * (1.0 / (rank + 1))
        if isinstance(emotion_dist, dict) and emotion_dist:
            for e, p in emotion_dist.items():
                self._emotion_tally[e] += float(p)
        elif emotion_label:
            self._emotion_tally[emotion_label] += 1.0

    def session_profile(self) -> dict:
        """Overall picture so far (dominant disorder(s) + emotional profile)."""
        top_dis = self._disorder_tally.most_common(2)
        top_emo = sorted(self._emotion_tally.items(), key=lambda x: -x[1])[:3]
        dom = top_dis[0][0] if top_dis else None
        return {
            "dominant_disorders": [d for d, _ in top_dis],
            "dominant_emotions":  [e for e, _ in top_emo],
            "overall_theme":      DISORDER_PLAIN.get(dom, dom),
            "turns":              self._turn,
        }

    def _session_profile_text(self) -> str:
        prof = self.session_profile()
        if not prof["dominant_disorders"] and not prof["dominant_emotions"]:
            return ""
        theme = prof["overall_theme"] or "their main concern"
        emo = ", ".join(prof["dominant_emotions"][:3]) or "mixed feelings"
        return ("SESSION FOCUS (stable across this whole session — keep grounding here, do NOT "
                f"recite it): the working theme is {theme}; the emotional tone so far is {emo}. "
                "Stay consistent with this focus instead of switching topic each turn.")

    def _fewshot_text(self, kg_response) -> str:
        """1-2 tier-1 GOLD real-therapist replies as style/approach exemplars."""
        exs = []
        for rs in getattr(kg_response, "retrieved_sessions", []) or []:
            th = (rs.therapist_text or "").strip()
            pt = (rs.patient_text or "").strip()
            if getattr(rs, "tier", 0) == 1 and len(th.split()) >= 6 and rs.similarity_score >= 0.35:
                exs.append((pt, th))
            if len(exs) >= 2:
                break
        if not exs:
            return ""
        lines = ["Similar moments from real therapy sessions (example approaches the "
                 "psychologist could adapt):"]
        for pt, th in exs:
            if pt:
                lines.append(f'  Client: "{pt[:160]}"')
            lines.append(f'  Therapist: "{th[:280]}"')
        return "\n".join(lines)

    def _build_system_prompt(self, kg_response, user_input: str,
                             emotion_label: str = "", emotion_dist=None) -> str:
        """Assemble the full system prompt. Tone adapts to the person's STATE:
        - ACUTE (panic/anxiety attack right now) → stabilise & ground, don't explore.
        - Otherwise → phase-based: explore/ask early, actively help later."""
        parts = [THERAPIST_SYSTEM_PROMPT.format(kg_context=kg_response.to_prompt_context())]
        prof = self._session_profile_text()
        if prof:
            parts.append(prof)

        acute = _is_acute(user_input, emotion_label, emotion_dist or {})
        if acute:
            # STABILISE mode overrides the normal phase — calm them in the moment.
            parts.append(f"CURRENT PHASE — STABILISE (acute distress, turn {self._turn}).\n"
                         + _STABILISE_GOAL)
            script = _pick_technique_script(["breathing", "grounding"])
            parts.append("Suggested stabilisation script for the psychologist: " + script)
            # No exploratory few-shot in an acute moment — keep it short and grounding.
            return "\n\n".join(parts)

        phase, goal = _phase_and_goal(self._turn, self._asked_for_help)
        parts.append(f"CURRENT PHASE — {phase} (turn {self._turn}).\n{goal}")
        if phase == "ACTIVE HELP" or (phase == "FOCUSING" and self._asked_for_help):
            script = _pick_technique_script(kg_response.recommended_techniques)
            if script:
                parts.append("A KG-matched intervention worth suggesting this turn: " + script)
        ex = self._fewshot_text(kg_response)
        if ex:
            parts.append(ex)
        return "\n\n".join(parts)

    # ── Public interface (same as PsychotherapyChatbot) ─────────────────────────
    def generate_response(self, user_input: str) -> tuple[str, dict]:
        """
        Main entry point. Called once per patient message.

        Returns
        -------
        (response_text, meta_dict)
        meta_dict keys: emotion, behaviour, crisis_flag, disorders, techniques, confidence
        """
        self._turn += 1
        user_input = user_input.strip()

        # ── Step 1: Detect emotion + behaviour via models ────────────────────────
        ms = get_model_server()
        if ms:
            emotion_label, emotion_dist = ms.predict_emotion(user_input)
            behaviour_label, _          = ms.predict_behaviour(user_input)
            symptoms                    = ms.predict_symptoms(user_input)
        elif self._emotion_model:
            emotion_label, emotion_dist = self._emotion_model(user_input)
            behaviour_label = (
                self._behaviour_model(user_input)
                if self._behaviour_model
                else detect_behaviour_keyword(user_input)
            )
            symptoms = []
        else:
            emotion_label, emotion_dist = detect_emotion_keyword(user_input)
            behaviour_label = detect_behaviour_keyword(user_input)
            symptoms = []

        # ── Step 2: GraphRAG KG query ────────────────────────────────────────────
        kg_request = build_request(
            patient_text  = user_input,
            emotion       = emotion_label,
            behaviour     = behaviour_label,
            symptoms      = symptoms,
            emotion_dist  = emotion_dist,
            session_id    = self.session_id,
            turn_number   = self._turn,
        )
        kg_response = self._kg_client.query(kg_request)

        # ── Step 3: Crisis short-circuit ────────────────────────────────────────
        if kg_response.was_crisis:
            meta = {
                "emotion": "crisis", "behaviour": "Help_Seeking",
                "crisis_flag": "true", "disorders": [], "techniques": [],
                "confidence": 1.0,
            }
            self._log_turn(user_input, CLINICIAN_CRISIS_ALERT, meta, kg_response)
            return CLINICIAN_CRISIS_ALERT, meta

        # ── Step 4: Build LLM prompt (profile + phase + micro-skill + few-shot) ──
        self._asked_for_help = self._asked_for_help or bool(_HELP_REQUEST_RE.search(user_input))
        self._update_profile(emotion_label, emotion_dist, kg_response)
        system_prompt   = self._build_system_prompt(kg_response, user_input,
                                                     emotion_label, emotion_dist)

        # Add patient message to memory
        self._memory.add_user(user_input)
        messages = self._memory.get_messages(system_prompt=system_prompt)

        # ── Step 5: LLM generates the therapist response ────────────────────────
        response_text = self.llm.generate(messages)

        # Store response in memory for next turn
        self._memory.add_assistant(response_text)

        # ── Step 6: Build metadata for GUI display ───────────────────────────────
        meta = {
            "emotion":      emotion_label,
            "behaviour":    behaviour_label,
            "crisis_flag":  "false",
            "disorders":    kg_response.disorders,
            "techniques":   kg_response.recommended_techniques,
            "confidence":   kg_response.confidence,
            "emotions_confirmed": kg_response.emotions_confirmed,
            "contexts":     kg_response.worsening_contexts,
            "session_profile": self.session_profile(),
            "signals":      ms.predict_signals(user_input) if ms else {},
        }

        # ── Step 7: Log for EHR ─────────────────────────────────────────────────
        self._log_turn(user_input, response_text, meta, kg_response)

        return response_text, meta

    def generate_response_stream(self, user_input: str):
        """
        Streaming variant of generate_response().

        Yields
        ------
        dict with one of two shapes:
          {"type": "meta",  "data": meta_dict}   — sent FIRST (before any tokens)
          {"type": "token", "data": token_str}   — one per LLM token

        Usage in Streamlit:
            collected = []
            for event in bot.generate_response_stream(text):
                if event["type"] == "meta":
                    meta = event["data"]
                elif event["type"] == "token":
                    collected.append(event["data"])
                    # write token to st.write_stream placeholder
            full_response = "".join(collected)
        """
        self._turn += 1
        user_input = user_input.strip()

        # ── Step 1: Emotion + behaviour ──────────────────────────────────────────
        ms = get_model_server()
        if ms:
            emotion_label, emotion_dist = ms.predict_emotion(user_input)
            behaviour_label, _          = ms.predict_behaviour(user_input)
            symptoms                    = ms.predict_symptoms(user_input)
        elif self._emotion_model:
            emotion_label, emotion_dist = self._emotion_model(user_input)
            behaviour_label = (
                self._behaviour_model(user_input)
                if self._behaviour_model
                else detect_behaviour_keyword(user_input)
            )
            symptoms = []
        else:
            emotion_label, emotion_dist = detect_emotion_keyword(user_input)
            behaviour_label = detect_behaviour_keyword(user_input)
            symptoms = []

        # ── Step 2: GraphRAG KG query ────────────────────────────────────────────
        kg_request = build_request(
            patient_text  = user_input,
            emotion       = emotion_label,
            behaviour     = behaviour_label,
            symptoms      = symptoms,
            emotion_dist  = emotion_dist,
            session_id    = self.session_id,
            turn_number   = self._turn,
        )
        kg_response = self._kg_client.query(kg_request)

        # ── Step 3: Crisis short-circuit — yield full text immediately ───────────
        if kg_response.was_crisis:
            meta = {
                "emotion": "crisis", "behaviour": "Help_Seeking",
                "crisis_flag": "true", "disorders": [], "techniques": [],
                "confidence": 1.0,
            }
            yield {"type": "meta", "data": meta}
            yield {"type": "token", "data": CLINICIAN_CRISIS_ALERT}
            self._log_turn(user_input, CLINICIAN_CRISIS_ALERT, meta, kg_response)
            return

        # ── Step 4: Build prompt (profile + phase + micro-skill + few-shot) ──────
        self._asked_for_help = self._asked_for_help or bool(_HELP_REQUEST_RE.search(user_input))
        self._update_profile(emotion_label, emotion_dist, kg_response)
        system_prompt   = self._build_system_prompt(kg_response, user_input,
                                                     emotion_label, emotion_dist)
        self._memory.add_user(user_input)
        messages = self._memory.get_messages(system_prompt=system_prompt)

        # ── Step 5: Emit meta BEFORE streaming starts ────────────────────────────
        meta = {
            "emotion":      emotion_label,
            "behaviour":    behaviour_label,
            "crisis_flag":  "false",
            "disorders":    kg_response.disorders,
            "techniques":   kg_response.recommended_techniques,
            "confidence":   kg_response.confidence,
            "emotions_confirmed": kg_response.emotions_confirmed,
            "contexts":     kg_response.worsening_contexts,
            "session_profile": self.session_profile(),
            "signals":      ms.predict_signals(user_input) if ms else {},
        }
        yield {"type": "meta", "data": meta}

        # ── Step 6: Stream tokens from LLM ──────────────────────────────────────
        collected: list[str] = []
        for token in self.llm.stream(messages):
            collected.append(token)
            yield {"type": "token", "data": token}

        # ── Step 7: Commit completed response to memory + EHR log ───────────────
        response_text = "".join(collected)
        self._memory.add_assistant(response_text)
        self._log_turn(user_input, response_text, meta, kg_response)

    # ── Session export (compatible with app.py's download_button) ───────────────
    def export_chat_session(self) -> dict:
        """Export in ChatML format compatible with original app.py."""
        return {
            "session_id": self.session_id,
            "turn_count": self._turn,
            "messages": self._memory.get_messages(),
            "session_log": [
                {
                    "turn":       t.turn,
                    "patient":    t.patient_text,
                    "therapist":  t.response,
                    "emotion":    t.emotion,
                    "behaviour":  t.behaviour,
                    "disorders":  t.disorders,
                    "techniques": t.techniques,
                    "was_crisis": t.was_crisis,
                    "timestamp":  t.timestamp,
                }
                for t in self._log
            ],
        }

    def close(self):
        self._save_log()
        self._kg_client.close()

    # ── Internals ───────────────────────────────────────────────────────────────
    def _log_turn(self, patient_text: str, response: str, meta: dict, kg_resp) -> None:
        self._log.append(SessionTurn(
            turn=self._turn,
            patient_text=patient_text,
            response=response,
            emotion=meta.get("emotion", "neutral"),
            behaviour=meta.get("behaviour", ""),
            disorders=meta.get("disorders", []),
            techniques=meta.get("techniques", []),
            was_crisis=meta.get("crisis_flag") == "true",
        ))
        self._save_log()

    def _save_log(self) -> None:
        if not self._log_dir:
            return
        os.makedirs(self._log_dir, exist_ok=True)
        path = os.path.join(self._log_dir, f"{self.session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export_chat_session(), f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTER FACTORY (one-line LLM selection)
# ═══════════════════════════════════════════════════════════════════════════════

def make_adapter(backend: str = "ollama", **kwargs) -> BaseLLMAdapter:
    """
    Factory for LLM adapters. Called from app_integrated.py or CLI.

    make_adapter("ollama", model="llama3.2")
    make_adapter("openai",     api_key="sk-...", model="gpt-4o-mini")
    make_adapter("gemini",     api_key="...",    model="gemini-1.5-flash")
    make_adapter("anthropic",  api_key="...",    model="<model-id>")
    """
    b = backend.lower()
    if b == "ollama":
        return OllamaAdapter(model=kwargs.get("model", "llama3.2"),
                             host=kwargs.get("host", "http://localhost:11434"))
    if b == "openai":
        return OpenAIAdapter(api_key=kwargs["api_key"],
                             model=kwargs.get("model", "gpt-4o-mini"))
    if b in ("gemini", "google"):
        return GeminiAdapter(api_key=kwargs["api_key"],
                             model=kwargs.get("model", "gemini-1.5-flash"))
    if b == "anthropic":
        return AnthropicAdapter(api_key=kwargs["api_key"],
                                model=kwargs.get("model", ""))
    raise ValueError(f"Unknown backend: '{backend}'. Use: ollama | openai | gemini | anthropic")
