# -*- coding: utf-8 -*-
"""
Central configuration for the Therapy project (Parts 1 & 2).
Single source of truth for connection details, model paths, LLM backend and the
tunable thresholds that were previously scattered across pipeline/api files.

Everything reads from environment variables first (see .env.example), with safe
local-dev defaults. Import it anywhere:

    from config.settings import NEO4J, MODELS, LLM, THRESHOLDS, DATASETS
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # .../Therapy
PART1 = ROOT / "part1_conversational"
PART2 = ROOT / "part2_graphrag"

# ── Neo4j (Part 2 GraphRAG) ───────────────────────────────────────────────────
NEO4J = {
    "uri":      os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    "user":     os.environ.get("NEO4J_USER", "neo4j"),
    "password": os.environ.get("NEO4J_PASSWORD", ""),   # set in env; never target system/neo4j
    "database": os.environ.get("NEO4J_DATABASE", "therapy"),
}

# ── ML models (Part 1: emotion + behaviour) ───────────────────────────────────
# kg_emotion / kg_behaviour = KG-aligned retrain (DEFAULT in model_server.py; MEND_MODELS=legacy reverts).
# emotion_model_1 = 11-class RoBERTa (legacy fallback). emotion_model = alternate config.
MODELS = {
    "kg_emotion":       PART1 / "models/kg_emotion/model",       # 35 EmotionalState labels
    "kg_behaviour":     PART1 / "models/kg_behaviour/model",     # 214 Behaviour / CognitivePattern / Symptom labels
    "emotion_active":   "emotion_model_1",
    "emotion_model_1":  PART1 / "models/emotion_model_1/emotion_model_1/model",
    "emotion_model":    PART1 / "models/emotion_model/emotion_model/models",   # alternate (weights not extracted)
    "behaviour":        PART1 / "models/behaviour_model/behaviour_model/models/distilbert_behaviour_model",
    "sentence_encoder": os.environ.get("SENTENCE_MODEL", "all-MiniLM-L6-v2"),  # GraphRAG semantic search
    "whisper_size":     os.environ.get("WHISPER_MODEL", "base"),  # cached; transcriber default is large-v3 (GPU)
}

# ── LLM therapist backend (Part 2 therapy_assistant) ──────────────────────────
LLM = {
    "backend": os.environ.get("LLM_BACKEND", "ollama"),       # ollama | openai | gemini | anthropic
    "model":   os.environ.get("LLM_MODEL", "llama3.1:8b"),     # installed locally via ollama
    "host":    os.environ.get("LLM_HOST", "http://localhost:11434"),
    "openai_api_key":    os.environ.get("OPENAI_API_KEY", ""),
    "gemini_api_key":    os.environ.get("GEMINI_API_KEY", ""),
    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
}

# ── Tunables ──────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "semantic_disorder_min": float(os.environ.get("SEM_DISORDER_MIN", "0.35")),  # session→disorder classify cutoff
    "vector_search_top_k":   int(os.environ.get("VECTOR_TOP_K", "3")),           # retrieved sessions per query
    "crisis_min_confidence": float(os.environ.get("CRISIS_MIN_CONF", "0.5")),
}

SESSION_LOG_DIR = Path(os.environ.get("SESSION_LOG_DIR", ROOT / "sessions"))

# ── Dataset provenance (Layer 6 source-of-truth manifest; see DATASETS.md) ─────
DATASETS = {
    "annomi":                        {"tier": "gold",      "kind": "real_MI_transcripts",     "sessions": 133},
    "counsel_chat":                  {"tier": "gold",      "kind": "licensed_therapist_QA",   "sessions": 2775},
    "mental_health_counseling":      {"tier": "gold",      "kind": "licensed_professional_QA","sessions": 3512},
    "esconv":                        {"tier": "silver",    "kind": "trained_roleplay",        "sessions": 1300},
    "extes":                         {"tier": "synthetic", "kind": "llm_generated",           "sessions": 90063},
    "empathetic_dialogues":          {"tier": "synthetic", "kind": "crowdsourced_nonclinical","sessions": 24850},
    # dropped: epitome (duplicate of MHCC), hope (gated — request access separately)
}

def neo4j_driver():
    """Convenience factory used by pipelines/api that want the shared config."""
    from neo4j import GraphDatabase
    pw = NEO4J["password"]
    if not pw:
        raise RuntimeError("NEO4J_PASSWORD not set. Add it to your .env / environment.")
    return GraphDatabase.driver(NEO4J["uri"], auth=(NEO4J["user"], pw))
