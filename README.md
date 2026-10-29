# Mend — Clinician Assistant

**A clinical decision-support tool for psychologists.** Mend is used by the
psychologist during or after a session; the client does not see or interact with it.

The psychologist enters (types or dictates) what the client says, their own
observations, or a direct question. For every entry Mend returns a **clinical brief**:

- **Observations** — affect, themes, cognitive patterns, notable language
- **Working hypotheses** — candidate presentations / differentials with supporting and missing evidence
- **Risk** — risk indicators, with an immediate risk alert and recommended assessment steps when crisis language is detected
- **Suggested interventions** — 1–3 evidence-based options (CBT, DBT, ACT, MI…) matched to the session stage
- **Possible next questions / responses** — example wording the psychologist can adapt
- **Notes for the record**

Each brief is grounded in a 6-layer Mental Health Knowledge Graph (123,926 real therapy
sessions) via GraphRAG. Output is decision support only — clinical judgement and
responsibility remain with the psychologist.

```bash
streamlit run part1_conversational/app_integrated.py
```

> The KG-aligned emotion and behaviour models are in this repository under
> `part1_conversational/models/kg_emotion/` and `part1_conversational/models/kg_behaviour/`.
> Their weight files (`model.safetensors`, 499 MB + 269 MB) are stored with Git LFS: run `git lfs install`
> once before cloning (or `git lfs pull` in an existing clone), otherwise you get small pointer files.
> The app uses, in order: these KG-aligned models, then the older 11-class emotion / 8-class behaviour
> models (not in the repository), then keyword detection.
> `MEND_MODELS=legacy` skips the KG-aligned pair. On Windows, `Start-Mend.bat` starts Neo4j, the LLM and the app.

---

# Therapy — Integrated AI Therapy System

A two-part system (originally a client-facing therapy chatbot, now reframed as a clinician assistant), grounded by a
6-layer Mental Health Knowledge Graph (MLKG) containing 123,926 real therapy sessions.

## Architecture

```
Patient (text / voice)
        │
        ▼
┌─────────────────────────────┐
│  part1_conversational/      │  Conversational AI Layer
│  ─────────────────────────  │
│  • pii_scrubber.py          │  PII removal (Presidio, HIPAA/DPDP)
│  • transcriber.py           │  Voice → text (Whisper, multilingual)
│  • annotator_and_exporter   │  RoBERTa emotion + DistilBERT behaviour
│  • enhanced_chatbot.py      │  Orchestrator (emotion → KG → LLM → response)
│  • graphrag_client.py       │  Bridge to Part 2
│  • app_integrated.py        │  Mend GUI (Streamlit)
└────────────┬────────────────┘
             │  GraphRAGRequest
             │  {patient_text, emotion, behaviour, emotion_distribution}
             ▼
┌─────────────────────────────┐
│  part2_graphrag/            │  GraphRAG Knowledge Graph Layer
│  ─────────────────────────  │
│  api/graphrag_bridge.py     │  Pipeline orchestrator
│  api/structured_query.py    │  Signal → KG traversal (Cypher)
│  api/graph_semantic_search  │  Vector search (123,926 sessions)
│  api/technique_guidance.py  │  Technique → natural therapist language
│  api/crisis_detector.py     │  Safety-first crisis detection
│  api/session_memory.py      │  Conversation history
│  api/therapy_assistant.py   │  LLM adapter (Ollama/OpenAI/Gemini/Anthropic)
└────────────┬────────────────┘
             │  GraphRAGResponse
             │  {disorders, techniques, guidance, retrieved_sessions}
             ▼
        LLM (IS the therapist)
             │
             ▼
     Therapist response → Patient
             │
             ▼
      sessions/ (EHR logs)
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Start Neo4j and verify the KG is loaded
Ensure Neo4j is running with the `therapy` database populated.

### 3. Install an LLM (default: Ollama)
```bash
# Download from https://ollama.com/download
ollama pull llama3.2
```

### 4. Run the demo
```bash
# Windows
run_demo.bat

# Mac/Linux
./run_demo.sh

# Manual
streamlit run part1_conversational/app_integrated.py
```

### 5. Switch LLM backend
Edit the top of `part1_conversational/app_integrated.py`:
```python
BACKEND = "openai"         # ollama | openai | gemini | anthropic
MODEL   = "gpt-4o-mini"
API_KEY = "sk-..."
```

## Project Structure

```
Therapy/
├── README.md
├── requirements.txt               Unified dependencies
├── .env.example                   Configuration template
├── run_demo.bat / run_demo.sh     One-click launcher
│
├── part1_conversational/          Part 1 — Conversational AI
│   ├── app_integrated.py          ★ Main entry point (Mend GUI)
│   ├── enhanced_chatbot.py        LLM-as-therapist orchestrator
│   ├── graphrag_client.py         Bridge client to Part 2
│   ├── pii_scrubber.py            PII removal
│   ├── transcriber.py             Voice → text (Whisper)
│   ├── annotator_and_exporter.py  Emotion + Behaviour models
│   ├── chatbot_demo.py            Original rule-based chatbot (reference)
│   ├── app.py                     Original Mend GUI (reference)
│   └── data/
│       └── base_dataset.jsonl     Curated therapy dataset
│
├── part2_graphrag/                Part 2 — GraphRAG KG
│   └── api/
│       ├── __init__.py
│       ├── graphrag_bridge.py     ★ Data contract + pipeline
│       ├── graph_semantic_search  Vector search + graph traversal
│       ├── structured_query.py    Signal → Cypher traversal
│       ├── technique_guidance.py  KG technique → therapist language
│       ├── crisis_detector.py     Safety layer
│       ├── session_memory.py      Conversation history
│       └── therapy_assistant.py   LLM adapter (all backends)
│   └── pipelines/                 KG ingestion scripts (reference)
│
└── sessions/                      EHR-ready session logs (auto-created)
```

## Data Contract

### Part 1 → Part 2 (GraphRAGRequest)
```python
{
    "patient_utterance":    str,   # PII-scrubbed patient text
    "emotion":              str,   # RoBERTa top label  e.g. "anxiety"
    "behaviour":            str,   # DistilBERT top label e.g. "Avoidance"
    "emotion_distribution": dict,  # {"anxiety": 0.72, "fear": 0.20, ...}
    "session_id":           str,
    "turn_number":          int,
}
```

### Part 2 → Part 1 (GraphRAGResponse)
```python
{
    "disorders":             list,  # ["SocialAnxietyDisorder"]
    "symptoms":              list,  # ["Avoidance", "Intrusive Thoughts"]
    "recommended_techniques":list,  # ["CognitiveRestructuring"]
    "worsening_contexts":    list,  # ["SocialRejection"]
    "co_occurring_disorders":list,
    "retrieved_sessions":    list,  # top 3 real therapy dialogues
    "technique_guidance":    list,  # natural therapist framing per technique
    "confidence":            float,
    "was_crisis":            bool,
    "crisis_response":       str | None,
}
```

## The three parts
1. **Conversational AI** (`part1_conversational/`) — PII scrubbing, voice→text, RoBERTa
   emotion + DistilBERT behaviour models, LLM-as-therapist orchestration. **Built.**
2. **GraphRAG** (`part2_graphrag/`) — the 6-layer MLKG (100 canonical ICD-11/DSM-5-TR
   disorders + ~120k sessions) that grounds the therapist. **Built.**
3. **EHR** — role-based review where a doctor approves a session and approved data is
   fed back into the graph as clinician-verified edges. **Not yet built (planned).**

## Build & data
- **`BUILD.md`** — reproducible graph build order (run `00_seed_100_disorders.py` first,
  finish with `10_cleanup_canonical_100.py`). `Therapy/` is the single source of truth;
  `finalmlkg/` is superseded.
- **`DATASETS.md`** — Layer-6 dataset provenance (gold / silver / synthetic tiers).
- **`config/settings.py`** — central config (Neo4j, model paths, LLM backend, thresholds).

## Notes
- Session logs in `sessions/` are stored as JSON for EHR review (Part 3, planned)
- The LLM IS the therapist — no human therapist is in the loop at inference time
- Graph invariant: exactly **100** `:Disorder` nodes; Borderline pattern is a
  `:DiagnosticQualifier`; suicidal ideation / self-injury are `:RiskMarker`.
