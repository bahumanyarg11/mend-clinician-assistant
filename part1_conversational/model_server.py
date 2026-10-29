"""
model_server.py
===============
Loads the 3 custom-trained local models from the models/ directory.
No internet or HuggingFace download needed — all weights are local.

  DEFAULT — KG-aligned models (retrained 2026-10-28, training/kg_retrain), labels are KG node names
    models/kg_emotion/   RoBERTa, 35 EmotionalState labels, multilabel
    models/kg_behaviour/ DistilBERT, 214 labels (27 Behaviour, 14 CognitivePattern, 173 Symptom), multilabel
    Decision: sigmoid >= max(per-label tuned threshold, floor). Floors stop rare labels with
    under-tuned thresholds firing on near-zero scores. Set MEND_MODELS=legacy to use the old pair below.

  LEGACY (fallback when models/kg_* is missing or fails to load):
  Model 1 — Emotion (11-class RoBERTa, multilabel)
    Path: models/emotion_model_1/emotion_model_1/model/
    Labels: anger, frustration, sadness, confusion, shame_guilt,
            fear, anxiety, relief, grief, positive_progress, neutral
    Decision: sigmoid + 0.5 threshold (multilabel)

  Model 2 — Behaviour (8-class DistilBERT, single-label)
    Path: models/behaviour_model/behaviour_model/models/distilbert_behaviour_model/
    Labels: Social_Withdrawal, Rumination, Avoidance, Negative_Self_Talk,
            Emotional_Expression, Help_Seeking, Active_Coping, Conversational_Act
    Decision: softmax + argmax

  Model 3 — Sentence-Transformers (all-MiniLM-L6-v2)
    Used by GraphRAG KG search — already loaded in part2_graphrag/api/

All three run on CPU. GPU stays entirely free for Ollama LLM.

Usage:
    python model_server.py              # self-test
    from model_server import ModelServer
    ms = ModelServer()
    emotion, dist = ms.predict_emotion("I feel so anxious")
    behaviour, conf = ms.predict_behaviour("I keep avoiding everyone")
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ModelServer")

# ── Absolute paths to extracted model directories ────────────────────────────────
_HERE = Path(__file__).parent

EMOTION_MODEL_DIR   = _HERE / "models" / "emotion_model_1" / "emotion_model_1" / "model"
BEHAVIOUR_MODEL_DIR = _HERE / "models" / "behaviour_model" / "behaviour_model" / "models" / "distilbert_behaviour_model"

# ── Label definitions (order must match training) ────────────────────────────────
EMOTION_LABELS = [
    "anger", "frustration", "sadness", "confusion", "shame_guilt",
    "fear", "anxiety", "relief", "grief", "positive_progress", "neutral"
]
EMOTION_THRESHOLD = 0.5   # multilabel sigmoid threshold

BEHAVIOUR_LABELS = [
    "Social_Withdrawal", "Rumination", "Avoidance", "Negative_Self_Talk",
    "Emotional_Expression", "Help_Seeking", "Active_Coping", "Conversational_Act"
]

# ── KG-aligned models (default) ──────────────────────────────────────────────────
KG_EMOTION_DIR   = _HERE / "models" / "kg_emotion"
KG_BEHAVIOUR_DIR = _HERE / "models" / "kg_behaviour"
USE_KG_MODELS    = os.environ.get("MEND_MODELS", "kg").lower() != "legacy"
KG_EMOTION_FLOOR   = 0.6
KG_BEHAVIOUR_FLOOR = 0.3
KG_THRESHOLD_OVERRIDE = {"EmotionalState:Emptiness": 0.9}   # tuned to 0.5 vs ~0.85 for the rest; fired on most inputs
KG_NEUTRAL = "Neutral"


class ModelServer:
    """
    Loads and caches both custom models at startup.
    Falls back to keyword heuristics if model files are missing.
    """

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._emotion_model      = None
        self._emotion_tokenizer  = None
        self._emotion_mode       = "keyword"

        self._behaviour_model     = None
        self._behaviour_tokenizer = None
        self._behaviour_mode      = "keyword"
        self._signals_cache       = (None, None)   # (text, signals): one behaviour pass serves behaviour + symptoms

        if not (USE_KG_MODELS and self._load_kg("emotion")):
            self._load_emotion()
        if not (USE_KG_MODELS and self._load_kg("behaviour")):
            self._load_behaviour()

    # ─────────────────────────── KG-ALIGNED MODELS ──────────────────────────────

    def _load_kg(self, task: str) -> bool:
        import json
        d = KG_EMOTION_DIR if task == "emotion" else KG_BEHAVIOUR_DIR
        if not (d / "model").exists():
            logger.warning(f"KG {task} model not found at {d}. Using the legacy model.")
            return False
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            logger.info(f"Loading KG-aligned {task} model from {d}...")
            tok   = AutoTokenizer.from_pretrained(str(d / "model"))
            model = AutoModelForSequenceClassification.from_pretrained(str(d / "model")).eval()
            labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
            tuned  = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))
            floor  = KG_EMOTION_FLOOR if task == "emotion" else KG_BEHAVIOUR_FLOOR
            th     = [max(KG_THRESHOLD_OVERRIDE.get(l, tuned.get(l, 0.5)), floor) for l in labels]
            if task == "emotion":
                self._emotion_tokenizer, self._emotion_model = tok, model
                self._kg_emotion = (labels, th)
                self._emotion_mode = "kg_roberta_35"
            else:
                self._behaviour_tokenizer, self._behaviour_model = tok, model
                self._kg_behaviour = (labels, th)
                self._behaviour_mode = "kg_distilbert_214"
            logger.info(f"  KG {task} model loaded OK ({len(labels)} labels).")
            return True
        except Exception as e:
            logger.warning(f"KG {task} model load failed: {e}. Using the legacy model.")
            return False

    def _kg_scores(self, tok, model, labels, th, text) -> list[tuple[str, str, float]]:
        """[(group, kg_name, prob)] for labels at or above their threshold, best first."""
        import torch
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            probs = torch.sigmoid(model(**enc).logits[0]).tolist()
        hits = [(*labels[i].split(":", 1), round(p, 4)) for i, p in enumerate(probs) if p >= th[i]]
        return sorted(hits, key=lambda h: -h[2])

    def predict_signals(self, text: str) -> dict:
        """KG behaviour model output grouped by node type: {'behaviours': [(name, p)], 'cognitive_patterns': [...], 'symptoms': [...]}.
        Names are KG node names (part2 structured_query maps the newer ones onto nodes the running graph has). Empty lists with the legacy model."""
        if self._behaviour_mode != "kg_distilbert_214":
            return {"behaviours": [], "cognitive_patterns": [], "symptoms": []}
        if self._signals_cache[0] == text:
            return self._signals_cache[1]
        out = {"behaviours": [], "cognitive_patterns": [], "symptoms": []}
        key = {"Behaviour": "behaviours", "CognitivePattern": "cognitive_patterns", "Symptom": "symptoms"}
        for group, name, p in self._kg_scores(self._behaviour_tokenizer, self._behaviour_model, *self._kg_behaviour, text):
            out[key[group]].append((name, p))
        self._signals_cache = (text, out)
        return out

    # ─────────────────────────── EMOTION ────────────────────────────────────────

    def _load_emotion(self):
        if not EMOTION_MODEL_DIR.exists():
            logger.warning(f"Emotion model not found at {EMOTION_MODEL_DIR}. Using keyword fallback.")
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            logger.info(f"Loading emotion model (RoBERTa 11-class) from {EMOTION_MODEL_DIR}...")
            self._emotion_tokenizer = AutoTokenizer.from_pretrained(str(EMOTION_MODEL_DIR))
            self._emotion_model     = AutoModelForSequenceClassification.from_pretrained(
                str(EMOTION_MODEL_DIR)
            )
            self._emotion_model.eval()
            if device := self._device:
                if device == "cuda":
                    import torch
                    if torch.cuda.is_available():
                        self._emotion_model = self._emotion_model.cuda()
            self._emotion_mode = "roberta_custom"
            logger.info("  Emotion model loaded OK.")
        except Exception as e:
            logger.warning(f"Emotion model load failed: {e}. Using keyword fallback.")

    def _load_behaviour(self):
        if not BEHAVIOUR_MODEL_DIR.exists():
            logger.warning(f"Behaviour model not found at {BEHAVIOUR_MODEL_DIR}. Using keyword fallback.")
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            logger.info(f"Loading behaviour model (DistilBERT 8-class) from {BEHAVIOUR_MODEL_DIR}...")
            self._behaviour_tokenizer = AutoTokenizer.from_pretrained(str(BEHAVIOUR_MODEL_DIR))
            self._behaviour_model     = AutoModelForSequenceClassification.from_pretrained(
                str(BEHAVIOUR_MODEL_DIR)
            )
            self._behaviour_model.eval()
            self._behaviour_mode = "distilbert_custom"
            logger.info("  Behaviour model loaded OK.")
        except Exception as e:
            logger.warning(f"Behaviour model load failed: {e}. Using keyword fallback.")

    # ─────────────────────────── PREDICT ────────────────────────────────────────

    def predict_emotion(self, text: str) -> tuple[str, dict[str, float]]:
        """Returns (top_label, {label: probability}). Legacy: all 11 classes. KG model: only emotions above threshold, KG node names."""
        if self._emotion_mode == "kg_roberta_35":
            hits = self._kg_scores(self._emotion_tokenizer, self._emotion_model, *self._kg_emotion, text)
            dist = {name: p for _, name, p in hits}
            if not dist:
                return KG_NEUTRAL, {KG_NEUTRAL: 1.0}
            return next(iter(dist)), dist
        if self._emotion_mode == "roberta_custom":
            return self._infer_emotion_roberta(text)
        return _keyword_emotion(text)

    def _infer_emotion_roberta(self, text: str) -> tuple[str, dict[str, float]]:
        import torch
        enc = self._emotion_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=64, padding=True
        )
        with torch.no_grad():
            logits = self._emotion_model(**enc).logits[0]

        probs = torch.sigmoid(logits).tolist()
        dist  = {label: round(p, 4) for label, p in zip(EMOTION_LABELS, probs)}

        # Primary label: highest probability above threshold
        above = {l: p for l, p in dist.items() if p >= EMOTION_THRESHOLD}
        if above:
            top = max(above, key=above.get)
        else:
            # Nothing above threshold — return highest anyway
            top = max(dist, key=dist.get)
        return top, dist

    def predict_behaviour(self, text: str) -> tuple[str, float]:
        """Returns (label, confidence). Legacy: one of 8 classes. KG model: the strongest Behaviour node, or 'Unclear'."""
        if self._behaviour_mode == "kg_distilbert_214":
            beh = self.predict_signals(text)["behaviours"]
            return beh[0] if beh else ("Unclear", 0.0)
        if self._behaviour_mode == "distilbert_custom":
            return self._infer_behaviour_distilbert(text)
        label = _keyword_behaviour(text)
        return label, 0.6

    def predict_symptoms(self, text: str) -> list[str]:
        """Symptom tags: KG Symptom nodes from the KG behaviour model, plus the keyword tags for the 38 proposed criteria gaps."""
        found = [n for n, _ in self.predict_signals(text)["symptoms"]]
        return found + [s for s in _keyword_symptom(text) if s not in found]

    def _infer_behaviour_distilbert(self, text: str) -> tuple[str, float]:
        import torch
        enc = self._behaviour_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128, padding=True
        )
        with torch.no_grad():
            logits = self._behaviour_model(**enc).logits[0]

        probs   = torch.softmax(logits, dim=-1).tolist()
        top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        return BEHAVIOUR_LABELS[top_idx], round(probs[top_idx], 4)

    def status(self) -> dict:
        return {
            "emotion_model":   self._emotion_mode,
            "behaviour_model": self._behaviour_mode,
            "emotion_model_dir":   str(KG_EMOTION_DIR if self._emotion_mode.startswith("kg_") else EMOTION_MODEL_DIR),
            "behaviour_model_dir": str(KG_BEHAVIOUR_DIR if self._behaviour_mode.startswith("kg_") else BEHAVIOUR_MODEL_DIR),
        }


# ── Keyword fallbacks (used when model files are missing) ────────────────────────

_EMOTION_KW = {
    "anxiety":          ["anxious","worried","panic","nervous","dread","terrified","stressed","restless","overwhelmed"],
    "sadness":          ["sad","depressed","hopeless","crying","unhappy","lonely","miserable","down","empty","numb"],
    "anger":            ["angry","furious","mad","irritated","pissed","hate","resentful","rage","bitter"],
    "frustration":      ["frustrated","stuck","annoyed","exhausted","tired of","fed up"],
    "shame_guilt":      ["guilty","ashamed","my fault","embarrassed","worthless","blame myself"],
    "fear":             ["scared","fearful","frightened","paralyzed","afraid","terrifying"],
    "grief":            ["lost","passed away","mourning","miss them","grieving","death"],
    "relief":           ["relieved","calmer","better now","feeling eased"],
    "positive_progress":["better","improving","hopeful","managed to","accomplished","proud"],
    "confusion":        ["confused","don't understand","not sure","lost","unclear","puzzled"],
}
_BEHAVIOUR_KW = {
    "Social_Withdrawal":  ["avoiding","isolating","shut inside","don't want to see","withdrawing"],
    "Rumination":         ["can't stop thinking","keeps replaying","over and over","stuck in my head","overthinking"],
    "Avoidance":          ["procrastinating","putting off","ignoring","running away from"],
    "Negative_Self_Talk": ["i am stupid","i'm a failure","nobody likes me","i always mess up","i ruin everything"],
    "Emotional_Expression":["i feel","i am experiencing","it hurts","i sense","my emotions"],
    "Help_Seeking":       ["what should i do","can you help","how do i cope","i need advice"],
    "Active_Coping":      ["i tried","i went for a walk","i practiced","i spoke to","journaled"],
    "Conversational_Act": ["hello","hi","good morning","thank you","bye","okay","yes"],
}

def _keyword_emotion(text: str) -> tuple[str, dict[str, float]]:
    lower = text.lower()
    scores = {}
    for emo, kws in _EMOTION_KW.items():
        m = sum(1 for kw in kws if kw in lower)
        if m: scores[emo] = m / len(kws)
    if not scores:
        return "neutral", {"neutral": 1.0}
    top = max(scores, key=scores.get)
    return top, scores

def _keyword_behaviour(text: str) -> str:
    lower = text.lower()
    for beh, kws in _BEHAVIOUR_KW.items():
        if any(kw in lower for kw in kws):
            return beh
    return "Emotional_Expression"


_SYMPTOM_KW = {
    "Low mood": ["sad", "down", "depressed", "empty", "irritable", "crying"],
    "Appetite/weight change": ["appetite", "weight", "eating too much", "not eating", "starving", "bingeing"],
    "Sleep disturbance": ["sleep", "insomnia", "waking up", "can't sleep", "nightmares"],
    "Fatigue/low energy": ["fatigue", "low energy", "exhausted", "tired", "no energy", "drained"],
    "Poor concentration/indecision": ["concentration", "focus", "indecisive", "mind blank", "distracted"],
    "Functional impact": ["work", "school", "getting out of bed", "can't function", "failing"],
    "Uncontrollable worry": ["worry", "can't stop worrying", "stressing", "overthinking"],
    "Restlessness/on edge": ["restless", "on edge", "jittery", "fidgeting"],
    "Muscle tension": ["muscle tension", "tense", "stiff", "aching", "tightness"],
    "Sudden-onset attack": ["panic attack", "sudden panic", "attack", "out of nowhere"],
    "Palpitations": ["palpitations", "heart racing", "heart pounding", "heart beating fast"],
    "Sweating": ["sweating", "sweaty", "drenched"],
    "Trembling": ["trembling", "shaking", "shivering", "quivering"],
    "Breathlessness": ["shortness of breath", "can't breathe", "breathless", "gasping"],
    "Choking sensation": ["choking", "lump in throat"],
    "Chest pain": ["chest pain", "chest hurts", "chest tightness", "chest pressure"],
    "Nausea/abdominal distress": ["nausea", "sick to my stomach", "stomach hurts", "throwing up"],
    "Dizziness/faintness": ["dizzy", "lightheaded", "faint", "passing out", "room spinning"],
    "Chills/heat sensations": ["chills", "hot flashes", "freezing", "burning up"],
    "Numbness/tingling": ["numbness", "tingling", "pins and needles"],
    "Derealization/depersonalization": ["unreal", "detached", "not myself", "like a dream", "out of body"],
    "Fear of losing control": ["losing control", "going crazy", "going mad"],
    "Fear of dying": ["dying", "going to die", "death"],
    "Anticipatory worry about attacks": ["worrying about another attack", "fear of panic", "happening again"],
    "Situational fear (agoraphobic situations)": ["crowds", "outside alone", "public transport", "trapped"],
    "Pain (by site)": ["pain", "hurts", "sore", "migraine", "headache"]
}

def _keyword_symptom(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for sym, kws in _SYMPTOM_KW.items():
        if any(kw in lower for kw in kws):
            found.append(sym)
    return found


# ── Self-test ────────────────────────────────────────────────────────────────────

def self_test():
    print("\n" + "="*65)
    print("  Model Server — Self Test")
    print("="*65)
    ms = ModelServer()
    print(f"\nStatus:")
    for k, v in ms.status().items():
        print(f"  {k}: {v}")

    samples = [
        "I feel so anxious and overwhelmed, I cannot breathe properly.",
        "I keep avoiding my friends and just stay in bed all day.",
        "I tried journaling today and went for a walk. It helped.",
        "I am a complete failure. Nobody would miss me if I was gone.",
        "Hello, good morning. How are you today?",
    ]
    print()
    for text in samples:
        t0 = time.time()
        emo, dist = ms.predict_emotion(text)
        beh, conf = ms.predict_behaviour(text)
        ms_taken  = (time.time() - t0) * 1000
        top3 = sorted(dist.items(), key=lambda x: -x[1])[:3]
        print(f"  Input    : {text[:60]}...")
        print(f"  Emotion  : {emo:20s}  top3={[(k, f'{v:.2f}') for k,v in top3]}")
        print(f"  Behaviour: {beh:25s}  conf={conf:.2f}")
        print(f"  Latency  : {ms_taken:.0f}ms\n")

    print("Self-test complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", default=True)
    self_test()
