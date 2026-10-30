"""
transcript_assist.py
====================
Live-session helper for the psychologist: takes a stretch of doctor-patient transcript with NO speaker
labels (as a voice transcript would arrive), works out who said each line, reads only the patient's lines
with the emotion/behaviour models and the knowledge graph, and returns a short note to glance at mid-session.

  1  speakers  LLM labels each line Doctor / Patient (heuristic fallback if the reply can't be parsed)
  2  signals   KG-aligned emotion + behaviour models on the patient's lines only (they were trained on client text)
  3  graph     GraphRAG lookup with the patient's words and signals; its crisis check runs first
  4  note      LLM writes a fixed-format note of about 60 words; a crisis puts a risk alert on top of it

Usage (from conversational_ai/, with Neo4j and the LLM up, e.g. after Start-Mend.bat):
    python part1_conversational/transcript_assist.py demo_transcripts.txt             # every case
    python part1_conversational/transcript_assist.py demo_transcripts.txt --case 3    # one case
LLM_MODEL / OLLAMA_HOST are read from the environment (.env as fallback).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

SPEAKER_PROMPT = """You label who is speaking in a transcript of a psychotherapy session between a psychologist \
(D) and their client (P). The lines are in speaking order and a speaker may say several lines in a row. The \
psychologist asks questions, reflects back what was said and guides the session; the client describes their own \
experiences and feelings. Reply with exactly one line per transcript line in the form "<number> D" or \
"<number> P", and nothing else."""

NOTE_PROMPT = """You are Mend, assisting a psychologist DURING a live session. They will glance at your note \
for a few seconds while still talking to the client, so it must be short and concrete.

Reply in exactly this format, at most 60 words in total, with no other text:
Likely: <one possible presentation> - "<the client's own words, at most 8 words>"
Also consider: <one or two alternative conditions>
Risk: <"none noted", or the client's words>
Ask next: "<one question the psychologist could ask now>"
Try: <one technique, a few words>

Rules: use only what the client actually said; the knowledge-graph candidates and model signals are hints, not \
facts, and may be wrong. Say "possible", never diagnose with certainty. Refer to "the client"; never assume a gender.
"Also consider" names other conditions (diagnoses) to rule out, never symptoms.
"Risk" is only for suicide or self-harm, harm to others, abuse or neglect, or another safeguarding concern; low \
self-worth, distress or other symptoms on their own are "none noted"."""

RISK_RULE = """
The client has disclosed thoughts of suicide or not wanting to live. Put their words under "Risk", and make \
"Ask next" a direct risk-assessment question (intent, plan, means or protective factors), not a general one."""

RISK_ALERT = "⚠ RISK ALERT - suicidal thoughts disclosed. Assess intent, plan and means now; agree a safety plan before the session ends."

# GoEmotions-style labels the emotion model knows that tell a psychologist little; kept out of the panel and the prompt
HIDDEN_EMOTIONS = {"Admiration", "Amusement", "Approval", "Caring", "Curiosity", "Desire", "Disapproval", "Excitement",
                   "Gratitude", "Joy", "Love", "Neutral", "Optimism", "Pride", "Realization", "Surprise"}


def _llm(host: str, model: str):
    import ollama
    client = ollama.Client(host=host)

    def call(system: str, user: str, max_tokens: int) -> str:
        # think=False: gemma4 otherwise spends the whole token budget on hidden reasoning and returns no text
        r = client.chat(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        options={"temperature": 0.2, "num_predict": max_tokens}, think=False)
        return r["message"]["content"].strip()
    return call


def load_cases(path: Path) -> list[dict]:
    """'### title' blocks; each line optionally starts with 'D:' / 'P:' (the true speaker, kept for scoring)."""
    cases, cur = [], None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("###"):
            cur = {"title": line.lstrip("#").strip(), "lines": [], "truth": []}
            cases.append(cur)
        elif line and not line.startswith("#") and cur is not None:
            m = re.match(r"([DP]):\s*(.+)", line)
            cur["truth"].append(m[1] if m else None)
            cur["lines"].append(m[2] if m else line)
    return cases


def label_speakers(llm, lines: list[str]) -> tuple[list[str], str]:
    numbered = "\n".join(f"{i} {t}" for i, t in enumerate(lines, 1))
    try:
        reply = llm(SPEAKER_PROMPT, numbered, 12 * len(lines) + 20)
        got = {int(n): s for n, s in re.findall(r"(\d+)\s*[:.)-]?\s*([DP])\b", reply)}
        if all(i in got for i in range(1, len(lines) + 1)):
            return [got[i] for i in range(1, len(lines) + 1)], "llm"
    except Exception:
        pass
    # Fallback: questions and reflections ("It sounds like", "So ...") are the psychologist's; the rest the client's
    doc = re.compile(r"\?\s*$|^(it sounds like|so |that must|i'?m (really )?glad you|tell me|thanks for)", re.I)
    return ["D" if doc.search(t) else "P" for t in lines], "heuristic"


def read_patient(ms, patient_lines: list[str]) -> dict:
    """Run the models on each patient line and keep each label's best score across the stretch."""
    emo, beh, cog, sym = {}, {}, {}, {}
    for t in patient_lines:
        _, dist = ms.predict_emotion(t)
        for k, p in dist.items():
            if k not in HIDDEN_EMOTIONS:
                emo[k] = max(emo.get(k, 0), p)
        sig = ms.predict_signals(t)
        for store, key in ((beh, "behaviours"), (cog, "cognitive_patterns"), (sym, "symptoms")):
            for k, p in sig.get(key, []):
                store[k] = max(store.get(k, 0), p)
    top = lambda d, n: sorted(d.items(), key=lambda x: -x[1])[:n]
    return {"emotions": top(emo, 4), "behaviours": top(beh, 3), "thinking": top(cog, 3), "symptoms": top(sym, 6)}


def assist(lines: list[str], llm, ms, kg) -> dict:
    from graphrag_client import build_request
    t = {}
    t0 = time.time(); speakers, how = label_speakers(llm, lines); t["speakers"] = time.time() - t0
    patient = [l for l, s in zip(lines, speakers) if s == "P"]
    t0 = time.time(); sig = read_patient(ms, patient); t["models"] = time.time() - t0
    names = lambda pairs: [n for n, _ in pairs]
    t0 = time.time()
    kg_resp = kg.query(build_request(
        patient_text=" ".join(patient),
        emotion=(names(sig["emotions"]) or ["Neutral"])[0],
        behaviour=(names(sig["behaviours"]) or ["Unclear"])[0],
        symptoms=names(sig["symptoms"]),
        emotion_dist=dict(sig["emotions"])))
    t["graph"] = time.time() - t0
    out = {"speakers": speakers, "speaker_method": how, "signals": sig, "crisis": kg_resp.was_crisis,
           "disorders": kg_resp.disorders[:4], "techniques": kg_resp.recommended_techniques[:4]}
    # A crisis still gets the full note, with the alert on top (the graph lookup returns nothing on a crisis)
    t0 = time.time()
    labelled = "\n".join(f"{'Psychologist' if s == 'D' else 'Client'}: {l}" for l, s in zip(lines, speakers))
    fmt = lambda pairs: ", ".join(f"{n} {p:.2f}" for n, p in pairs) or "none"
    note = llm(NOTE_PROMPT + (RISK_RULE if kg_resp.was_crisis else ""),
               f"Transcript so far:\n{labelled}\n\n"
               f"Model signals from the client's lines - emotions: {fmt(sig['emotions'])}; behaviours: "
               f"{fmt(sig['behaviours'])}; thinking patterns: {fmt(sig['thinking'])}; symptoms: {fmt(sig['symptoms'])}\n"
               f"Knowledge-graph candidates - presentations: {', '.join(out['disorders']) or 'none'}; "
               f"techniques: {', '.join(out['techniques']) or 'none'}", 160)
    out["note"] = f"{RISK_ALERT}\n\n{note}" if kg_resp.was_crisis else note
    t["note"] = time.time() - t0
    out["timings"] = {k: round(v, 1) for k, v in t.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts", type=Path)
    ap.add_argument("--case", type=int, help="run only this case number (1-based)")
    ap.add_argument("--out", type=Path, help="markdown report (default outputs/transcript_demo_<date>.md next to the project)")
    a = ap.parse_args()

    env = _HERE.parent / ".env"
    if env.exists():
        for l in env.read_text(encoding="utf-8").splitlines():
            if "=" in l and not l.lstrip().startswith("#"):
                k, v = l.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("LLM_MODEL", "gemma4:26b")

    from model_server import ModelServer
    from graphrag_client import GraphRAGClient
    llm, ms, kg = _llm(host, model), ModelServer(), GraphRAGClient()

    cases = load_cases(a.transcripts)
    if a.case:
        cases = [cases[a.case - 1]]
    report = [f"# Mend live-session demo ({datetime.now():%Y-%m-%d %H:%M})", "",
              f"LLM {model}. Each case is given to Mend without speaker labels.", ""]
    for c in cases:
        r = assist(c["lines"], llm, ms, kg)
        right = sum(s == t for s, t in zip(r["speakers"], c["truth"]) if t)
        known = sum(1 for t in c["truth"] if t)
        total = sum(r["timings"].values())
        pretty = lambda pairs: ", ".join(f"{n} {p:.0%}" for n, p in pairs) or "-"
        block = [f"## {c['title']}", "", "**Transcript as Mend labelled it**", ""]
        for line, s, t in zip(c["lines"], r["speakers"], c["truth"]):
            mark = "" if not t or s == t else "  ⚠ wrong speaker"
            block.append(f"- **{'Doctor' if s == 'D' else 'Patient'}:** {line}{mark}")
        block += ["", f"Speakers: {right}/{known} correct ({r['speaker_method']})", "",
                  f"Signals (patient only): emotions {pretty(r['signals']['emotions'])} · behaviours {pretty(r['signals']['behaviours'])}"
                  f" · thinking {pretty(r['signals']['thinking'])} · symptoms {pretty(r['signals']['symptoms'][:4])}", "",
                  f"Knowledge graph: {', '.join(r['disorders']) or '-'} · techniques {', '.join(r['techniques']) or '-'}", "",
                  "**Note for the doctor**", "", "```", r["note"], "```", "",
                  "Time: " + ", ".join(f"{k} {v}s" for k, v in r["timings"].items()) + f" (total {total:.1f}s)", ""]
        report += block
        print("\n".join(block), flush=True)
    out = a.out or _HERE.parents[1] / "outputs" / f"transcript_demo_{datetime.now():%Y-%m-%d}.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
