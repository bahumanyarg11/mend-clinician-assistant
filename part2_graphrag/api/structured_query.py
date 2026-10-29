"""
structured_query.py
===================
Direct graph traversal from structured emotion/behaviour signals.
This is separate from vector semantic search — it takes the output of
Part 1's RoBERTa (emotion) and DistilBERT (behaviour) models and traverses
the KG ontology directly via Cypher.

No embeddings needed. Pure graph traversal.

Part 1 emotion labels  →  KG EmotionalState node names
Part 1 behaviour labels →  KG Behaviour node names

Usage:
    from finalmlkg.api.structured_query import query_by_signals
    results = query_by_signals(
        driver,
        emotions=["anxiety", "fear"],
        behaviours=["Avoidance"],
        emotion_distribution={"anxiety": 0.55, "fear": 0.40, "neutral": 0.05}
    )
"""

__all__ = ["query_by_signals", "EMOTION_LABEL_MAP", "BEHAVIOUR_LABEL_MAP"]

from neo4j import GraphDatabase

# ── Label mapping: Part 1 model output → KG node names ─────────────────────────
EMOTION_LABEL_MAP: dict[str, str] = {
    "anxiety":          "Anxiety",
    "sadness":          "Sadness",
    "anger":            "Anger",
    "frustration":      "Frustration",
    "shame_guilt":      "Shame",      # RoBERTa combined label
    "fear":             "Fear",
    "grief":            "Grief",
    "relief":           "Relief",
    "positive_progress":"PositiveProgress",
    "confusion":        "Confusion",
    "neutral":          "Neutral",
    # aliases / extra coverage
    "guilt":            "Guilt",
    "shame":            "Shame",
    "hopelessness":     "Hopelessness",
    "loneliness":       "Loneliness",
    "dissociation":     "Dissociation",
    "emptiness":        "Emptiness",
    "worthlessness":    "Worthlessness",
    "panic":            "Panic",
    "stress":           "Stress",
    "worry":            "Worry",
    "grief_loss":       "Grief",
    "embarrassment":    "Embarrassment",
    "shameguilt":       "Shame",      # KG-aligned emotion model (2026-10-28)
}

BEHAVIOUR_LABEL_MAP: dict[str, str] = {
    # ── Exact graph node names (verified against Neo4j) ────────────────────────
    "Social_Withdrawal":    "SocialWithdrawal",     # graph node: SocialWithdrawal
    "Rumination":           "Rumination",
    "Avoidance":            "Avoidance",
    "Help_Seeking":         "HelpSeeking",
    "Active_Coping":        "ActiveCoping",
    "Self_Harm":            "SelfHarm",
    "Substance_Use":        "SubstanceUse",
    # ── Model labels with no direct graph node → map to nearest real node ──────
    "Emotional_Expression": "SocialWithdrawal",     # broad expression → generic
    "Negative_Self_Talk":   "Rumination",           # self-talk is a form of rumination
    "Conversational_Act":   "HelpSeeking",          # reaching out = help-seeking
    # KG-aligned behaviour model (2026-10-28): labels newer than the 'therapy' graph -> nearest existing node
    "AvoidanceBehaviour":            "Avoidance",
    "NegativeSelfTalk":              "Rumination",
    "AggressionVerbalPhysical":      "Aggression",
    "AngerOutburstWithDamage":       "Aggression",
    "ImpulsiveAngerOutburst":        "Aggression",
    "PlannedInstrumentalAggression": "Aggression",
    "ReassuranceSeeking":            "HelpSeeking",
    "CompanionSeeking":              "HelpSeeking",
}


def _map_emotions(emotions: list[str]) -> list[str]:
    """Convert Part 1 emotion labels to KG node names, deduplicated."""
    mapped = []
    for e in emotions:
        kg_name = EMOTION_LABEL_MAP.get(e.lower(), e.capitalize())
        if kg_name and kg_name not in mapped:
            mapped.append(kg_name)
    return mapped


def _map_behaviours(behaviours: list[str]) -> list[str]:
    """Convert Part 1 behaviour labels to KG node names."""
    return [BEHAVIOUR_LABEL_MAP.get(b, b) for b in behaviours]


def query_by_signals(
    driver,
    emotions: list[str],
    behaviours: list[str],
    symptoms: list[str] | None = None,
    emotion_distribution: dict[str, float] | None = None,
    top_k_disorders: int = 3,
) -> dict:
    """
    Pure graph traversal from emotion + behaviour labels.
    Returns disorders, symptoms, recommended techniques, comorbidities,
    worsening contexts, and confidence scores — all from Cypher traversal only.

    Parameters
    ----------
    driver : neo4j.GraphDatabase.Driver
    emotions : list[str]
        Emotion label(s) from Part 1's RoBERTa model (e.g. ["anxiety", "fear"])
    behaviours : list[str]
        Behaviour label(s) from Part 1's DistilBERT model (e.g. ["Avoidance"])
    emotion_distribution : dict[str, float] | None
        Full probability distribution from RoBERTa (e.g. {"anxiety": 0.55, "fear": 0.40})
        If provided, secondary emotions above 0.25 threshold are also included.
    top_k_disorders : int
        Maximum number of disorders to return.

    Returns
    -------
    dict with keys: disorders, symptoms, techniques, comorbidities, contexts,
                    modalities, confidence, emotion_nodes_matched, behaviour_nodes_matched
    """

    # ── Build effective emotion list (primary + high-confidence secondary) ──────
    effective_emotions = list(emotions)
    if emotion_distribution:
        for emo, prob in emotion_distribution.items():
            if prob >= 0.25 and emo not in effective_emotions:
                effective_emotions.append(emo)

    kg_emotions   = _map_emotions(effective_emotions)
    kg_behaviours = _map_behaviours(behaviours)

    # ── CYPHER: emotion → disorder → technique (via ASSOCIATED_WITH + RECOMMENDED_FOR) ──
    EMOTION_QUERY = """
    UNWIND $emotions AS emo_name
    MATCH (e:EmotionalState) WHERE toLower(e.name) = toLower(emo_name)
    OPTIONAL MATCH (e)-[a:ASSOCIATED_WITH]->(d:Disorder)
    OPTIONAL MATCH (d)-[r:RECOMMENDED_FOR]-(t:TherapeuticTechnique)
    OPTIONAL MATCH (d)-[:MANIFESTS_AS]->(sym:Symptom)
    OPTIONAL MATCH (d)-[:CO_OCCURS_WITH]->(d2:Disorder)
    OPTIONAL MATCH (ctx:ContextCategory)-[:WORSENS]->(d)
    OPTIONAL MATCH (m:TherapeuticModality)-[:RECOMMENDED_FOR]->(d)
    RETURN
        collect(DISTINCT e.name)   AS emotion_nodes_matched,
        collect(DISTINCT {name: d.name, confidence: a.confidence}) AS disorder_hits,
        collect(DISTINCT t.name)   AS techniques,
        collect(DISTINCT sym.name) AS symptoms,
        collect(DISTINCT d2.name)  AS comorbidities,
        collect(DISTINCT ctx.name) AS contexts,
        collect(DISTINCT m.name)   AS modalities
    """

    # ── CYPHER: behaviour → disorder → technique ────────────────────────────────
    BEHAVIOUR_QUERY = """
    UNWIND $behaviours AS beh_name
    MATCH (b:Behaviour) WHERE toLower(b.name) = toLower(beh_name)
    OPTIONAL MATCH (s:Session)-[:EXHIBITS_BEHAVIOUR]->(b)
    OPTIONAL MATCH (s)-[:INDICATES_DISORDER]->(d:Disorder)
    OPTIONAL MATCH (d)-[:RECOMMENDED_FOR]-(t:TherapeuticTechnique)
    RETURN
        collect(DISTINCT b.name)  AS behaviour_nodes_matched,
        collect(DISTINCT d.name)  AS disorders_from_behaviour,
        collect(DISTINCT t.name)  AS techniques_from_behaviour
    """

    # ── CYPHER: symptom → disorder → technique ──────────────────────────────────
    SYMPTOM_QUERY = """
    UNWIND $symptoms AS sym_name
    MATCH (sym:Symptom) WHERE toLower(replace(sym.name, " ", "")) = toLower(replace(sym_name, " ", ""))
    OPTIONAL MATCH (d:Disorder)-[:MANIFESTS_AS]->(sym)
    OPTIONAL MATCH (d)-[:RECOMMENDED_FOR]-(t:TherapeuticTechnique)
    RETURN
        collect(DISTINCT sym.name)  AS symptom_nodes_matched,
        collect(DISTINCT d.name)    AS disorders_from_symptom,
        collect(DISTINCT t.name)    AS techniques_from_symptom
    """

    result = {
        "disorders":              [],
        "symptoms":               [],
        "techniques":             [],
        "comorbidities":          [],
        "contexts":               [],
        "modalities":             [],
        "emotion_nodes_matched":  [],
        "behaviour_nodes_matched":[],
        "symptom_nodes_matched":  [],
        "confidence":             0.0,
    }

    with driver.session(database="therapy") as session:
        # Emotion traversal
        if kg_emotions:
            row = session.run(EMOTION_QUERY, emotions=kg_emotions).single()
            if row:
                result["emotion_nodes_matched"] = row["emotion_nodes_matched"] or []
                result["symptoms"]   = [s for s in (row["symptoms"] or []) if s]
                result["techniques"] = [t for t in (row["techniques"] or []) if t]
                result["comorbidities"] = [d for d in (row["comorbidities"] or []) if d]
                result["contexts"]   = [c for c in (row["contexts"] or []) if c]
                result["modalities"] = [m for m in (row["modalities"] or []) if m]

                # Sort disorders by confidence, take top_k
                disorder_hits = [h for h in (row["disorder_hits"] or []) if h.get("name")]
                disorder_hits.sort(key=lambda x: x.get("confidence") or 0, reverse=True)
                result["disorders"] = [h["name"] for h in disorder_hits[:top_k_disorders]]

                # Compute overall confidence from primary emotion match
                if disorder_hits:
                    result["confidence"] = disorder_hits[0].get("confidence") or 0.6

        # Behaviour traversal — merge additional signals
        if kg_behaviours:
            row2 = session.run(BEHAVIOUR_QUERY, behaviours=kg_behaviours).single()
            if row2:
                result["behaviour_nodes_matched"] = row2["behaviour_nodes_matched"] or []

                # Merge disorders from behaviour (add if not already in list)
                for d in (row2["disorders_from_behaviour"] or []):
                    if d and d not in result["disorders"]:
                        result["disorders"].append(d)

                # Merge techniques
                for t in (row2["techniques_from_behaviour"] or []):
                    if t and t not in result["techniques"]:
                        result["techniques"].append(t)

        # Symptom traversal — merge additional signals
        if symptoms:
            row3 = session.run(SYMPTOM_QUERY, symptoms=symptoms).single()
            if row3:
                result["symptom_nodes_matched"] = row3["symptom_nodes_matched"] or []
                
                # Merge disorders from symptoms
                for d in (row3["disorders_from_symptom"] or []):
                    if d and d not in result["disorders"]:
                        result["disorders"].append(d)
                
                # Merge techniques from symptoms
                for t in (row3["techniques_from_symptom"] or []):
                    if t and t not in result["techniques"]:
                        result["techniques"].append(t)
                        
                # Ensure symptoms are reported in the result
                for s in (row3["symptom_nodes_matched"] or []):
                    if s and s not in result["symptoms"]:
                        result["symptoms"].append(s)

    # Clamp lists
    result["disorders"]  = result["disorders"][:top_k_disorders]
    result["techniques"] = result["techniques"][:6]
    result["symptoms"]   = result["symptoms"][:8]
    result["contexts"]   = result["contexts"][:5]

    return result
