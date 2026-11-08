"""Non-régression : comparaison d'une version à la référence (baseline) figée.

Les critères et tolérances sont définis AVANT toute comparaison.
"""

CRITERES_ACCEPTATION = {"recall_min": 0.70, "f1_min": 0.60}
TOLERANCES = {"recall": 0.03, "f1": 0.03, "roc_auc": 0.02}
TOLERANCE_PROBABILITE = 0.02  # écart absolu maximal toléré par demande de référence


def compare_to_baseline(baseline: dict, nouveau: dict):
    """Renvoie (décision, motifs) : ACCEPTER, INVESTIGUER ou REJETER."""
    motifs = []
    if nouveau["recall"] < CRITERES_ACCEPTATION["recall_min"]:
        motifs.append(f"rappel {nouveau['recall']:.3f} < {CRITERES_ACCEPTATION['recall_min']}")
    if nouveau["f1"] < CRITERES_ACCEPTATION["f1_min"]:
        motifs.append(f"F1 {nouveau['f1']:.3f} < {CRITERES_ACCEPTATION['f1_min']}")
    if motifs:
        return "REJETER", motifs

    for metrique, tolerance in TOLERANCES.items():
        delta = nouveau[metrique] - baseline[metrique]
        if delta < -tolerance:
            motifs.append(f"{metrique} : {delta:+.3f} (tolérance -{tolerance})")
    if motifs:
        return "INVESTIGUER", motifs
    return "ACCEPTER", []