"""Rapport de non-régression : versions candidates comparées à la baseline figée."""

import json

import joblib
import numpy as np
import pandas as pd

from app.preprocessing import VARIABLES_BRUTES, preprocess
from app.regression import TOLERANCE_PROBABILITE, compare_to_baseline
from scripts.entrainer_modele import charger_et_decouper, entrainer, evaluer

with open("reports/baseline.json", encoding="utf-8") as f:
    baseline = json.load(f)
ref_metriques = baseline["metriques_test"]
ref_demandes = pd.DataFrame(baseline["demandes_reference"])

X_train, X_test, y_train, y_test = charger_et_decouper()
candidats = {
    "v1.1.0 : correction ANO-01 (alertes sur variables dérivées)":
        joblib.load("models/model_v1.joblib")["pipeline"],
    "v2-a : sans pondération des classes (class_weight=None)":
        entrainer(X_train, y_train, class_weight=None),
    "v2-b : régularisation très forte (C=0,001)":
        entrainer(X_train, y_train, C=0.001),
}

print(f"BASELINE {baseline['version_modele']} : " +
      " | ".join(f"{k} {v:.4f}" for k, v in ref_metriques.items()))
rapport = {"baseline": baseline["version_modele"], "candidats": {}}
for nom, pipeline in candidats.items():
    m = evaluer(pipeline, X_test, y_test)
    decision, motifs = compare_to_baseline(ref_metriques, m)
    probas = pipeline.predict_proba(preprocess(ref_demandes[VARIABLES_BRUTES]))[:, 1]
    ecart = float(np.max(np.abs(probas - ref_demandes["probabilite_defaut"])))
    if decision == "ACCEPTER" and ecart > TOLERANCE_PROBABILITE:
        decision, motifs = "INVESTIGUER", [f"écart de probabilité {ecart:.4f} > {TOLERANCE_PROBABILITE}"]
    print(f"\n{nom}")
    print("  " + " | ".join(f"{k} {v:.4f} ({v - ref_metriques[k]:+.4f})" for k, v in m.items())
          + f" | écart proba max {ecart:.4f}")
    print(f"  -> {decision}" + (f" : {motifs}" if motifs else ""))
    rapport["candidats"][nom] = {"decision": decision, "motifs": motifs,
                                 "metriques": {k: float(v) for k, v in m.items()},
                                 "ecart_probabilite_max": ecart}

with open("reports/rapport_non_regression.json", "w", encoding="utf-8") as f:
    json.dump(rapport, f, indent=2, ensure_ascii=False)
print("\nRapport écrit dans reports/rapport_non_regression.json")