"""Fige la version de référence (baseline) à partir du modèle versionné actuel.

Refuse d'écraser une baseline existante sans --force (décision explicite et revue).
"""

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from app.preprocessing import VARIABLES_BRUTES, preprocess
from app.regression import CRITERES_ACCEPTATION, TOLERANCE_PROBABILITE, TOLERANCES
from scripts.entrainer_modele import charger_et_decouper, evaluer

CHEMIN_BASELINE = Path("reports/baseline.json")
CHEMIN_MODELE = "models/model_v1.joblib"

if CHEMIN_BASELINE.exists() and "--force" not in sys.argv:
    sys.exit(f"ARRÊT : {CHEMIN_BASELINE} existe déjà. Remplacement uniquement par décision "
             "explicite : python -m scripts.creer_baseline --force")

paquet = joblib.load(CHEMIN_MODELE)
X_train, X_test, y_train, y_test = charger_et_decouper()
metriques = {k: float(v) for k, v in evaluer(paquet["pipeline"], X_test, y_test).items()}

# 50 demandes de référence fixes : leurs probabilités servent au contrôle fin
reference = X_test.sample(50, random_state=0)
probas = paquet["pipeline"].predict_proba(preprocess(reference))[:, 1]

baseline = {
    "date_creation": datetime.now(UTC).date().isoformat(),
    "version_modele": paquet["version"], "parametres": paquet["parametres"],
    "versions_bibliotheques": {"python": platform.python_version(),
                               "scikit-learn": sklearn.__version__, "pandas": pd.__version__},
    "graine": paquet["graine"],
    "criteres_acceptation": CRITERES_ACCEPTATION, "tolerances": TOLERANCES,
    "tolerance_probabilite": TOLERANCE_PROBABILITE,
    "metriques_test": metriques,
    "demandes_reference": [
        {**{v: (int(r[v]) if v in ("age", "duree_mois", "anciennete_activite") else float(r[v]))
            for v in VARIABLES_BRUTES}, "probabilite_defaut": float(p)}
        for (_, r), p in zip(reference.iterrows(), probas)],
}
CHEMIN_BASELINE.parent.mkdir(exist_ok=True)
CHEMIN_BASELINE.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Baseline figée : modèle {baseline['version_modele']}")
print(json.dumps(metriques, indent=2))