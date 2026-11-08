"""Tests de non-régression (Partie V) : version actuelle contre baseline figée."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.preprocessing import VARIABLES_BRUTES, preprocess
from app.regression import TOLERANCE_PROBABILITE, compare_to_baseline
from scripts.entrainer_modele import charger_et_decouper, evaluer

RACINE = Path(__file__).resolve().parents[1]
BASELINE = json.loads((RACINE / "reports" / "baseline.json").read_text(encoding="utf-8"))
CHEMIN_MODELE = RACINE / "models" / "model_v1.joblib"
REF = {"recall": 0.73, "f1": 0.68, "roc_auc": 0.89}


# ---------- Logique de décision (tolérances définies AVANT la comparaison) ----------

def test_version_identique_acceptee():
    assert compare_to_baseline(REF, dict(REF)) == ("ACCEPTER", [])


def test_variation_dans_la_tolerance_acceptee():
    assert compare_to_baseline(REF, {**REF, "recall": 0.71})[0] == "ACCEPTER"


def test_baisse_excessive_du_f1_a_investiguer():
    decision, motifs = compare_to_baseline(REF, {**REF, "f1": 0.63})
    assert decision == "INVESTIGUER"
    assert "f1" in motifs[0]


@pytest.mark.parametrize("modif, motif", [({"recall": 0.60}, "rappel"), ({"f1": 0.50}, "F1")])
def test_critere_critique_viole_rejete(modif, motif):
    decision, motifs = compare_to_baseline(REF, {**REF, **modif})
    assert decision == "REJETER"
    assert motif in motifs[0]


# ---------- La baseline elle-même ----------

def test_baseline_complete():
    for cle in ["version_modele", "parametres", "versions_bibliotheques", "graine",
                "criteres_acceptation", "tolerances", "metriques_test", "demandes_reference"]:
        assert cle in BASELINE, f"Élément manquant : {cle}"
    assert len(BASELINE["demandes_reference"]) == 50


# ---------- Version actuelle contre baseline ----------

@pytest.fixture(scope="module")
def paquet():
    return joblib.load(CHEMIN_MODELE)


def test_metriques_actuelles_contre_baseline(paquet):
    """Rappel et F1 du modèle actuel comparés à la référence figée."""
    _, X_test, _, y_test = charger_et_decouper(str(RACINE / "data" / "credit_synthetique.csv"))
    actuel = evaluer(paquet["pipeline"], X_test, y_test)
    decision, motifs = compare_to_baseline(BASELINE["metriques_test"], actuel)
    assert decision == "ACCEPTER", f"{decision} : {motifs}"


def test_probabilites_de_reference_dans_la_tolerance(paquet):
    """Les 50 demandes de référence : écart de probabilité <= tolérance documentée."""
    ref = pd.DataFrame(BASELINE["demandes_reference"])
    actuelles = paquet["pipeline"].predict_proba(preprocess(ref[VARIABLES_BRUTES]))[:, 1]
    ecart_max = float(np.max(np.abs(actuelles - ref["probabilite_defaut"])))
    assert ecart_max <= TOLERANCE_PROBABILITE, f"Écart maximal {ecart_max:.4f}"


def test_sorties_conformes_au_contrat(client):
    """Les demandes de référence, rejouées via l'API, respectent toujours le contrat."""
    for demande in BASELINE["demandes_reference"][:20]:
        corps = {v: demande[v] for v in VARIABLES_BRUTES}
        r = client.post("/predict", json=corps)
        assert r.status_code == 200
        assert 0 <= r.json()["probabilite_defaut"] <= 1


def test_coherence_apres_sauvegarde_et_rechargement(paquet, tmp_path):
    ref = preprocess(pd.DataFrame(BASELINE["demandes_reference"])[VARIABLES_BRUTES])
    chemin = tmp_path / "copie.joblib"
    joblib.dump(paquet, chemin)
    recharge = joblib.load(chemin)
    np.testing.assert_array_equal(paquet["pipeline"].predict_proba(ref),
                                  recharge["pipeline"].predict_proba(ref))