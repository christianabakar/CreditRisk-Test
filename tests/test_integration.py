"""Tests d'intégration (Partie III) : validation -> prétraitement -> modèle -> API."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.main import service
from app.prediction import ModeleIndisponibleError, ServiceModele, niveau_risque
from app.preprocessing import VARIABLES_MODELE, PreprocessingError, preprocess
from tests.conftest import DEMANDE_VALIDE

CHEMIN_MODELE = "models/model_v1.joblib"

# ---------- Scénario principal de bout en bout ----------


def test_transformation_des_variables():
    """Les variables dérivées sont calculées correctement (oracle calculé à la main)."""
    X = preprocess(pd.DataFrame([DEMANDE_VALIDE]))

    assert list(X.columns) == VARIABLES_MODELE
    assert X.loc[0, "mensualite"] == pytest.approx(1_000_000 / 12)
    assert X.loc[0, "taux_effort"] == pytest.approx((1_000_000 / 12) / 250_000)


def test_scenario_principal_bout_en_bout(client, demande):
    """Requête valide -> réponse conforme ET identique au calcul direct du pipeline."""
    r = client.post("/predict", json=demande)
    assert r.status_code == 200

    paquet = joblib.load(CHEMIN_MODELE)
    attendu = paquet["pipeline"].predict_proba(preprocess(pd.DataFrame([demande])))[0, 1]
    resultat = r.json()

    assert resultat["probabilite_defaut"] == pytest.approx(attendu, abs=1e-12)
    assert resultat["niveau_risque"] == niveau_risque(attendu)


def test_metadonnees_coherentes(client, demande):
    """La version renvoyée par /predict est celle annoncée par /model-info et le fichier."""
    info = client.get("/model-info").json()
    paquet = joblib.load(CHEMIN_MODELE)

    assert client.post("/predict", json=demande).json()["version_modele"] == info["version"]
    assert info["version"] == paquet["version"]
    assert info["variables"] == VARIABLES_MODELE
    assert info["metriques_reference"] == paquet["metriques"]


# ---------- Le modèle ne peut pas être chargé ----------

def test_fichier_modele_absent(tmp_path):
    with pytest.raises(ModeleIndisponibleError, match="introuvable"):
        ServiceModele().charger(str(tmp_path / "absent.joblib"))


def test_fichier_modele_corrompu(tmp_path):
    chemin = tmp_path / "corrompu.joblib"
    chemin.write_bytes(b"pas un modele")
    with pytest.raises(ModeleIndisponibleError, match="corrompu"):
        ServiceModele().charger(str(chemin))


def test_version_sklearn_incompatible(tmp_path):
    paquet = joblib.load(CHEMIN_MODELE)
    paquet["sklearn_version"] = "0.0.1"
    chemin = tmp_path / "ancien.joblib"
    joblib.dump(paquet, chemin)
    with pytest.raises(ModeleIndisponibleError, match="Version"):
        ServiceModele().charger(str(chemin))


def test_variables_du_modele_incompatibles(tmp_path):
    paquet = joblib.load(CHEMIN_MODELE)
    paquet["variables"] = ["age", "revenu_mensuel"]
    chemin = tmp_path / "variables.joblib"
    joblib.dump(paquet, chemin)
    with pytest.raises(ModeleIndisponibleError, match="Variables"):
        ServiceModele().charger(str(chemin))


# ---------- Variable absente / erreur de prétraitement ----------

def test_variable_attendue_absente():
    incomplet = pd.DataFrame([{k: v for k, v in DEMANDE_VALIDE.items() if k != "duree_mois"}])
    with pytest.raises(PreprocessingError, match="duree_mois"):
        preprocess(incomplet)


def test_pretraitement_valeur_non_finie():
    """Un revenu nul contourné (sans validation) produirait une division par zéro."""
    with pytest.raises(PreprocessingError, match="non finies"):
        preprocess(pd.DataFrame([{**DEMANDE_VALIDE, "revenu_mensuel": 0}]))


def test_erreur_pretraitement_via_api(client_erreurs, demande, monkeypatch):
    def pretraitement_defaillant(_):
        raise PreprocessingError("détail interne confidentiel")

    monkeypatch.setattr("app.prediction.preprocess", pretraitement_defaillant)
    r = client_erreurs.post("/predict", json=demande)

    assert r.status_code == 500
    assert r.json()["erreur"] == "erreur_interne"
    assert "confidentiel" not in r.text


# ---------- Dimensions incompatibles ----------

def test_dimensions_incompatibles(client_erreurs, demande, monkeypatch):
    """Un modèle attendant 5 variables au lieu de 7 est détecté et l'erreur est contenue."""
    mauvais = LogisticRegression().fit(np.random.default_rng(0).normal(size=(20, 5)),
                                       [0, 1] * 10)
    monkeypatch.setitem(service.paquet, "pipeline", mauvais)

    with pytest.raises(ValueError, match="Dimensions incompatibles"):
        service.predire_proba(pd.DataFrame([demande]))
    r = client_erreurs.post("/predict", json=demande)
    assert r.status_code == 500
    assert "Dimensions" not in r.text


# ---------- Erreur interne pendant la prédiction ----------

def test_erreur_interne_sans_trace_technique(client_erreurs, demande, monkeypatch):
    def panne(_):
        raise RuntimeError("Traceback: /srv/app/secret_config.py ligne 42")

    monkeypatch.setattr(service.paquet["pipeline"], "predict_proba", panne)
    r = client_erreurs.post("/predict", json=demande)

    assert r.status_code == 500
    corps = r.json()
    assert set(corps) == {"erreur", "identifiant", "message"}
    assert len(corps["identifiant"]) == 12  # permet de retrouver l'erreur dans les journaux
    for fuite in ["Traceback", "RuntimeError", "secret_config", ".py"]:
        assert fuite not in r.text