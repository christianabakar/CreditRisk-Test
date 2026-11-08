"""Tests fonctionnels de l'API (Partie II) : contrat HTTP de chaque endpoint."""

import math

import pytest

# ---------- /health et /model-info ----------


def test_health_modele_charge(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"statut": "ok", "modele_charge": True}


def test_model_info_metadonnees(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    info = r.json()
    assert info["version"] == "1.0.0"
    assert info["variables"][:5] == ["age", "revenu_mensuel", "montant_credit",
                                     "duree_mois", "anciennete_activite"]
    assert set(info["metriques_reference"]) >= {"recall", "f1"}


# ---------- POST /predict : scénarios obligatoires ----------

def test_s01_requete_valide(client, demande):
    r = client.post("/predict", json=demande)
    assert r.status_code == 200
    resultat = r.json()
    assert set(resultat) == {"probabilite_defaut", "niveau_risque", "version_modele", "alertes"}
    assert math.isfinite(resultat["probabilite_defaut"])
    assert 0 <= resultat["probabilite_defaut"] <= 1
    assert resultat["niveau_risque"] in {"FAIBLE", "MODERE", "ELEVE"}
    assert resultat["alertes"] == []


def test_s02_champ_obligatoire_absent(client, demande):
    del demande["age"]
    r = client.post("/predict", json=demande)
    assert r.status_code == 422
    assert r.json()["erreur"] == "donnees_invalides"
    assert r.json()["details"][0]["champ"] == "age"


def test_s03_age_negatif(client, demande):
    r = client.post("/predict", json={**demande, "age": -5})
    assert r.status_code == 422
    assert r.json()["details"][0]["champ"] == "age"


def test_s04_revenu_texte(client, demande):
    r = client.post("/predict", json={**demande, "revenu_mensuel": "250000"})
    assert r.status_code == 422
    assert r.json()["details"][0]["champ"] == "revenu_mensuel"
    assert "250000" not in r.text  # la valeur reçue n'est jamais renvoyée


@pytest.mark.parametrize("montant", [0, -1000000])
def test_s05_montant_nul_ou_negatif(client, demande, montant):
    r = client.post("/predict", json={**demande, "montant_credit": montant})
    assert r.status_code == 422
    assert r.json()["details"][0]["champ"] == "montant_credit"


def test_s06_duree_non_entiere(client, demande):
    r = client.post("/predict", json={**demande, "duree_mois": 12.5})
    assert r.status_code == 422
    assert r.json()["details"][0]["champ"] == "duree_mois"


def test_s07_champ_supplementaire(client, demande):
    r = client.post("/predict", json={**demande, "sexe": "F"})
    assert r.status_code == 422
    assert r.json()["details"][0]["champ"] == "sexe"


def test_s08_valeurs_extremes_valides(client, demande):
    """Inhabituelles mais valides : traitées (200) AVEC alerte."""
    r = client.post("/predict", json={**demande, "revenu_mensuel": 90_000_000,
                                      "montant_credit": 900_000_000})
    assert r.status_code == 200
    assert 0 <= r.json()["probabilite_defaut"] <= 1
    assert any("revenu_mensuel" in a for a in r.json()["alertes"])


def test_s08b_valeurs_au_dela_des_limites(client, demande):
    r = client.post("/predict", json={**demande, "revenu_mensuel": 1e12})
    assert r.status_code == 422


def test_s09_modele_indisponible(client_sans_modele, demande):
    r = client_sans_modele.post("/predict", json=demande)
    assert r.status_code == 503
    assert r.json()["erreur"] == "modele_indisponible"
    assert client_sans_modele.get("/health").status_code == 503
    assert client_sans_modele.get("/model-info").status_code == 503


def test_s10_requete_mal_formee(client):
    r = client.post("/predict", content=b'{"age": 35,',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert r.json()["erreur"] == "requete_mal_formee"


def test_s10b_corps_non_objet(client):
    r = client.post("/predict", json=[1, 2, 3])
    assert r.status_code == 422


def test_methode_non_autorisee(client):
    assert client.get("/predict").status_code == 405