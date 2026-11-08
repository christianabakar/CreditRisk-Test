"""Tests de robustesse (Partie IV) : entrées limites, invalides et inhabituelles."""

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.conftest import DEMANDE_VALIDE

CHAMPS = list(DEMANDE_VALIDE)


def poster_brut(client, corps: bytes):
    return client.post("/predict", content=corps, headers={"Content-Type": "application/json"})


# ---------- Entrées INVALIDES : doivent être rejetées (422) ----------

@pytest.mark.parametrize("champ", CHAMPS)
def test_valeur_manquante_null(client, demande, champ):
    r = client.post("/predict", json={**demande, champ: None})
    assert r.status_code == 422
    assert r.json()["details"][0]["champ"] == champ


@pytest.mark.parametrize("valeur", ["abc", "", [], {}, True])
@pytest.mark.parametrize("champ", CHAMPS)
def test_valeur_non_numerique(client, demande, champ, valeur):
    assert client.post("/predict", json={**demande, champ: valeur}).status_code == 422


@pytest.mark.parametrize("jeton", ["NaN", "Infinity", "-Infinity"])
def test_valeur_non_finie(client, jeton):
    """JSON non standard accepté par le parseur Python : la validation doit l'arrêter."""
    corps = ('{"age": 35, "revenu_mensuel": ' + jeton + ', "montant_credit": 1000000, '
             '"duree_mois": 12, "anciennete_activite": 4}').encode()
    r = poster_brut(client, corps)
    assert r.status_code == 422
    assert r.json()["details"][0]["champ"] == "revenu_mensuel"


@pytest.mark.parametrize("corps", [
    b'[{"age": 35}]',                                           # tableau au lieu d'objet
    b'{"demande": {"age": 35}}',                                 # objet imbriqué
    b'{"Age": 35, "Revenu": 250000}',                            # noms de champs incorrects
    b'"texte"',                                                  # chaîne JSON
    b'{}',                                                       # objet vide
])
def test_schema_incorrect(client, corps):
    assert poster_brut(client, corps).status_code == 422


# ---------- Valeurs AUX LIMITES des intervalles autorisés ----------

@pytest.mark.parametrize("modif, code", [
    ({"age": 18, "anciennete_activite": 2}, 200), ({"age": 17, "anciennete_activite": 0}, 422),
    ({"age": 75}, 200), ({"age": 76}, 422),
    ({"duree_mois": 1}, 200), ({"duree_mois": 0}, 422),
    ({"duree_mois": 120}, 200), ({"duree_mois": 121}, 422),
    ({"revenu_mensuel": 100_000_000}, 200), ({"revenu_mensuel": 100_000_001}, 422),
    ({"montant_credit": 1_000_000_000}, 200), ({"montant_credit": 1_000_000_001}, 422),
    ({"anciennete_activite": 0}, 200), ({"anciennete_activite": -1}, 422),
])
def test_valeurs_limites(client, demande, modif, code):
    assert client.post("/predict", json={**demande, **modif}).status_code == code


# ---------- Entrées INHABITUELLES mais VALIDES : traitées AVEC alerte ----------

@pytest.mark.parametrize("modif, variable", [
    ({"revenu_mensuel": 50_000_000}, "revenu_mensuel"),
    ({"montant_credit": 500_000_000}, "montant_credit"),
    ({"duree_mois": 120}, "duree_mois"),
    ({"age": 75, "anciennete_activite": 40}, "anciennete_activite"),
])
def test_entree_eloignee_signalee(client, demande, modif, variable):
    r = client.post("/predict", json={**demande, **modif})
    assert r.status_code == 200
    assert any(variable in a for a in r.json()["alertes"])


def test_combinaison_inhabituelle_signalee(client):
    """Chaque variable est dans sa plage, mais le TAUX D'EFFORT (50) est 12 fois
    supérieur au maximum vu à l'entraînement (4) : le modèle extrapole."""
    demande = {"age": 35, "revenu_mensuel": 100_000, "montant_credit": 15_000_000,
               "duree_mois": 3, "anciennete_activite": 4}
    r = client.post("/predict", json=demande)
    assert r.status_code == 200
    assert any("taux_effort" in a for a in r.json()["alertes"])


# ---------- Contrat et cohérence des prédictions ----------

@st.composite
def demandes_valides(draw):
    age = draw(st.integers(18, 75))
    return {"age": age,
            "revenu_mensuel": draw(st.floats(1, 100_000_000)),
            "montant_credit": draw(st.floats(1, 1_000_000_000)),
            "duree_mois": draw(st.integers(1, 120)),
            "anciennete_activite": draw(st.integers(0, min(60, age - 16)))}


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(demande=demandes_valides())
def test_toute_demande_valide_respecte_le_contrat(client, demande):
    """Propriété : toute demande valide -> 200, probabilité finie dans [0, 1]."""
    r = client.post("/predict", json=demande)
    assert r.status_code == 200
    probabilite = r.json()["probabilite_defaut"]
    assert math.isfinite(probabilite)
    assert 0.0 <= probabilite <= 1.0


def test_predictions_deterministes(client, demande):
    p1 = client.post("/predict", json=demande).json()["probabilite_defaut"]
    p2 = client.post("/predict", json=demande).json()["probabilite_defaut"]
    assert p1 == p2


@pytest.mark.parametrize("profil", [
    {"age": 25, "revenu_mensuel": 80_000, "duree_mois": 6, "anciennete_activite": 2},
    {"age": 35, "revenu_mensuel": 250_000, "duree_mois": 12, "anciennete_activite": 4},
    {"age": 55, "revenu_mensuel": 1_000_000, "duree_mois": 36, "anciennete_activite": 20},
])
def test_coherence_metier_montant(client, profil):
    """Test métamorphique : à profil égal, demander plus ne doit pas réduire le risque."""
    probas = [client.post("/predict", json={**profil, "montant_credit": m}).json()["probabilite_defaut"]
              for m in [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]]
    assert probas == sorted(probas)