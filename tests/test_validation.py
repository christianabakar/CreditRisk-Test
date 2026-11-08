import pytest
from pydantic import ValidationError

from app.validation import DemandeCredit
from tests.conftest import DEMANDE_VALIDE


def valider(**modifs):
    return DemandeCredit.model_validate({**DEMANDE_VALIDE, **modifs})


def test_demande_valide_acceptee():
    d = valider()
    assert d.age == 35 and d.revenu_mensuel == 250000.0


# ---------- EX01 : âge dans [18 ; 75] — valeurs limites ----------

@pytest.mark.parametrize("age", [18, 75])
def test_age_aux_bornes_accepte(age):
    assert valider(age=age, anciennete_activite=0).age == age


@pytest.mark.parametrize("age", [17, 76, -5, 0])
def test_age_hors_bornes_rejete(age):
    with pytest.raises(ValidationError, match="age"):
        valider(age=age, anciennete_activite=0)


# ---------- EX02 : revenu numérique et strictement positif ----------

@pytest.mark.parametrize("revenu", [0, -1, -250000])
def test_revenu_nul_ou_negatif_rejete(revenu):
    with pytest.raises(ValidationError, match="revenu_mensuel"):
        valider(revenu_mensuel=revenu)


@pytest.mark.parametrize("revenu", ["250000", "abc", True, None])
def test_revenu_non_numerique_rejete(revenu):
    """Mode strict : "250000" n'est PAS converti silencieusement en nombre."""
    with pytest.raises(ValidationError, match="revenu_mensuel"):
        valider(revenu_mensuel=revenu)


def test_revenu_minimal_positif_accepte():
    assert valider(revenu_mensuel=0.01).revenu_mensuel == 0.01


# ---------- EX03 : durée entière dans [1 ; 120] ----------

@pytest.mark.parametrize("duree", [1, 120])
def test_duree_aux_bornes_acceptee(duree):
    assert valider(duree_mois=duree).duree_mois == duree


@pytest.mark.parametrize("duree", [0, 121, -12, 12.5, 12.0, "12"])
def test_duree_invalide_rejetee(duree):
    with pytest.raises(ValidationError, match="duree_mois"):
        valider(duree_mois=duree)


# ---------- Montant, ancienneté, cohérence ----------

@pytest.mark.parametrize("montant", [0, -1000])
def test_montant_nul_ou_negatif_rejete(montant):
    with pytest.raises(ValidationError, match="montant_credit"):
        valider(montant_credit=montant)


def test_anciennete_coherente_avec_age_limite():
    """Valeur limite : 30 ans -> 14 ans d'ancienneté maximum (activité dès 16 ans)."""
    assert valider(age=30, anciennete_activite=14).anciennete_activite == 14
    with pytest.raises(ValidationError, match="incohérente"):
        valider(age=30, anciennete_activite=15)


# ---------- EX06 : complétude et champs non autorisés ----------

@pytest.mark.parametrize("champ", list(DEMANDE_VALIDE))
def test_chaque_champ_est_obligatoire(champ):
    incomplet = {k: v for k, v in DEMANDE_VALIDE.items() if k != champ}
    with pytest.raises(ValidationError, match=champ):
        DemandeCredit.model_validate(incomplet)


def test_champ_supplementaire_rejete():
    with pytest.raises(ValidationError, match="sexe"):
        valider(sexe="F")