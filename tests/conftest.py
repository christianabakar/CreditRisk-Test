import pytest
from fastapi.testclient import TestClient

from app.main import app, get_service
from app.prediction import ServiceModele

DEMANDE_VALIDE = {"age": 35, "revenu_mensuel": 250000, "montant_credit": 1000000,
                  "duree_mois": 12, "anciennete_activite": 4}


@pytest.fixture
def demande():
    """Une copie fraîche de la demande valide de référence (énoncé)."""
    return dict(DEMANDE_VALIDE)


@pytest.fixture(scope="session")
def client():
    """Client de test : l'API démarre et charge le modèle versionné."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_sans_modele():
    """API dont le service de modèle est vide (modèle non chargé)."""
    app.dependency_overrides[get_service] = lambda: ServiceModele()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()