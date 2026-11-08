# CreditRisk-Test

Système de prédiction du risque de défaut de crédit (microfinance) et sa stratégie de test.

> Les données de `data/` sont **synthétiques** : elles servent à tester le système et ne
> démontrent pas la performance réelle d'un modèle de crédit.

## Installation (Python 3.14)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Commandes

| Action | Commande |
|---|---|
| Générer les données synthétiques | `python -m scripts.generer_donnees` |
| Entraîner et versionner le modèle | `python -m scripts.entrainer_modele` |
| Lancer l'API | `uvicorn app.main:app` puis http://127.0.0.1:8000/docs |
| Style | `ruff check .` |
| Tests + couverture | `pytest --cov=app --cov-report=term-missing` |
| Figer la baseline (une seule fois) | `python -m scripts.creer_baseline` |
| Rapport de non-régression | `python -m scripts.comparer_versions` |
| Mesures de performance | `python -m scripts.mesurer_performance` |

## API

| Endpoint | Rôle | Codes |
|---|---|---|
| `GET /health` | Disponibilité du service | 200, 503 |
| `GET /model-info` | Métadonnées du modèle | 200, 503 |
| `POST /predict` | Probabilité de défaut et niveau de risque | 200, 400, 422, 500, 503 |