"""API de prédiction du risque de défaut."""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.prediction import ModeleIndisponibleError, ServiceModele, niveau_risque
from app.validation import DemandeCredit, ReponsePrediction

logger = logging.getLogger(__name__)
CHEMIN_MODELE = os.environ.get("CREDITRISK_MODEL_PATH", "models/model_v1.joblib")
service = ServiceModele()


@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    try:
        service.charger(CHEMIN_MODELE)
        logger.info("Modèle chargé : %s", service.paquet["version"])
    except ModeleIndisponibleError as erreur:
        logger.error("Démarrage sans modèle : %s", erreur)  # l'API démarre en mode dégradé
    yield


app = FastAPI(title="CreditRisk API", version="1.0.0", lifespan=cycle_de_vie)


def get_service() -> ServiceModele:
    return service


Service = Annotated[ServiceModele, Depends(get_service)]


# ---------- Gestion des erreurs : jamais de trace technique côté client ----------

@app.exception_handler(RequestValidationError)
async def erreur_validation(request: Request, exc: RequestValidationError):
    erreurs = exc.errors()
    if any(e["type"] == "json_invalid" for e in erreurs):
        return JSONResponse(status_code=400, content={
            "erreur": "requete_mal_formee", "message": "Le corps de la requête n'est pas un JSON valide"})
    details = [{"champ": ".".join(str(x) for x in e["loc"][1:]) or "corps",
                "message": e["msg"]} for e in erreurs]  # la valeur reçue n'est PAS renvoyée
    return JSONResponse(status_code=422, content={"erreur": "donnees_invalides", "details": details})


@app.exception_handler(ModeleIndisponibleError)
async def erreur_modele(request: Request, exc: ModeleIndisponibleError):
    return JSONResponse(status_code=503, content={
        "erreur": "modele_indisponible", "message": "Le service de prédiction est temporairement indisponible"})


@app.exception_handler(Exception)
async def erreur_interne(request: Request, exc: Exception):
    identifiant = uuid.uuid4().hex[:12]
    logger.exception("Erreur interne [%s]", identifiant)  # détail côté serveur uniquement
    return JSONResponse(status_code=500, content={
        "erreur": "erreur_interne", "identifiant": identifiant,
        "message": "Une erreur interne est survenue"})


# ---------- Endpoints ----------

@app.get("/health")
def health(s: Service):
    if not s.pret:
        return JSONResponse(status_code=503, content={"statut": "degrade", "modele_charge": False})
    return {"statut": "ok", "modele_charge": True}


@app.get("/model-info")
def model_info(s: Service):
    return s.info()


@app.post("/predict", response_model=ReponsePrediction)
def predict(demande: DemandeCredit, s: Service):
    donnees = pd.DataFrame([demande.model_dump()])
    probabilite = float(s.predire_proba(donnees)[0])
    logger.info("Prédiction effectuée (niveau %s)", niveau_risque(probabilite))
    return ReponsePrediction(probabilite_defaut=probabilite,
                             niveau_risque=niveau_risque(probabilite),
                             version_modele=s.paquet["version"],
                             alertes=s.alertes(donnees))