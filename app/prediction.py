"""Module de prédiction : chargement du modèle versionné et inférence."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from app.preprocessing import VARIABLES_MODELE, preprocess

SEUILS_RISQUE = {"FAIBLE": 0.30, "MODERE": 0.60}  # au-delà : ELEVE


class ModeleIndisponibleError(RuntimeError):
    """Le modèle n'est pas chargé ou ne peut pas l'être."""


class ServiceModele:
    def __init__(self):
        self.paquet = None

    @property
    def pret(self) -> bool:
        return self.paquet is not None

    def charger(self, chemin: str) -> None:
        if not Path(chemin).exists():
            raise ModeleIndisponibleError("Fichier modèle introuvable")
        try:
            paquet = joblib.load(chemin)
        except Exception as erreur:
            raise ModeleIndisponibleError("Fichier modèle illisible ou corrompu") from erreur
        if not isinstance(paquet, dict) or "pipeline" not in paquet:
            raise ModeleIndisponibleError("Format de modèle invalide")
        if paquet.get("sklearn_version") != sklearn.__version__:
            raise ModeleIndisponibleError("Version de scikit-learn incompatible")
        if list(paquet.get("variables", [])) != VARIABLES_MODELE:
            raise ModeleIndisponibleError("Variables du modèle incompatibles")
        self.paquet = paquet

    def _exiger_modele(self):
        if not self.pret:
            raise ModeleIndisponibleError("Modèle non chargé")

    def info(self) -> dict:
        self._exiger_modele()
        p = self.paquet
        return {"version": p["version"], "date_entrainement": p["date_entrainement"],
                "algorithme": p["algorithme"], "variables": p["variables"],
                "sklearn_version": p["sklearn_version"],
                "metriques_reference": p["metriques"]}

    def predire_proba(self, donnees: pd.DataFrame) -> np.ndarray:
        self._exiger_modele()
        X = preprocess(donnees)
        attendu = self.paquet["pipeline"].n_features_in_
        if X.shape[1] != attendu:
            raise ValueError(f"Dimensions incompatibles : {X.shape[1]} au lieu de {attendu}")
        return self.paquet["pipeline"].predict_proba(X)[:, 1]

    def alertes(self, donnees: pd.DataFrame) -> list[str]:
        """Entrées valides mais inhabituelles : hors de la plage d'entraînement.

        Contrôle les variables brutes ET dérivées (correction ANO-01 : une combinaison
        de valeurs individuellement normales peut produire un taux d'effort extrême).
        """
        self._exiger_modele()
        X = preprocess(donnees)
        plages = self.paquet["plages_entrainement"]
        return [f"{v} hors de la plage d'entraînement"
                for v, (mini, maxi) in plages.items()
                if not (mini <= float(X[v].iloc[0]) <= maxi)]


def niveau_risque(probabilite: float) -> str:
    if probabilite < SEUILS_RISQUE["FAIBLE"]:
        return "FAIBLE"
    if probabilite < SEUILS_RISQUE["MODERE"]:
        return "MODERE"
    return "ELEVE"