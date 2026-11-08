"""Module de prétraitement : variables brutes -> variables attendues par le modèle."""

import numpy as np
import pandas as pd

VARIABLES_BRUTES = ["age", "revenu_mensuel", "montant_credit", "duree_mois",
                    "anciennete_activite"]
VARIABLES_MODELE = VARIABLES_BRUTES + ["mensualite", "taux_effort"]


class PreprocessingError(ValueError):
    """Le prétraitement n'a pas pu produire des variables valides."""


def preprocess(donnees: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables dérivées et renvoie exactement VARIABLES_MODELE."""
    manquantes = [v for v in VARIABLES_BRUTES if v not in donnees.columns]
    if manquantes:
        raise PreprocessingError(f"Variables manquantes : {manquantes}")

    X = donnees[VARIABLES_BRUTES].astype(float).copy()
    X["mensualite"] = X["montant_credit"] / X["duree_mois"]
    X["taux_effort"] = X["mensualite"] / X["revenu_mensuel"]

    if not np.isfinite(X.to_numpy()).all():
        raise PreprocessingError("Le prétraitement a produit des valeurs non finies")
    return X[VARIABLES_MODELE]