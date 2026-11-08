"""Génère un jeu de données SYNTHÉTIQUE de demandes de crédit (microfinance, FCFA).

Ces données servent uniquement à TESTER le système. Elles ne permettent pas de
démontrer la performance réelle d'un modèle de crédit.
"""

from pathlib import Path

import numpy as np
import pandas as pd

GRAINE = 42
N = 5_000


def generer(n: int = N, graine: int = GRAINE) -> pd.DataFrame:
    rng = np.random.default_rng(graine)
    age = rng.integers(18, 71, n)
    revenu = np.clip(rng.lognormal(np.log(200_000), 0.6, n), 30_000, 5_000_000).round(-3)
    montant = (revenu * rng.uniform(1, 12, n)).round(-4)
    duree = rng.choice([3, 6, 9, 12, 18, 24, 36], n)
    anciennete = np.minimum(rng.integers(0, 31, n), age - 16)

    mensualite = montant / duree
    taux_effort = mensualite / revenu
    score = (-3.0 + 3.2 * taux_effort - 0.06 * anciennete
             - 0.02 * (age - 18) + 0.03 * duree)
    proba = 1 / (1 + np.exp(-score))
    defaut = (rng.random(n) < proba).astype(int)

    return pd.DataFrame({
        "age": age, "revenu_mensuel": revenu, "montant_credit": montant,
        "duree_mois": duree, "anciennete_activite": anciennete, "defaut": defaut,
    })


if __name__ == "__main__":
    df = generer()
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/credit_synthetique.csv", index=False)
    print(f"{len(df)} demandes générées, taux de défaut : {df['defaut'].mean():.1%}")
    print(df.describe().round(1).to_string())