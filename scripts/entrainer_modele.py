"""Entraîne et versionne le modèle de test (models/model_v1.joblib).

Usage : python -m scripts.entrainer_modele [--version 1.0.0] [--C 1.0] [--sortie models/model_v1.joblib]
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.preprocessing import VARIABLES_BRUTES, VARIABLES_MODELE, preprocess

GRAINE = 42


def charger_et_decouper(chemin="data/credit_synthetique.csv"):
    df = pd.read_csv(chemin)
    X, y = df[VARIABLES_BRUTES], df["defaut"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=GRAINE)
    return X_train, X_test, y_train, y_test


def entrainer(X_train, y_train, C=1.0, class_weight="balanced"):
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("modele", LogisticRegression(C=C, class_weight=class_weight,
                                                       max_iter=1000, random_state=GRAINE))])
    return pipeline.fit(preprocess(X_train), y_train)


def evaluer(pipeline, X_test, y_test, seuil=0.5):
    proba = pipeline.predict_proba(preprocess(X_test))[:, 1]
    pred = (proba >= seuil).astype(int)
    return {"recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
            "precision": precision_score(y_test, pred), "roc_auc": roc_auc_score(y_test, proba)}


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--version", default="1.1.0")
    args.add_argument("--C", type=float, default=1.0)
    args.add_argument("--class-weight", default="balanced")
    args.add_argument("--sortie", default="models/model_v1.joblib")
    a = args.parse_args()
    cw = None if a.class_weight == "none" else a.class_weight

    X_train, X_test, y_train, y_test = charger_et_decouper()
    pipeline = entrainer(X_train, y_train, C=a.C, class_weight=cw)
    X_pre = preprocess(X_train)
    metriques = {k: round(float(v), 4) for k, v in evaluer(pipeline, X_test, y_test).items()}

    paquet = {
        "pipeline": pipeline, "version": a.version, "algorithme": "LogisticRegression",
        "parametres": {"C": a.C, "class_weight": a.class_weight},
        "date_entrainement": datetime.now(UTC).date().isoformat(),
        "variables": VARIABLES_MODELE, "sklearn_version": sklearn.__version__,
        "graine": GRAINE, "metriques": metriques,
        "plages_entrainement": {v: (float(X_pre[v].min()), float(X_pre[v].max()))
                                for v in VARIABLES_MODELE},
    }
    Path(a.sortie).parent.mkdir(exist_ok=True)
    joblib.dump(paquet, a.sortie)
    print(f"Modèle {a.version} sauvegardé dans {a.sortie}")
    print(json.dumps(metriques, indent=2))