"""Mesures de performance (Partie VI) : latence d'inférence, temps de réponse de l'API,
P95, débit, et comportement sous charge progressive. L'API est lancée pour de vrai
(uvicorn, HTTP local) afin d'inclure le traitement HTTP et la validation."""

import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd
import sklearn

from app.prediction import ServiceModele
from app.preprocessing import preprocess

PORT = 8001
URL = f"http://127.0.0.1:{PORT}"
DEMANDE = {"age": 35, "revenu_mensuel": 250000, "montant_credit": 1000000,
           "duree_mois": 12, "anciennete_activite": 4}
OBJECTIFS = {"p95_api_ms": 200, "debit_min_req_s": 20, "taux_erreur_max": 0.0}


def stats_ms(durees):
    d = np.array(durees) * 1000
    return {"moyenne": round(float(d.mean()), 2), "p50": round(float(np.percentile(d, 50)), 2),
            "p95": round(float(np.percentile(d, 95)), 2), "p99": round(float(np.percentile(d, 99)), 2)}


def chrono(fonction, n, echauffement=50):
    for _ in range(echauffement):
        fonction()
    durees = []
    for _ in range(n):
        t = time.perf_counter()
        fonction()
        durees.append(time.perf_counter() - t)
    return durees


# ---------- 1. Latence d'inférence (sans HTTP) ----------
service = ServiceModele()
service.charger("models/model_v1.joblib")
df = pd.DataFrame([DEMANDE])
X = preprocess(df)
modele_seul = stats_ms(chrono(lambda: service.paquet["pipeline"].predict_proba(X), 1000))
pipeline_complet = stats_ms(chrono(lambda: service.predire_proba(df), 1000))

# ---------- 2. API réelle : démarrage d'uvicorn ----------
serveur = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app",
                            "--port", str(PORT), "--log-level", "warning"])
try:
    for _ in range(100):
        try:
            if httpx.get(f"{URL}/health", timeout=1).status_code == 200:
                break
        except httpx.TransportError:
            time.sleep(0.1)

    with httpx.Client(base_url=URL) as client:
        api_sequentiel = stats_ms(chrono(lambda: client.post("/predict", json=DEMANDE), 500))

    # ---------- 3. Charge progressive : 1 à 16 utilisateurs simultanés ----------
    def utilisateur(n_requetes):
        durees, erreurs = [], 0
        with httpx.Client(base_url=URL) as c:
            for _ in range(n_requetes):
                t = time.perf_counter()
                r = c.post("/predict", json=DEMANDE)
                durees.append(time.perf_counter() - t)
                erreurs += r.status_code != 200
        return durees, erreurs

    paliers = []
    for concurrents in [1, 2, 4, 8, 16]:
        debut = time.perf_counter()
        with ThreadPoolExecutor(concurrents) as pool:
            resultats = list(pool.map(utilisateur, [100] * concurrents))
        duree_totale = time.perf_counter() - debut
        durees = [d for r in resultats for d in r[0]]
        erreurs = sum(r[1] for r in resultats)
        paliers.append({"utilisateurs": concurrents, "requetes": len(durees),
                        "debit_req_s": round(len(durees) / duree_totale, 1),
                        "taux_erreur": erreurs / len(durees), **stats_ms(durees)})
finally:
    serveur.terminate()
    serveur.wait()

resultats = {
    "configuration": {"systeme": f"{platform.system()} {platform.release()}",
                      "processeur": platform.processor(), "coeurs_logiques": os.cpu_count(),
                      "python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "serveur": "uvicorn, 1 processus, HTTP local 127.0.0.1",
                      "methode": "time.perf_counter, 50 itérations d'échauffement exclues"},
    "objectifs": OBJECTIFS,
    "inference_modele_seul_ms": modele_seul, "inference_pretraitement_compris_ms": pipeline_complet,
    "api_sequentielle_ms": api_sequentiel, "charge_progressive": paliers,
}
with open("reports/performance.json", "w", encoding="utf-8") as f:
    json.dump(resultats, f, indent=2, ensure_ascii=False)

c = resultats["configuration"]
print(f"{c['systeme']} | {c['processeur']} | {c['coeurs_logiques']} coeurs | Python {c['python']}\n")
print("LATENCE (ms)                          moyenne     P50     P95     P99")
for nom, s in [("Inférence, modèle seul", modele_seul),
               ("Inférence + prétraitement", pipeline_complet),
               ("API HTTP complète (séquentiel)", api_sequentiel)]:
    print(f"  {nom:34} {s['moyenne']:7.2f} {s['p50']:7.2f} {s['p95']:7.2f} {s['p99']:7.2f}")
print("\nCHARGE PROGRESSIVE   débit (req/s)   P50 (ms)   P95 (ms)   erreurs")
for p in paliers:
    print(f"  {p['utilisateurs']:>2} utilisateur(s) {p['debit_req_s']:>14} {p['p50']:>10} "
          f"{p['p95']:>10} {100 * p['taux_erreur']:>8.1f} %")
print(f"\nObjectifs : P95 API <= {OBJECTIFS['p95_api_ms']} ms | débit >= "
      f"{OBJECTIFS['debit_min_req_s']} req/s | 0 % d'erreur")
print("Résultats écrits dans reports/performance.json")