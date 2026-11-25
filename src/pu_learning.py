"""
Étape 3d : PU learning (Positive-Unlabeled) pour étiqueter les candidates de selection_pot.

Données (pu/features_augmentees.parquet, toutes évaluées par le même Stockfish) :
  * P : parties du mois (étiquetées positives, s = 1) ;
  * U : groupe témoin (non étiqueté, s = 0) : un échantillon de la population, qui contient en théorie quelques
        parties remarquables mais surtout des parties ordinaires ;
  * candidates : parties de selection_pot, à étiqueter.

Méthode (Elkan et Noto, 2008, « Learning classifiers from only positive and unlabeled data ») :
  1. un classifieur probabiliste g(x) ≈ P(s = 1 | x) est entraîné à séparer P de U ;
  2. sous l'hypothèse que les positifs étiquetés sont tirés au hasard parmi tous les positifs, g(x) = c · P(y = 1 | x),
     où c = P(s = 1 | y = 1) s'estime par la moyenne de g sur des positifs NON vus à l'entraînement (validation
     croisée) ;
  3. pour une candidate : P(y = 1 | x) = g(x) / c (borné à 1) ; elle est étiquetée positive si cette probabilité
     dépasse --seuil (0,5 par défaut).

Classifieur par défaut : régression logistique régularisée (peu de positifs : un modèle simple généralise mieux),
force de régularisation choisie par validation croisée (AUC). Option : --modele hgb (gradient boosting).
Features : celles de features_core, sans les variables de contexte (elo, n_plies, rarete_ouverture : les parties
du mois sont jouées par des joueurs bien plus forts, le modèle apprendrait le niveau et non l'intérêt) ;
piece_mateuse est codée en indicateurs 0/1 ; transformation log signée, valeurs manquantes -> médiane, standardisation.

Diagnostics : AUC hors échantillon (P contre U), estimation de c, part du témoin qui serait étiquetée positive
(estimation du taux de faux positifs dans une population ordinaire), coefficients du modèle.

Sorties (dans <dossier>/pu/) :
  pu_candidats.parquet / .csv  candidates avec g, probabilité PU, étiquette (positif), infos de selection_pot
  pu_scores.parquet            scores hors échantillon de P et U (pour les graphiques)
  pu_coefficients.csv          poids des features (régression logistique)
  pu_resume.json               chiffres clés, dont le nombre de candidates étiquetées positives
  pu_scores.png                distributions des probabilités des trois groupes

Exemple :
    python src/pu_learning.py
    python src/pu_learning.py --seuil 0.7 --modele hgb
"""
import argparse
import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402

try:                                                 # permet d'importer les modules placés à côté de ce fichier
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
from features_core import FEATURES_PARTIE  # noqa: E402
from outils import DOSSIER, chemin  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")   # changements d'API annoncés

CONTEXTE = ("n_plies", "elo", "rarete_ouverture")
SEUIL = 0.5
CS = np.logspace(-3, 1, 9)           # forces de régularisation essayées (régression logistique)
GRAINE = 0


def log_signe(X):
    return np.sign(X) * np.log1p(np.abs(X))


def matrice(df, exclure):
    """Features numériques (contexte et exclusions retirés) + indicateurs de la pièce mateuse."""
    cols = [f for f in FEATURES_PARTIE if f in df and f != "piece_mateuse" and f not in CONTEXTE and f not in exclure]
    X = df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    if "piece_mateuse" in df:
        for p in "PNBRQK":
            X[f"mat_{p}"] = (df["piece_mateuse"].fillna("") == p).astype(float)
    return X


def modele(nom, cv_interne, graine):
    etapes = [("log", FunctionTransformer(log_signe)), ("imputation", SimpleImputer(strategy="median")),
              ("standard", StandardScaler())]
    if nom == "hgb":
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                             min_samples_leaf=10, random_state=graine)
    elif cv_interne >= 2:
        clf = LogisticRegressionCV(Cs=CS, cv=StratifiedKFold(cv_interne, shuffle=True, random_state=graine),
                                   scoring="roc_auc", max_iter=5000)
    else:
        clf = LogisticRegression(C=0.1, max_iter=5000)
    return Pipeline(etapes + [("clf", clf)])


def tracer(scores, cand, seuil, chemin_png):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bords = np.linspace(0, 1, 41)
    for nom, x, coul in (("témoin (U, hors échantillon)", scores.loc[scores.groupe == "temoin", "p_pu"], "#9e9e9e"),
                         ("parties du mois (P, hors échantillon)", scores.loc[scores.groupe == "positif", "p_pu"],
                          "#2a7de1"),
                         ("candidates de selection_pot", cand["p_pu"], "#e8710a")):
        if len(x):
            ax.hist(x, bins=bords, alpha=0.55, color=coul, label=f"{nom} (n={len(x):,})",
                    weights=np.full(len(x), 1 / len(x)))
    ax.axvline(seuil, color="k", ls="--", lw=1.2, label=f"seuil = {seuil:g}")
    ax.set_xlabel("probabilité d'être une partie remarquable, P(y=1|x) = g(x)/c")
    ax.set_ylabel("part du groupe")
    ax.set_yscale("log")
    ax.set_title("PU learning : distributions des probabilités par groupe")
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(chemin_png, dpi=110)
    plt.close(fig)


def main(argv=None):
    a = argparse.ArgumentParser(description="PU learning : étiquette les candidates de selection_pot.")
    a.add_argument("--dossier", default=DOSSIER, help="dossier des données (défaut : %(default)s)")
    a.add_argument("--entree", default="pu/features_augmentees.parquet")
    a.add_argument("--sortie", default="pu", help="sous-dossier de sortie (défaut : %(default)s)")
    a.add_argument("--seuil", type=float, default=SEUIL, help="probabilité minimale pour étiqueter positif")
    a.add_argument("--modele", choices=["logistique", "hgb"], default="logistique")
    a.add_argument("--exclure", default="", help="features à exclure, séparées par des virgules")
    a.add_argument("--plis", type=int, default=5, help="plis de la validation croisée (défaut : %(default)s)")
    a.add_argument("--graine", type=int, default=GRAINE)
    o = a.parse_args(argv)

    t0 = time.perf_counter()
    entree = chemin(o.dossier, o.entree)
    if not os.path.exists(entree):
        sys.exit(f"Fichier introuvable : {entree}\nLancez d'abord src/augmentation.py.")
    df = pd.read_parquet(entree)
    df = df[df["acpl_gagnant"].notna()].reset_index(drop=True)          # parties évaluées par Stockfish
    X = matrice(df, {s for s in o.exclure.split(",") if s})
    pos, tem, cand = (df["source"] == "positif").to_numpy(), (df["source"] == "temoin").to_numpy(), \
        (df["source"] == "candidat").to_numpy()
    n_p, n_u, n_c = int(pos.sum()), int(tem.sum()), int(cand.sum())
    print(f"P (parties du mois) : {n_p} | U (témoin) : {n_u:,} | candidates : {n_c:,} | features : {X.shape[1]}")
    if n_p < 5 or n_u < 20:
        sys.exit("Trop peu de parties du mois ou de parties témoin pour apprendre (au moins 5 et 20).")
    variables = [c for c in X if X.loc[pos | tem, c].nunique(dropna=True) > 1]
    X = X[variables]

    # --- 1. g(x) = P(s=1|x) hors échantillon, par validation croisée stratifiée
    Xt, s = X[pos | tem], pos[pos | tem].astype(int)
    plis = max(2, min(o.plis, n_p))
    cv_interne = min(3, (n_p * (plis - 1)) // plis)
    cv = StratifiedKFold(plis, shuffle=True, random_state=o.graine)
    oof = cross_val_predict(modele(o.modele, cv_interne, o.graine), Xt, s, cv=cv, method="predict_proba")[:, 1]
    auc, ap = roc_auc_score(s, oof), average_precision_score(s, oof)

    # --- 2. c = P(s=1 | y=1) : moyenne de g sur les positifs non vus (Elkan et Noto)
    c = float(oof[s == 1].mean())

    # --- 3. modèle final, appliqué aux candidates : P(y=1|x) = g(x) / c
    final = modele(o.modele, cv_interne, o.graine).fit(Xt, s)
    g_cand = final.predict_proba(X[cand])[:, 1]
    p_cand = np.clip(g_cand / c, 0, 1)
    p_oof = np.clip(oof / c, 0, 1)
    taux_temoin = float((p_oof[s == 0] >= o.seuil).mean())
    rappel_p = float((p_oof[s == 1] >= o.seuil).mean())

    rep = chemin(o.dossier, o.sortie)
    os.makedirs(rep, exist_ok=True)
    garder = ["Site", "ECO", "Opening", "Result", "elo", "n_plies", "groupe_ref", "d2", "p_pot", "rarete",
              "rang_groupe", "features_extremes"]
    res = df.loc[cand, [k for k in garder if k in df]].reset_index(drop=True)
    res["g"], res["p_pu"], res["positif"] = g_cand, p_cand, p_cand >= o.seuil
    res = res.sort_values("p_pu", ascending=False).reset_index(drop=True)
    res.to_parquet(os.path.join(rep, "pu_candidats.parquet"))
    res.to_csv(os.path.join(rep, "pu_candidats.csv"), index=False)
    scores = pd.DataFrame({"groupe": np.where(s == 1, "positif", "temoin"), "g_oof": oof, "p_pu": p_oof,
                           "Site": df.loc[pos | tem, "Site"].to_numpy()})
    scores.to_parquet(os.path.join(rep, "pu_scores.parquet"))
    tracer(scores, res, o.seuil, os.path.join(rep, "pu_scores.png"))

    coef = None
    clf = final.named_steps["clf"]
    if hasattr(clf, "coef_"):
        coef = pd.DataFrame({"feature": variables, "coefficient": np.ravel(clf.coef_)})
        coef = coef.reindex(coef["coefficient"].abs().sort_values(ascending=False).index)
        coef.to_csv(os.path.join(rep, "pu_coefficients.csv"), index=False)
    n_pos = int(res["positif"].sum())
    resume = dict(n_parties_du_mois=n_p, n_temoin=n_u, n_candidates=n_c, n_features=len(variables),
                  modele=o.modele, C=float(np.ravel(clf.C_)[0]) if hasattr(clf, "C_") else None, auc_hors_echantillon=auc,
                  precision_moyenne=ap, c=c, seuil=o.seuil, n_candidates_positives=n_pos,
                  part_temoin_etiquetee_positive=taux_temoin, part_parties_du_mois_retrouvees=rappel_p)
    with open(os.path.join(rep, "pu_resume.json"), "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, ensure_ascii=False)

    print(f"\nAUC hors échantillon (parties du mois contre témoin) : {auc:.3f} | précision moyenne : {ap:.3f}")
    print(f"c = P(s=1|y=1) estimé : {c:.3f}" + (f" | régularisation choisie C = {resume['C']:.3g}" if resume["C"] else ""))
    print(f"Parties du mois retrouvées (hors échantillon, p >= {o.seuil:g}) : {100 * rappel_p:.0f} %")
    print(f"Part du témoin étiquetée positive (faux positifs attendus dans une population ordinaire) : "
          f"{100 * taux_temoin:.2f} %")
    if coef is not None:
        print("\nFeatures les plus influentes :")
        print(coef.head(10).round(3).to_string(index=False))
    print(f"\n==> {n_pos} candidates sur {n_c:,} étiquetées positives (p >= {o.seuil:g})")
    if auc < 0.7:
        print("  ATTENTION : AUC faible, le modèle distingue mal les parties du mois du témoin.")
    print(f"Fichiers écrits dans {rep} ({time.perf_counter() - t0:.1f} s)")


if __name__ == "__main__":
    main()
