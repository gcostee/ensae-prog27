"""
Étape 3 : sélection des parties atypiques, code ECO par code ECO.

Pour chaque groupe d'ouverture (code ECO) :
  1. les dizaines de features sont ramenées à UNE métrique : la distance de Mahalanobis robuste (d²) de chaque
     partie au centre de son groupe, en tenant compte des corrélations entre features ;
  2. la queue de la distribution de d² est modélisée par la méthode POT (Peaks Over Threshold : loi de Pareto
     généralisée ajustée sur les dépassements d'un seuil élevé) ;
  3. chaque partie reçoit une p-valeur de rareté (probabilité qu'une partie du même groupe soit au moins
     aussi atypique) ; les parties dont p <= --p-seuil sont retenues.

Entrée  : <dossier>/resultats/features.parquet (sortie de analyse_features.py)
Sorties : <dossier>/resultats/selection/
            parties_remarquables.csv / .parquet   parties retenues, triées par ECO puis par rareté
            resume_par_groupe.csv                 une ligne par groupe : taille, seuil POT, forme, qualité, nb retenues
            scores.parquet                        (option --ecrire-scores) d² et rareté de toutes les parties

Exemple :
    python src/selection_pot.py
    python src/selection_pot.py --p-seuil 1e-4 --exclure n_coups_roi,n_coups_sans_capture_max

Choix techniques (pour des millions de parties) :
  * lecture colonnaire des seules colonnes utiles ; groupes extraits par un seul tri d'entiers ;
  * estimation du centre et de la covariance sur un sous-échantillon pour les très gros groupes, puis score de
    TOUTES les parties par blocs (mémoire bornée) ;
  * covariance robuste par élagage itératif (10 % des parties les plus éloignées écartées), régularisée ;
  * ajustement de la loi de Pareto par moments pondérés (formule fermée, sans optimiseur, stable sur peu de
    points) ; l'estimation par maximum de vraisemblance est disponible (--ajustement mle, nécessite scipy) ;
  * les calculs lourds sont des produits de matrices (BLAS) : aucune boucle Python sur les parties.
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:                                                 # permet d'importer features_core.py, placé à côté de ce fichier
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
from features_core import FEATURES_PARTIE

RACINE = Path(__file__).resolve().parent.parent
DOSSIER = os.environ.get("LICHESS_DATA", str(RACINE / "donnees"))

# ----------------------------------------------------------------------------
# Paramètres
# ----------------------------------------------------------------------------
CONTEXTE = ("n_plies", "elo", "rarete_ouverture")     # décrivent le contexte, pas l'intérêt : exclus du score
COLONNES_ID = ("Site", "ECO", "Opening", "Result", "n_plies", "elo")   # colonnes recopiées dans la sortie

N_MIN = 500          # taille minimale d'un groupe (sinon regroupé avec sa famille A-E, puis avec le reste)
Q_SEUIL = 0.95       # quantile du seuil POT (le nombre de dépassements est borné ci-dessous)
M_MIN = 50           # nombre minimal de dépassements pour ajuster la loi de Pareto
M_MAX_FRAC = 0.25    # au plus 25 % du groupe est considéré comme « queue »
P_SEUIL = 1e-3       # une partie est retenue si sa p-valeur de rareté est <= P_SEUIL
TRIM = 0.9           # part des parties conservée pour estimer la covariance (élagage)
N_ITER = 3           # itérations d'élagage
SHRINK = 0.05        # régularisation de la covariance vers l'identité
Z_MAX = 10.0         # les écarts standardisés sont bornés à ±Z_MAX
N_AJUST = 200_000    # taille maximale de l'échantillon servant à estimer centre et covariance
CHUNK = 100_000      # parties scorées à la fois
P_PLANCHER = 1e-15   # plancher des p-valeurs (rareté maximale = 15)


# ----------------------------------------------------------------------------
# Distance de Mahalanobis robuste
# ----------------------------------------------------------------------------
def transformer(X):
    """log1p signé : atténue les queues lourdes des compteurs, conserve le signe (min_bal_gagnant < 0…)."""
    return np.sign(X) * np.log1p(np.abs(X))


def ajuster(T):
    """Modèle robuste d'un groupe à partir d'un échantillon T (déjà transformé, avec d'éventuels NaN) :
    centre et échelle par la médiane et la MAD, puis covariance par élagage itératif.
    Renvoie None si moins de 2 features varient."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                 # colonnes entièrement NaN
        med = np.nanmedian(T, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(T - med), axis=0)
        sd = np.nanstd(T, axis=0)
    ech = np.where(mad > 1e-9, mad, sd)                                 # MAD nulle (variable creuse) : écart-type
    cols = np.flatnonzero(np.isfinite(ech) & (ech > 1e-9))              # features constantes écartées
    if cols.size < 2:
        return None
    med, ech = med[cols], ech[cols]
    Z = np.clip((T[:, cols] - med) / ech, -Z_MAX, Z_MAX)
    np.nan_to_num(Z, copy=False)                                        # valeur manquante -> médiane (écart nul)
    n, q = Z.shape
    h = min(n, max(int(np.ceil(TRIM * n)), q + 2))
    d2 = np.einsum("ij,ij->i", Z, Z)                                    # départ : distance euclidienne
    I = np.eye(q)
    for _ in range(N_ITER):
        garde = np.argpartition(d2, h - 1)[:h]                          # les h parties les moins éloignées
        Zk = Z[garde]
        mu = Zk.mean(axis=0)
        Zc = Zk - mu
        S = (1 - SHRINK) * (Zc.T @ Zc) / (h - 1) + SHRINK * I           # covariance élaguée, régularisée
        W = np.linalg.inv(np.linalg.cholesky(S)).T                      # blanchiment : (Z-mu) @ W a pour covariance I
        D = (Z - mu) @ W
        d2 = np.einsum("ij,ij->i", D, D)
    return dict(cols=cols, med=med, ech=ech, mu=mu, W=W, S=S)


def _standardiser(m, X):
    A = transformer(X[:, m["cols"]].astype(np.float64))
    nan_frac = np.isnan(A).mean(axis=1)
    Z = np.clip((A - m["med"]) / m["ech"], -Z_MAX, Z_MAX)
    np.nan_to_num(Z, copy=False)
    return Z, nan_frac


def distances(m, X):
    """d² de Mahalanobis et part de valeurs manquantes de chaque ligne de X (lignes brutes), par blocs."""
    n = X.shape[0]
    d2, nan_frac = np.empty(n), np.empty(n)
    for a in range(0, n, CHUNK):
        Z, nf = _standardiser(m, X[a:a + CHUNK])
        D = (Z - m["mu"]) @ m["W"]
        d2[a:a + CHUNK] = np.einsum("ij,ij->i", D, D)
        nan_frac[a:a + CHUNK] = nf
    return d2, nan_frac


def principales_contributions(m, X, noms, k=3):
    """Pour chaque ligne de X, les k features qui contribuent le plus à d², avec leur écart standardisé signé.
    La contribution de la feature j est Zc_j * (S⁻¹ Zc)_j ; leur somme vaut d² (au facteur d'élagage près)."""
    Z, _ = _standardiser(m, X)
    Zc = Z - m["mu"]
    C = Zc * (Zc @ np.linalg.inv(m["S"]))
    top = np.argsort(-C, axis=1)[:, :k]
    return [", ".join(f"{noms[m['cols'][j]]}({Z[i, j]:+.1f})" for j in top[i]) for i in range(len(Z))]


# ----------------------------------------------------------------------------
# POT : loi de Pareto généralisée sur les dépassements
# ----------------------------------------------------------------------------
def ajuster_gpd(exc, methode="pwm"):
    """(xi, sigma) de la loi de Pareto généralisée des dépassements `exc` (> 0).
    pwm : moments pondérés (Hosking et Wallis, 1987), formule fermée ; mle : maximum de vraisemblance (scipy)."""
    if methode == "mle":
        try:
            from scipy.stats import genpareto
            xi, _, sigma = genpareto.fit(exc, floc=0)
            return float(xi), float(sigma)
        except Exception:
            pass                                                        # repli sur pwm
    y = np.sort(exc)
    m = y.size
    a0 = y.mean()
    a1 = np.mean(y * (1 - (np.arange(1, m + 1) - 0.35) / m))
    den = a0 - 2 * a1
    if den <= 0:
        return 0.0, float(a0)                                           # repli : loi exponentielle
    xi, sigma = 2 - a0 / den, 2 * a0 * a1 / den
    return float(np.clip(xi, -0.5, 1.0)), float(max(sigma, 1e-12))


def gpd_sf(y, xi, sigma):
    """Fonction de survie de la loi de Pareto généralisée."""
    if abs(xi) < 1e-8:
        return np.exp(-y / sigma)
    t = np.maximum(1 + xi * y / sigma, 0.0)
    with np.errstate(divide="ignore"):
        return t ** (-1.0 / xi)


def pot(d2, q, m_min, methode):
    """Seuil POT, paramètres de la loi de Pareto et p-valeur de rareté de chaque partie.
    Au-dessus du seuil u : p = (n_dépassements / n) * S_GPD(d² - u) ; en dessous : fréquence empirique.
    Renvoie (p, ordre trié de d², infos du groupe)."""
    n = d2.size
    m = int(np.clip(np.ceil((1 - q) * n), m_min, max(m_min, int(M_MAX_FRAC * n))))
    ordre = np.sort(d2)
    u = ordre[max(n - m - 1, 0)]
    haut = d2 > u
    ne = int(haut.sum())
    p = np.empty(n)
    if ne < 10:                                                         # queue inexploitable : p empirique partout
        p[:] = (n - np.searchsorted(ordre, d2, side="left")) / n
        return np.maximum(p, P_PLANCHER), ordre, dict(u=u, n_exc=ne, xi=np.nan, sigma=np.nan, ks=np.nan)
    exc = d2[haut] - u
    xi, sigma = ajuster_gpd(exc, methode)
    p[haut] = (ne / n) * gpd_sf(exc, xi, sigma)
    p[~haut] = (n - np.searchsorted(ordre, d2[~haut], side="left")) / n
    # qualité de l'ajustement : écart maximal entre fonctions de répartition empirique et ajustée (Kolmogorov-Smirnov)
    F = 1 - gpd_sf(np.sort(exc), xi, sigma)
    k = np.arange(1, ne + 1) / ne
    ks = float(max(np.max(k - F), np.max(F - (k - 1 / ne))))
    return np.maximum(p, P_PLANCHER), ordre, dict(u=u, n_exc=ne, xi=xi, sigma=sigma, ks=ks)


# ----------------------------------------------------------------------------
# Groupes d'ouverture
# ----------------------------------------------------------------------------
def groupes_reference(codes, noms, n_min):
    """Groupe de référence de chaque code ECO : le code lui-même s'il a au moins n_min parties, sinon sa famille
    (« B* » : codes B trop rares regroupés), sinon « GLOBAL ». Renvoie (identifiant de groupe par partie, noms)."""
    cnt = np.bincount(codes, minlength=len(noms))
    famille = {}
    for nom, c in zip(noms, cnt):
        if c < n_min:
            famille[nom[:1]] = famille.get(nom[:1], 0) + c
    ref = [nom if c >= n_min else (nom[:1] + "*" if famille[nom[:1]] >= n_min else "GLOBAL")
           for nom, c in zip(noms, cnt)]
    noms_ref, inv = np.unique(np.array(ref, dtype=object), return_inverse=True)
    return inv[codes], list(noms_ref)


# ----------------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------------
def charger(chemin, exclure):
    """Table Arrow des colonnes utiles, matrice float32 des features, noms des features, codes ECO."""
    pf = pq.ParquetFile(chemin)
    presentes = set(pf.schema_arrow.names)
    if "ECO" not in presentes:
        sys.exit("Colonne ECO absente du parquet.")
    exclus = set(CONTEXTE) | set(exclure) | {"piece_mateuse"}
    feats = [f for f in FEATURES_PARTIE if f in presentes and f not in exclus]
    if len(feats) < 2:
        sys.exit("Moins de 2 features trouvées dans le parquet : est-ce bien la sortie de analyse_features.py ?")
    ids = [c for c in COLONNES_ID if c in presentes]
    t = pq.read_table(chemin, columns=list(dict.fromkeys(ids + feats)))
    X = np.empty((t.num_rows, len(feats)), dtype=np.float32)
    for j, f in enumerate(feats):
        X[:, j] = t.column(f).to_numpy()                                # valeurs nulles -> NaN
    eco = pc.fill_null(t.column("ECO"), "?").dictionary_encode().combine_chunks()
    return t, X, feats, eco.indices.to_numpy(), [str(s) for s in eco.dictionary.to_pylist()]


# ----------------------------------------------------------------------------
# Sélection
# ----------------------------------------------------------------------------
def selectionner(chemin, p_seuil=P_SEUIL, q=Q_SEUIL, m_min=M_MIN, n_min=N_MIN, exclure=(),
                 methode="pwm", max_par_groupe=None, verbose=True):
    """Renvoie (parties retenues, résumé par groupe, d², rareté de toutes les parties, table d'origine)."""
    t0 = time.perf_counter()
    t, X, feats, codes, noms_eco = charger(chemin, exclure)
    N = X.shape[0]
    n_min = max(n_min, 10 * len(feats))
    if verbose:
        print(f"{N:,} parties, {len(feats)} features, {len(noms_eco)} codes ECO "
              f"(chargé en {time.perf_counter() - t0:.1f} s)")
    gid, noms_g = groupes_reference(codes, noms_eco, n_min)
    ordre_g = np.argsort(gid, kind="stable")
    bornes = np.concatenate(([0], np.cumsum(np.bincount(gid, minlength=len(noms_g)))))
    rng = np.random.default_rng(0)

    d2_all = np.full(N, np.nan, dtype=np.float32)
    rar_all = np.full(N, np.nan, dtype=np.float32)
    nan_all = np.zeros(N, dtype=np.float32)
    retenues, resume = [], []
    flags = []                                                           # (indices globaux, groupe) des retenues

    for g, nom in enumerate(noms_g):
        idx = ordre_g[bornes[g]:bornes[g + 1]]
        n = idx.size
        ligne = dict(groupe=nom, n=n)
        if n < max(m_min + 10, 30):                                      # trop petit même après regroupement
            resume.append({**ligne, "statut": "trop petit : non évalué"})
            continue
        Xg = X[idx]
        echantillon = Xg if n <= N_AJUST else Xg[rng.choice(n, N_AJUST, replace=False)]
        modele = ajuster(transformer(echantillon.astype(np.float64)))
        if modele is None:
            resume.append({**ligne, "statut": "features constantes : non évalué"})
            continue
        d2, nan_frac = distances(modele, Xg)
        p, ordre, info = pot(d2, q, m_min, methode)
        d2_all[idx], rar_all[idx], nan_all[idx] = d2, -np.log10(p), nan_frac

        ok = np.flatnonzero(p <= p_seuil)
        if max_par_groupe and ok.size > max_par_groupe:                  # option : ne garder que les plus rares
            ok = ok[np.argsort(p[ok])[:max_par_groupe]]
        if ok.size:
            contrib = principales_contributions(modele, Xg[ok], feats)
            retenues.append(pd.DataFrame({
                "idx": idx[ok], "groupe_ref": nom, "n_groupe": n, "d2": d2[ok], "p_pot": p[ok],
                "rarete": -np.log10(p[ok]), "rang_groupe": n - np.searchsorted(ordre, d2[ok], side="right") + 1,
                "features_extremes": contrib, "nan_frac": nan_frac[ok]}))
        resume.append({**ligne, "statut": "ok", "seuil_u": info["u"], "n_depassements": info["n_exc"],
                       "xi": info["xi"], "sigma": info["sigma"], "ks": info["ks"], "n_retenues": int(ok.size),
                       "n_attendues_hasard": round(n * p_seuil, 2)})

    resume = pd.DataFrame(resume)
    if retenues:
        sel = pd.concat(retenues, ignore_index=True)
        origine = t.select([c for c in COLONNES_ID if c in t.column_names]).take(pa.array(sel["idx"].to_numpy()))
        sel = pd.concat([origine.to_pandas().reset_index(drop=True), sel.drop(columns="idx")], axis=1)
        tri = ["ECO", "rarete"] if "ECO" in sel else ["groupe_ref", "rarete"]
        sel = sel.sort_values(tri, ascending=[True, False]).reset_index(drop=True)
    else:
        sel = pd.DataFrame()
    if verbose:
        print(f"Sélection terminée en {time.perf_counter() - t0:.1f} s")
    return sel, resume, d2_all, rar_all, t


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Sélection POT des parties atypiques par code ECO (Mahalanobis robuste).")
    p.add_argument("--dossier", default=DOSSIER, help="dossier des données (défaut : %(default)s)")
    p.add_argument("--fichier", default="resultats/features.parquet",
                   help="parquet des features, relatif au dossier (défaut : %(default)s)")
    p.add_argument("--sortie", default="resultats/selection", help="sous-dossier de sortie (défaut : %(default)s)")
    p.add_argument("--p-seuil", type=float, default=P_SEUIL, help="p-valeur de rareté maximale (défaut : %(default)s)")
    p.add_argument("--q", type=float, default=Q_SEUIL, help="quantile du seuil POT (défaut : %(default)s)")
    p.add_argument("--m-min", type=int, default=M_MIN, help="dépassements minimaux (défaut : %(default)s)")
    p.add_argument("--n-min", type=int, default=N_MIN, help="taille minimale d'un groupe (défaut : %(default)s)")
    p.add_argument("--exclure", default="", help="features à exclure du score, séparées par des virgules")
    p.add_argument("--ajustement", choices=["pwm", "mle"], default="pwm", help="estimateur de la loi de Pareto")
    p.add_argument("--max-par-groupe", type=int, default=None, help="au plus N parties retenues par groupe")
    p.add_argument("--ecrire-scores", action="store_true", help="écrire aussi d² et la rareté de toutes les parties")
    a = p.parse_args(argv)

    chemin = os.path.join(a.dossier, a.fichier)
    if not os.path.exists(chemin):
        sys.exit(f"Parquet introuvable : {chemin}\nLancez d'abord src/analyse_features.py, ou précisez --dossier / --fichier.")
    sortie = os.path.join(a.dossier, a.sortie)
    os.makedirs(sortie, exist_ok=True)

    sel, resume, d2, rar, t = selectionner(
        chemin, a.p_seuil, a.q, a.m_min, a.n_min, [s for s in a.exclure.split(",") if s], a.ajustement,
        a.max_par_groupe)

    resume.to_csv(os.path.join(sortie, "resume_par_groupe.csv"), index=False)
    sel.to_csv(os.path.join(sortie, "parties_remarquables.csv"), index=False)
    sel.to_parquet(os.path.join(sortie, "parties_remarquables.parquet"))
    if a.ecrire_scores:
        cols = [c for c in ("Site", "ECO") if c in t.column_names]
        sc = t.select(cols).to_pandas()
        sc["d2"], sc["rarete"] = d2, rar
        sc.to_parquet(os.path.join(sortie, "scores.parquet"))

    pd.set_option("display.width", 200, "display.max_rows", 60, "display.max_colwidth", 80)
    evalues = resume[resume["statut"] == "ok"]
    print(f"\n{len(evalues)} groupes évalués sur {len(resume)} ; {len(sel):,} parties retenues "
          f"(p <= {a.p_seuil:g}).")
    douteux = evalues[evalues["ks"] > 0.15]
    if len(douteux):
        print(f"Ajustement de la queue douteux (KS > 0,15) pour : {', '.join(douteux['groupe'])}")
    if len(sel):
        print("\nParties retenues par groupe (les 15 premiers) :")
        print(evalues.sort_values("n_retenues", ascending=False)[
            ["groupe", "n", "n_depassements", "xi", "ks", "n_retenues", "n_attendues_hasard"]].head(15)
            .round(3).to_string(index=False))
        cols = [c for c in ["Site", "ECO", "Result", "rarete", "rang_groupe", "features_extremes"] if c in sel]
        print("\nLes 10 parties les plus rares :")
        print(sel.sort_values("rarete", ascending=False).head(10)[cols].round(2).to_string(index=False))
    print(f"\nFichiers écrits dans {sortie}")


if __name__ == "__main__":
    main()