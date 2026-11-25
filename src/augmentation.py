"""
Étape 3c : évalue avec Stockfish toutes les parties du PU learning et recalcule leurs features (features_core).

Les trois groupes :
  * positifs  : parties du mois (pu/gotm.pgn, produit par scraping.py) ;
  * témoin    : parties tirées au hasard (pu/temoin.parquet, produit par temoin.py) ;
  * candidates: parties retenues par selection_pot (resultats/selection/base_selection.parquet).
Leurs évaluations d'origine ne sont pas comparables (les parties du mois n'en ont souvent pas, celles de la base
viennent d'analyses Lichess de profondeurs variées) : toutes sont donc réévaluées par le MÊME moteur, à la MÊME
profondeur, puis passent par le même calcul de features. C'est ce qui rend les trois groupes comparables.

Profondeur par défaut : 12. Stockfish (réseau NNUE) y joue déjà bien au-dessus du niveau des parties étudiées et
repère gaffes, coups seuls et tactiques courtes ; chaque demi-profondeur de plus coûte environ 1,5 fois plus de
temps. Ordre de grandeur : 1 à 3 s par partie et par cœur à profondeur 12 (selon le processeur), soit environ
1 à 2 heures pour ~8 000 parties sur 4 cœurs.

Stockfish n'est pas installé par pip : télécharger l'exécutable (https://stockfishchess.org/download/) puis
indiquer son chemin avec --stockfish ou la variable d'environnement STOCKFISH_PATH (sinon il est cherché dans le PATH).

Reprise : les évaluations sont mises en cache (pu/cache_stockfish_d<profondeur>/), lot par lot ; une exécution
interrompue reprend là où elle s'était arrêtée, et modifier features_core.py ne demande pas de relancer Stockfish
(seules les features sont recalculées, en quelques secondes).

Sortie : <dossier>/pu/features_augmentees.parquet (une ligne par partie : groupe, identifiants, informations de
selection_pot pour les candidates, évaluations Stockfish, features).

Exemple :
    python src/augmentation.py --stockfish "C:\\outils\\stockfish\\stockfish-windows-x86-64-avx2.exe"
    python src/augmentation.py --profondeur 14 --workers 6
"""
import argparse
import glob
import hashlib
import os
import shutil
import sys
import time
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:                                                 # permet d'importer les modules placés à côté de ce fichier
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
from features_core import ECHEC, FEATURES_PARTIE, features_partie
from outils import DOSSIER, chemin, extraire_lignes

PROFONDEUR = 12
HASH_MO = 64                 # table de transposition par moteur
TAILLE_TACHE = 20            # parties par tâche envoyée à un moteur
LOT_CACHE = 200              # parties par fichier de cache
META_CANDIDATS = ["groupe_ref", "n_groupe", "d2", "p_pot", "rarete", "rang_groupe", "features_extremes"]


# ----------------------------------------------------------------------------
# Chargement des trois groupes
# ----------------------------------------------------------------------------
def cle(source, site, coups):
    return f"{source}:{hashlib.md5((str(site) + '|' + coups).encode()).hexdigest()[:16]}"


def charger_positifs(chemin_pgn):
    lignes = []
    with open(chemin_pgn, encoding="utf-8") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            board, sans = g.board(), []
            for coup in g.mainline_moves():
                sans.append(board.san(coup))
                board.push(coup)
            h = g.headers
            lignes.append(dict(source="positif", Site=h.get("Site"), Result=h.get("Result"), ECO=h.get("ECO"),
                               Opening=h.get("Opening"), TimeControl=h.get("TimeControl"),
                               WhiteElo=pd.to_numeric(h.get("WhiteElo"), errors="coerce"),
                               BlackElo=pd.to_numeric(h.get("BlackElo"), errors="coerce"),
                               GOTM=h.get("GOTM"), GOTMSource=h.get("GOTMSource"), Moves_clean=" ".join(sans)))
    return pd.DataFrame(lignes)


def completer_coups(df, chemin_secours):
    """Coups manquants (base écrite sans Moves_clean) : lus dans parties_filtrees.parquet d'après le Site."""
    manque = df["Moves_clean"].isna() if "Moves_clean" in df else pd.Series(True, index=df.index)
    if not manque.any():
        return df
    if not os.path.exists(chemin_secours):
        sys.exit(f"Coups absents de la base des candidates et {chemin_secours} introuvable.")
    sites = pq.read_table(chemin_secours, columns=["Site"]).column("Site").combine_chunks()
    voulus = pa.array(df.loc[manque, "Site"].tolist(), type=sites.type)
    idx = np.flatnonzero(pc.is_in(sites, value_set=voulus).to_numpy(zero_copy_only=False))
    trouves = extraire_lignes(chemin_secours, idx, ["Site", "Moves_clean"]).set_index("Site")["Moves_clean"]
    df = df.copy()
    df.loc[manque, "Moves_clean"] = df.loc[manque, "Site"].map(trouves)
    return df


def charger_groupes(o):
    morceaux = []
    p_pos, p_tem, p_cand = chemin(o.dossier, o.positifs), chemin(o.dossier, o.temoin), chemin(o.dossier, o.candidats)
    for nom, p in (("positifs", p_pos), ("témoin", p_tem), ("candidates", p_cand)):
        if not os.path.exists(p):
            sys.exit(f"Fichier des {nom} introuvable : {p}")
    pos = charger_positifs(p_pos)
    tem = pd.read_parquet(p_tem).assign(source="temoin")
    cand = completer_coups(pd.read_parquet(p_cand), chemin(o.dossier, o.secours)).assign(source="candidat")
    if o.max_candidats and len(cand) > o.max_candidats:              # garder les plus rares
        cand = cand.sort_values("rarete", ascending=False).head(o.max_candidats)
    if o.max_temoin and len(tem) > o.max_temoin:
        tem = tem.sample(o.max_temoin, random_state=0)
    for d in (pos, cand, tem):                                       # ordre de traitement : les plus précieux d'abord
        garder = ["source", "Site", "Result", "ECO", "Opening", "TimeControl", "WhiteElo", "BlackElo", "GOTM",
                  "GOTMSource", "Moves_clean"] + META_CANDIDATS
        morceaux.append(d[[c for c in garder if c in d]])
    df = pd.concat(morceaux, ignore_index=True)
    df = df[df["Moves_clean"].notna() & (df["Moves_clean"].str.len() > 0)].reset_index(drop=True)
    df.insert(1, "cle", [cle(s, site, m) for s, site, m in zip(df["source"], df["Site"], df["Moves_clean"])])
    return df.drop_duplicates("cle").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Stockfish
# ----------------------------------------------------------------------------
# Un moteur est ouvert au début de chaque tâche (TAILLE_TACHE parties) et fermé à la fin : le démarrage coûte
# ~0,3 s, négligeable devant l'analyse de 20 parties. Garder un moteur ouvert pendant toute la vie du processus de
# calcul l'empêche de s'arrêter (le fil d'exécution de python-chess qui le pilote n'est pas un démon).
_MOTEUR = None
_PARAMS = None


def _init(chemin_sf, profondeur, hash_mo):
    global _PARAMS
    _PARAMS = (chemin_sf, profondeur, hash_mo)


def _demarrer():
    global _MOTEUR
    chemin_sf, _, hash_mo = _PARAMS
    _MOTEUR = chess.engine.SimpleEngine.popen_uci(chemin_sf)
    _MOTEUR.configure({"Threads": 1, "Hash": hash_mo})


def _fermer():
    global _MOTEUR
    try:
        if _MOTEUR is not None:
            _MOTEUR.quit()
    except Exception:
        try:
            _MOTEUR.close()
        except Exception:
            pass
    _MOTEUR = None


def coups_san(texte):
    """Coups SAN d'un texte de coups, découpés selon la même règle que features_partie (numéros, résultat et
    annotations ignorés) : les évaluations restent alignées sur les demi-coups comptés par les features."""
    for tok in texte.split():
        if "." in tok:
            tok = tok[tok.rfind(".") + 1:]
        if tok and tok[0] not in "0123456789*$":
            yield tok.rstrip("?!")


def evaluer(coups, id_partie):
    """Évaluation après chaque demi-coup, au format de la colonne Evals (int16, point de vue des blancs) :
    centipions bornés à ±9999 ; mat en k coups : +(10000+k) pour les blancs, -(10000+k) pour les noirs."""
    board, vals = chess.Board(), []
    limite = chess.engine.Limit(depth=_PARAMS[1])
    for san in coups_san(coups):
        board.push_san(san)
        if board.is_checkmate():                                     # le camp au trait est mat
            vals.append(-10001 if board.turn == chess.WHITE else 10001)
            continue
        if board.is_stalemate() or board.is_insufficient_material():
            vals.append(0)
            continue
        sc = _MOTEUR.analyse(board, limite, game=id_partie)["score"].white()
        if sc.is_mate():
            k = sc.mate()
            vals.append(0 if k == 0 else (10000 + k if k > 0 else -10000 + k))
        else:
            vals.append(max(-9999, min(9999, sc.score())))
    return array("h", vals).tobytes()


def traiter_tache(tache):
    """tache : liste de (clé, coups). Renvoie une liste de (clé, évaluations ou None, message d'erreur)."""
    sortie = []
    _demarrer()
    try:
        _evaluer_tache(tache, sortie)
    finally:
        _fermer()
    return sortie


def _evaluer_tache(tache, sortie):
    for k, coups in tache:
        for essai in range(2):
            try:
                sortie.append((k, evaluer(coups, k), ""))
                break
            except (chess.engine.EngineTerminatedError, chess.engine.EngineError) as e:
                _fermer()                                            # moteur arrêté : on le relance une fois
                _demarrer()
                if essai:
                    sortie.append((k, None, f"moteur : {e!r}"[:200]))
            except Exception as e:                                   # coup illisible, etc.
                sortie.append((k, None, repr(e)[:200]))
                break


# ----------------------------------------------------------------------------
# Cache des évaluations
# ----------------------------------------------------------------------------
def lire_cache(rep):
    fichiers = sorted(glob.glob(os.path.join(rep, "lot_*.parquet")))
    if not fichiers:
        return {}, 0
    t = pa.concat_tables([pq.read_table(f) for f in fichiers]).to_pandas()
    return dict(zip(t["cle"], zip(t["evals"], t["erreur"]))), len(fichiers)


def ecrire_cache(rep, numero, resultats):
    t = pa.table({"cle": [r[0] for r in resultats], "evals": pa.array([r[1] for r in resultats], type=pa.binary()),
                  "erreur": [r[2] for r in resultats]})
    final = os.path.join(rep, f"lot_{numero:06d}.parquet")
    pq.write_table(t, final + ".tmp")
    os.replace(final + ".tmp", final)


def fmt(s):
    h, r = divmod(int(s), 3600)
    return f"{h}:{r // 60:02d}:{r % 60:02d}"


def analyser(df, chemin_sf, profondeur, workers, rep_cache):
    os.makedirs(rep_cache, exist_ok=True)
    cache, n_fichiers = lire_cache(rep_cache)
    a_faire = [(k, m) for k, m in zip(df["cle"], df["Moves_clean"]) if k not in cache]
    print(f"  {len(df):,} parties ; déjà évaluées (cache) : {len(df) - len(a_faire):,} ; à évaluer : {len(a_faire):,}")
    if not a_faire:
        return cache
    taches = [a_faire[i:i + TAILLE_TACHE] for i in range(0, len(a_faire), TAILLE_TACHE)]
    tampon, numero, faites, t0 = [], n_fichiers, 0, time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(chemin_sf, profondeur, HASH_MO)) as ex:
        futurs = [ex.submit(traiter_tache, t) for t in taches]
        for f in as_completed(futurs):
            res = f.result()
            tampon.extend(res)
            faites += len(res)
            for k, ev, err in res:
                cache[k] = (ev, err)
            if len(tampon) >= LOT_CACHE:
                ecrire_cache(rep_cache, numero, tampon)
                numero, tampon = numero + 1, []
            dt = time.perf_counter() - t0
            if faites % (5 * TAILLE_TACHE) < len(res) or faites == len(a_faire):
                print(f"  {faites:,}/{len(a_faire):,} parties | {dt / faites:.2f} s par partie (tous moteurs) | "
                      f"écoulé {fmt(dt)} | reste ~{fmt(dt / faites * (len(a_faire) - faites))}")
    if tampon:
        ecrire_cache(rep_cache, numero, tampon)
    return cache


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
def trouver_stockfish(indique):
    for c in (indique, os.environ.get("STOCKFISH_PATH"), shutil.which("stockfish"), shutil.which("stockfish.exe")):
        if c and os.path.isfile(c):
            return c
    sys.exit("Stockfish introuvable. Téléchargez-le (https://stockfishchess.org/download/) puis indiquez le chemin de "
             "l'exécutable avec --stockfish ou la variable d'environnement STOCKFISH_PATH.")


def main(argv=None):
    a = argparse.ArgumentParser(description="Évalue les parties du PU learning avec Stockfish et recalcule les features.")
    a.add_argument("--dossier", default=DOSSIER, help="dossier des données (défaut : %(default)s)")
    a.add_argument("--positifs", default="pu/gotm.pgn")
    a.add_argument("--temoin", default="pu/temoin.parquet")
    a.add_argument("--candidats", default="resultats/selection/base_selection.parquet")
    a.add_argument("--secours", default="parties_filtrees.parquet", help="pour retrouver des coups manquants")
    a.add_argument("--sortie", default="pu/features_augmentees.parquet")
    a.add_argument("--stockfish", default=None, help="chemin de l'exécutable Stockfish")
    a.add_argument("--profondeur", type=int, default=PROFONDEUR, help="profondeur d'analyse (défaut : %(default)s)")
    a.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="moteurs en parallèle (défaut : nombre de cœurs - 1 = %(default)s)")
    a.add_argument("--max-candidats", type=int, default=None, help="limiter aux N candidates les plus rares")
    a.add_argument("--max-temoin", type=int, default=None, help="limiter le témoin à N parties")
    o = a.parse_args(argv)

    t0 = time.perf_counter()
    chemin_sf = trouver_stockfish(o.stockfish)
    df = charger_groupes(o)
    print(f"Parties à traiter : {df['source'].value_counts().to_dict()}")
    print(f"Stockfish : {chemin_sf} | profondeur {o.profondeur} | {o.workers} moteur(s) en parallèle")
    rep_cache = chemin(o.dossier, os.path.join(os.path.dirname(o.sortie), f"cache_stockfish_d{o.profondeur}"))
    cache = analyser(df, chemin_sf, o.profondeur, o.workers, rep_cache)

    print("Calcul des features...")
    evals = [cache.get(k, (None, "absent du cache"))[0] for k in df["cle"]]
    erreurs = [cache.get(k, (None, "absent du cache"))[1] for k in df["cle"]]
    lignes = []
    for m, r, ev in zip(df["Moves_clean"], df["Result"], evals):
        try:
            lignes.append(features_partie(m, r, ev) if ev is not None else ECHEC)
        except Exception:
            lignes.append(ECHEC)
    feats = pd.DataFrame(lignes, columns=FEATURES_PARTIE)
    out = pd.concat([df.reset_index(drop=True), feats], axis=1)
    out["elo"] = (pd.to_numeric(out["WhiteElo"], errors="coerce") + pd.to_numeric(out["BlackElo"], errors="coerce")) / 2
    out["Evals_sf"], out["erreur_sf"], out["profondeur"] = evals, erreurs, o.profondeur
    sortie = chemin(o.dossier, o.sortie)
    os.makedirs(os.path.dirname(sortie), exist_ok=True)
    out.to_parquet(sortie)

    ok = out["acpl_gagnant"].notna()
    bilan = out.assign(evaluees=ok).groupby("source").agg(parties=("cle", "size"), evaluees=("evaluees", "sum"))
    print("\n" + bilan.to_string())
    if (out["erreur_sf"].fillna("") != "").any():
        print(f"  {int((out['erreur_sf'].fillna('') != '').sum())} parties en erreur (colonne erreur_sf)")
    print(f"\nFeatures écrites dans {sortie} ({fmt(time.perf_counter() - t0)})")


if __name__ == "__main__":
    main()
