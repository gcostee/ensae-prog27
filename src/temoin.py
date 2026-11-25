"""
Étape 3b : groupe témoin (parties « non étiquetées » du PU learning), tiré au hasard dans la base mensuelle.

Pourquoi 5 000 parties par défaut :
  * en PU learning, le groupe non étiqueté doit représenter la population dans laquelle on cherche les parties
    remarquables ; avec ~60 features, quelques milliers de parties suffisent à estimer leurs distributions et
    corrélations (l'erreur sur une proportion est d'environ 0,7 point pour 5 000 tirages) ;
  * il doit être bien plus grand que le groupe positif (quelques dizaines de parties du mois : rapport ~100) ;
  * chaque partie passe ensuite dans Stockfish (augmentation.py) : 5 000 parties à profondeur 12 représentent
    de l'ordre d'une heure sur 4 cœurs ; au-delà, le gain statistique est faible pour un coût linéaire.

Deux sources possibles :
  * --source parquet (défaut) : les parties analysées à l'étape 2 (resultats/features.parquet), c'est-à-dire
    exactement la population dans laquelle selection_pot a choisi les candidates (mêmes filtres : Elo >= 1500,
    20 à 120 coups, résultat décisif…). C'est le choix cohérent pour le PU learning, et il prend quelques secondes ;
  * --source dump : tirage uniforme (échantillonnage par réservoir) dans le dump .pgn.zst entier, parmi les
    parties qui passent les mêmes filtres d'en-tête et de longueur, avec ou sans évaluation (un passage complet
    sur le dump : de l'ordre d'une demi-heure à une heure).
Les candidates de selection_pot sont exclues du témoin dans les deux cas.

Sortie : <dossier>/pu/temoin.parquet (Site, Result, Elo, ECO, cadence, coups Moves_clean).

Exemple :
    python src/temoin.py
    python src/temoin.py --n 10000 --source dump --dump lichess_db_standard_rated_2026-01.pgn.zst
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:                                                 # permet d'importer les modules placés à côté de ce fichier
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import filtrer_parties as fp
from features_core import sans_variations
from outils import DOSSIER, DUMP, chemin, decouper, extraire_lignes, iter_parties

N_TEMOIN = 5_000
COLONNES = ["Site", "Event", "Result", "WhiteElo", "BlackElo", "ECO", "Opening", "TimeControl", "Moves_clean"]


def sites_exclus(chemin_base):
    if chemin_base and os.path.exists(chemin_base):
        return pq.read_table(chemin_base, columns=["Site"]).column("Site").combine_chunks()
    return pa.array([], type=pa.string())


# ----------------------------------------------------------------------------
# Source 1 : parquet de l'étape 2
# ----------------------------------------------------------------------------
def depuis_parquet(source, secours, n, exclus, graine):
    pf = pq.ParquetFile(source)
    total = pf.metadata.num_rows
    lecture = source
    if "Moves_clean" not in pf.schema_arrow.names:                 # features.parquet écrit sans les coups
        print(f"  {os.path.basename(source)} n'a pas les coups : lecture dans {os.path.basename(secours)} "
              f"(mêmes {total:,} premières lignes)")
        lecture = secours
    sites = pq.read_table(lecture, columns=["Site"]).column("Site").slice(0, total).combine_chunks()
    exclu = pc.is_in(sites, value_set=exclus.cast(sites.type)).to_numpy(zero_copy_only=False)
    eligibles = np.flatnonzero(~exclu)
    rng = np.random.default_rng(graine)
    tirage = np.sort(rng.choice(eligibles, size=min(n, eligibles.size), replace=False))
    print(f"  {total:,} parties dans la population, {total - eligibles.size:,} candidates exclues, "
          f"{tirage.size:,} tirées")
    df = extraire_lignes(lecture, tirage, COLONNES)
    df.insert(0, "ligne_source", tirage)
    return df


# ----------------------------------------------------------------------------
# Source 2 : dump complet (échantillonnage par réservoir)
# ----------------------------------------------------------------------------
def nettoyer(texte):
    if "(" in texte:
        texte = sans_variations(texte)
    sans = fp.RE_COMMENTAIRE.sub("", texte) if "{" in texte else texte
    numeros = fp.RE_NUMERO.findall(sans)
    return fp.nettoyer_coups(sans), (int(numeros[-1]) if numeros else 0)


def depuis_dump(chemin_dump, n, exclus, graine):
    exclus = set(exclus.to_pylist())
    rng = random.Random(graine)
    reservoir, eligibles, lues, t0 = [], 0, 0, time.perf_counter()
    for partie in iter_parties(chemin_dump):
        lues += 1
        if lues % 5_000_000 == 0:
            print(f"  {lues:,} parties lues, {eligibles:,} éligibles ({time.perf_counter() - t0:,.0f} s)")
        if b'[Termination "Normal"]' not in partie or (b'[Result "1-0"]' not in partie
                                                       and b'[Result "0-1"]' not in partie):
            continue                                                 # préfiltre rapide sur les octets
        h, coups = decouper(partie)
        if not fp.passe_filtres_entetes(h):
            continue
        propres, nb = nettoyer(coups)
        if not (fp.COUPS_MIN <= nb <= fp.COUPS_MAX) or h.get("Site") in exclus:
            continue
        eligibles += 1
        ligne = {c: h.get(c) for c in COLONNES[:-1]}
        ligne["Moves_clean"] = propres
        if len(reservoir) < n:                                       # réservoir (algorithme R)
            reservoir.append(ligne)
        else:
            j = rng.randrange(eligibles)
            if j < n:
                reservoir[j] = ligne
    print(f"  {lues:,} parties lues, {eligibles:,} éligibles, {len(reservoir):,} tirées "
          f"({time.perf_counter() - t0:,.0f} s)")
    df = pd.DataFrame(reservoir, columns=COLONNES)
    for c in ("WhiteElo", "BlackElo"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main(argv=None):
    a = argparse.ArgumentParser(description="Tire au hasard le groupe témoin (non étiqueté) du PU learning.")
    a.add_argument("--dossier", default=DOSSIER, help="dossier des données (défaut : %(default)s)")
    a.add_argument("--n", type=int, default=N_TEMOIN, help="nombre de parties tirées (défaut : %(default)s)")
    a.add_argument("--source", choices=["parquet", "dump"], default="parquet", help="population de tirage")
    a.add_argument("--parquet", default="resultats/features.parquet", help="parquet de l'étape 2")
    a.add_argument("--secours", default="parties_filtrees.parquet", help="parquet de l'étape 1 (si les coups manquent)")
    a.add_argument("--dump", default=DUMP, help="dump .pgn.zst (source dump)")
    a.add_argument("--candidates", default="resultats/selection/base_selection.parquet",
                   help="base de selection_pot (ses parties sont exclues du témoin)")
    a.add_argument("--sortie", default="pu/temoin.parquet", help="fichier produit (défaut : %(default)s)")
    a.add_argument("--graine", type=int, default=0, help="graine du tirage (reproductibilité)")
    o = a.parse_args(argv)

    t0 = time.perf_counter()
    exclus = sites_exclus(chemin(o.dossier, o.candidates))
    if o.source == "parquet":
        src = chemin(o.dossier, o.parquet)
        if not os.path.exists(src):
            sys.exit(f"Parquet introuvable : {src}\nLancez d'abord src/analyse_features.py.")
        print(f"Tirage de {o.n:,} parties dans {src}")
        df = depuis_parquet(src, chemin(o.dossier, o.secours), o.n, exclus, o.graine)
    else:
        src = chemin(o.dossier, o.dump)
        if not os.path.exists(src):
            sys.exit(f"Dump introuvable : {src}")
        print(f"Tirage de {o.n:,} parties dans {src} (passage complet sur le dump)")
        df = depuis_dump(src, o.n, exclus, o.graine)
    sortie = chemin(o.dossier, o.sortie)
    os.makedirs(os.path.dirname(sortie), exist_ok=True)
    df.to_parquet(sortie)
    print(f"{len(df):,} parties témoin écrites dans {sortie} ({time.perf_counter() - t0:,.1f} s)")
    if len(df) < o.n:
        print(f"  ATTENTION : seulement {len(df):,} parties disponibles (demandé : {o.n:,}).")


if __name__ == "__main__":
    main()
