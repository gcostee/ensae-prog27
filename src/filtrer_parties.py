"""
Étape 1 : filtre un dump Lichess (.pgn.zst) et écrit un parquet des parties retenues.

Critères : terminaison normale, résultat 1-0 / 0-1, cadence définie, Elo >= 1500 pour les deux joueurs
(écart <= 400), nombre de coups dans [COUPS_MIN, COUPS_MAX], et (par défaut) évaluations présentes.
La colonne 'Evals' contient une évaluation int16 par demi-coup (voir features_core.extraire_evals).

Exemple :
    python src/filtrer_parties.py --dump lichess_db_standard_rated_2026-01.pgn.zst
    python src/filtrer_parties.py --max-parties 1000000          # essai rapide
"""
import argparse
import io
import os
import re
import sys
import time
import statistics
from pathlib import Path

import zstandard as zstd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:                                                 # permet d'importer features_core.py, placé à côté de ce fichier
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
from features_core import extraire_evals, sans_variations      # lecture des évaluations (même code que l'analyse)

# Dossier des données : variable d'environnement LICHESS_DATA, sinon <racine du dépôt>/donnees
RACINE = Path(__file__).resolve().parent.parent
DOSSIER = os.environ.get("LICHESS_DATA", str(RACINE))
FICHIER = "mini_dump.pgn.zst"    # nom du dump, dans DOSSIER
SORTIE = "parties_filtrees.parquet"                      # parquet produit, dans DOSSIER

debut = time.perf_counter()

TAILLE_TRANCHE = 500_000      # parties lues par tranche (pour les statistiques)
TAILLE_LOT = 100_000          # parties gardées écrites à chaque flush sur disque
COUPS_MIN, COUPS_MAX = 20, 120
EXIGER_EVAL = True            # ne garder que les parties comportant l'évaluation ([%eval ...])
MAX_PARTIES = None            # ex. 5_000_000 pour tester ; None = tout le fichier

RE_COMMENTAIRE = re.compile(r"\{[^}]*\}")
RE_NUMERO = re.compile(r"(\d+)\.")
RE_NUM_COUP = re.compile(r"\d+\.(?:\.\.)?")
RE_RESULTAT = re.compile(r"\s*(?:1-0|0-1|1/2-1/2|\*)\s*$")
RE_ANNOT = re.compile(r"[?!]+")

SCHEMA = pa.schema([
    ("Site", pa.string()),
    ("Event", pa.string()),
    ("Result", pa.string()),
    ("WhiteElo", pa.int16()),
    ("BlackElo", pa.int16()),
    ("ECO", pa.string()),
    ("Opening", pa.string()),
    ("TimeControl", pa.string()),
    ("nb_coups", pa.int16()),
    ("has_clk", pa.bool_()),
    ("has_eval", pa.bool_()),
    ("Moves_clean", pa.string()),
    ("Evals", pa.binary()),          # NOUVEAU : évaluations alignées sur les demi-coups (int16, voir extraire_evals)
])


def passe_filtres_entetes(h):
    if h.get("Termination") != "Normal":
        return False
    if h.get("Result") not in ("1-0", "0-1"):
        return False
    if h.get("TimeControl", "-") == "-":
        return False
    try:
        elo_b = int(h["WhiteElo"])
        elo_n = int(h["BlackElo"])
    except (KeyError, ValueError):
        return False
    return elo_b >= 1500 and elo_n >= 1500 and abs(elo_b - elo_n) <= 400


def nettoyer_coups(sans_commentaires):
    s = RE_RESULTAT.sub("", sans_commentaires)
    s = RE_NUM_COUP.sub("", s)
    s = RE_ANNOT.sub("", s)
    return " ".join(s.split())


def vider_lot(lot, writer):
    if lot:
        writer.write_table(pa.Table.from_pylist(lot, schema=SCHEMA))
        lot.clear()


def filtrer_et_ecrire(chemin, sortie):
    comptes = []                       # nb de parties gardées, une valeur par tranche complète
    n_lues = n_gardees = n_clk = n_eval = 0
    n_sans_eval = 0                    # parties qui passent les filtres d'en-tête mais sans évaluation
    n_tranche = g_tranche = 0          # compteurs de la tranche courante
    entetes = {}
    lot = []

    with open(chemin, "rb") as fh, \
         pq.ParquetWriter(sortie, SCHEMA, compression="zstd") as writer:
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        texte = io.TextIOWrapper(reader, encoding="utf-8")

        for ligne in texte:
            if ligne.startswith("["):
                cle, valeur = ligne.strip()[1:-1].split(" ", 1)
                entetes[cle] = valeur.strip('"')
                continue

            ligne = ligne.strip()
            if not ligne:
                continue

            # ligne des coups : fin de la partie
            if entetes:
                n_lues += 1
                n_tranche += 1

                if passe_filtres_entetes(entetes):
                    # NOUVEAU CRITÈRE : présence de l'évaluation. Test d'une simple sous-chaîne, placé AVANT
                    # le comptage des coups (regex) pour écarter à moindre coût les parties non analysées.
                    a_eval = "[%eval" in ligne
                    if EXIGER_EVAL and not a_eval:
                        n_sans_eval += 1
                    else:
                        if "(" in ligne:                  # variations d'analyse : ne gardent que la ligne principale
                            ligne = sans_variations(ligne)
                        sans = RE_COMMENTAIRE.sub("", ligne) if "{" in ligne else ligne
                        numeros = RE_NUMERO.findall(sans)
                        nb = int(numeros[-1]) if numeros else 0

                        if COUPS_MIN <= nb <= COUPS_MAX:
                            a_clk = "[%clk" in ligne
                            n_gardees += 1
                            g_tranche += 1
                            n_clk += a_clk
                            n_eval += a_eval
                            lot.append({
                                "Site": entetes.get("Site"),
                                "Event": entetes.get("Event"),
                                "Result": entetes.get("Result"),
                                "WhiteElo": int(entetes["WhiteElo"]),
                                "BlackElo": int(entetes["BlackElo"]),
                                "ECO": entetes.get("ECO"),
                                "Opening": entetes.get("Opening"),
                                "TimeControl": entetes.get("TimeControl"),
                                "nb_coups": nb,
                                "has_clk": a_clk,
                                "has_eval": a_eval,
                                "Moves_clean": nettoyer_coups(sans),
                                "Evals": extraire_evals(ligne),      # NOUVEAU : évaluations par demi-coup
                            })
                            if len(lot) >= TAILLE_LOT:
                                vider_lot(lot, writer)

                entetes = {}

                # fin d'une tranche complète
                if n_tranche == TAILLE_TRANCHE:
                    comptes.append(g_tranche)
                    n_tranche = 0
                    g_tranche = 0
                    if len(comptes) % 10 == 0:
                        print(f"{n_lues:,} lues | {n_gardees:,} gardées | "
                              f"{len(comptes)} tranches | "
                              f"{time.perf_counter() - debut:,.0f} s")

                if MAX_PARTIES and n_lues >= MAX_PARTIES:
                    break

        vider_lot(lot, writer)

    return comptes, (g_tranche, n_tranche), n_lues, n_gardees, n_clk, n_eval, n_sans_eval


def main(argv=None):
    global debut, EXIGER_EVAL, MAX_PARTIES, COUPS_MIN, COUPS_MAX, TAILLE_TRANCHE
    p = argparse.ArgumentParser(description="Filtre un dump Lichess et écrit un parquet des parties retenues.")
    p.add_argument("--dossier", default=DOSSIER, help="dossier des données (défaut : %(default)s)")
    p.add_argument("--dump", default=FICHIER, help="nom du dump .pgn.zst dans le dossier (défaut : %(default)s)")
    p.add_argument("--sortie", default=SORTIE, help="nom du parquet produit (défaut : %(default)s)")
    p.add_argument("--max-parties", type=int, default=MAX_PARTIES, help="s'arrêter après N parties lues")
    p.add_argument("--sans-eval", action="store_true", help="ne pas exiger les évaluations")
    p.add_argument("--coups-min", type=int, default=COUPS_MIN)
    p.add_argument("--coups-max", type=int, default=COUPS_MAX)
    p.add_argument("--tranche", type=int, default=TAILLE_TRANCHE, help="taille des tranches des statistiques")
    a = p.parse_args(argv)
    MAX_PARTIES, EXIGER_EVAL = a.max_parties, not a.sans_eval
    COUPS_MIN, COUPS_MAX, TAILLE_TRANCHE = a.coups_min, a.coups_max, a.tranche
    debut = time.perf_counter()

    chemin, sortie = os.path.join(a.dossier, a.dump), os.path.join(a.dossier, a.sortie)
    if not os.path.exists(chemin):
        sys.exit(f"Dump introuvable : {chemin}\n"
                 "Placez le .pgn.zst dans le dossier des données, ou précisez --dossier / --dump.")
    comptes, (reste_g, reste_l), n_lues, n_gardees, n_clk, n_eval, n_sans_eval = filtrer_et_ecrire(chemin, sortie)

    # ---------- Statistiques par tranche ----------
    if len(comptes) >= 2:
        moyenne = statistics.mean(comptes)
        variance = statistics.variance(comptes)          # variance corrigée (n-1)
        ecart_type = statistics.stdev(comptes)
        print(f"\nTranches complètes de {TAILLE_TRANCHE:,} parties : {len(comptes)}")
        print(f"Moyenne empirique        : {moyenne:,.1f} parties gardées / tranche")
        print(f"Variance (n-1)           : {variance:,.1f}")
        print(f"Écart-type               : {ecart_type:,.1f}")
        print(f"Coefficient de variation : {ecart_type / moyenne:.3%}")
        print(f"Taux de rétention moyen  : {moyenne / TAILLE_TRANCHE:.2%}")
        print(f"Tranche finale incomplète (exclue) : {reste_g:,} gardées sur {reste_l:,} lues")
    else:
        print("\n(moins de 2 tranches complètes : statistiques par tranche non calculées ; voir --tranche)")

    # ---------- Totaux ----------
    print(f"\nParties lues            : {n_lues:,}")
    if n_gardees:
        print(f"Parties gardées         : {n_gardees:,} ({n_gardees / n_lues:.2%})")
        print(f"Rejetées faute d'évaluation (après filtres d'en-tête) : {n_sans_eval:,}")
        print(f"Avec %clk               : {n_clk:,} ({n_clk / n_gardees:.2%} des parties gardées)")
        print(f"Avec %eval              : {n_eval:,} ({n_eval / n_gardees:.2%} des parties gardées)")
    else:
        print("Aucune partie gardée : vérifier les critères (--sans-eval, --coups-min/max).")
    pd.DataFrame({"tranche": range(1, len(comptes) + 1), "n_gardees": comptes}).to_csv(
        os.path.join(a.dossier, "comptes_par_tranche.csv"), index=False)
    print(f"Parquet écrit : {sortie}")
    print(f"Temps écoulé : {time.perf_counter() - debut:,.0f} secondes")


if __name__ == "__main__":
    main()