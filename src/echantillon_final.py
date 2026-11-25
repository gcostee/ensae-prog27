"""
Étape 3e : échantillon final de 20 parties parmi les candidates étiquetées positives par le PU learning.

  * 20 positives ou moins : toutes sont renvoyées ;
  * plus de 20 : on mélange deux profils pour ne pas montrer que des cas extrêmes :
      - la moitié (--n-extremes, 10) parmi les plus grandes distances : rareté de selection_pot la plus forte
        (rareté = -log10 de la p-valeur POT, comparable d'un code ECO à l'autre, contrairement à d²) ;
      - l'autre moitié parmi les positives « plus communes » : celles dont la rareté est inférieure à la médiane
        des positives, réparties en tranches de rareté (une partie par tranche, la plus probable selon le PU
        learning), pour couvrir toute cette zone ;
    au plus --max-par-eco parties par code ECO (3 par défaut), pour varier les ouvertures.

Sortie : <dossier>/pu/echantillon_final.csv (et .parquet).

Exemple :
    python src/echantillon_final.py
    python src/echantillon_final.py --n 20 --n-extremes 10 --max-par-eco 2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

try:                                                 # permet d'importer outils.py, placé à côté de ce fichier
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
from outils import DOSSIER, chemin

N_FINAL = 20
MAX_PAR_ECO = 3


def choisir(pos, n=N_FINAL, n_extremes=None, max_par_eco=MAX_PAR_ECO):
    """pos : DataFrame des candidates positives (colonnes rarete, p_pu, ECO). Renvoie la sélection, avec une colonne
    « profil » (distance élevée / plus commune)."""
    if len(pos) <= n:
        return pos.assign(profil="toutes les positives").sort_values("rarete", ascending=False)
    n_extremes = n // 2 if n_extremes is None else min(n_extremes, n)
    par_eco = {}

    def admissible(i):
        return max_par_eco is None or par_eco.get(pos.at[i, "ECO"], 0) < max_par_eco

    def prendre(i, profil, choix):
        choix.append((i, profil))
        par_eco[pos.at[i, "ECO"]] = par_eco.get(pos.at[i, "ECO"], 0) + 1

    choix = []
    # 1. plus grandes distances (rareté décroissante, à égalité la probabilité PU la plus forte)
    for i in pos.sort_values(["rarete", "p_pu"], ascending=False).index:
        if len(choix) >= n_extremes:
            break
        if admissible(i):
            prendre(i, "distance élevée", choix)
    # 2. plus communes : moitié basse des raretés, découpée en tranches ; dans chaque tranche, la plus probable
    deja = {i for i, _ in choix}
    reste = pos.drop(index=list(deja))
    communes = reste[reste["rarete"] <= pos["rarete"].median()].sort_values("rarete")
    n_com = n - len(choix)
    if len(communes) and n_com > 0:
        for tranche in np.array_split(communes.index.to_numpy(), min(n_com, len(communes))):
            for i in communes.loc[tranche].sort_values("p_pu", ascending=False).index:
                if admissible(i):
                    prendre(i, "plus commune", choix)
                    break
    # 3. complément si les contraintes ont laissé des places vides : les plus probables restantes
    #    (d'abord en respectant la limite par code ECO, puis sans elle si cela ne suffit pas)
    for relacher in (False, True):
        deja = {i for i, _ in choix}
        for i in pos.drop(index=list(deja)).sort_values("p_pu", ascending=False).index:
            if len(choix) >= n:
                break
            if relacher or admissible(i):
                prendre(i, "complément (plus probable)", choix)
    sel = pos.loc[[i for i, _ in choix]].copy()
    sel["profil"] = [p for _, p in choix]
    return sel


def main(argv=None):
    a = argparse.ArgumentParser(description="Choisit les 20 parties finales parmi les positives du PU learning.")
    a.add_argument("--dossier", default=DOSSIER, help="dossier des données (défaut : %(default)s)")
    a.add_argument("--entree", default="pu/pu_candidats.parquet")
    a.add_argument("--sortie", default="pu/echantillon_final.csv")
    a.add_argument("--n", type=int, default=N_FINAL, help="taille de l'échantillon (défaut : %(default)s)")
    a.add_argument("--n-extremes", type=int, default=None, help="parties à grande distance (défaut : la moitié)")
    a.add_argument("--max-par-eco", type=int, default=MAX_PAR_ECO, help="0 = pas de limite (défaut : %(default)s)")
    o = a.parse_args(argv)

    entree = chemin(o.dossier, o.entree)
    if not os.path.exists(entree):
        sys.exit(f"Fichier introuvable : {entree}\nLancez d'abord src/pu_learning.py.")
    cand = pd.read_parquet(entree)
    pos = cand[cand["positif"]].reset_index(drop=True)
    print(f"{len(pos)} candidates étiquetées positives sur {len(cand):,}")
    sel = choisir(pos, o.n, o.n_extremes, o.max_par_eco or None).reset_index(drop=True)
    sel.insert(0, "rang", range(1, len(sel) + 1))
    sortie = chemin(o.dossier, o.sortie)
    os.makedirs(os.path.dirname(sortie), exist_ok=True)
    sel.to_csv(sortie, index=False)
    sel.to_parquet(sortie.replace(".csv", ".parquet"))
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    cols = [c for c in ["rang", "profil", "Site", "ECO", "Result", "rarete", "p_pu", "features_extremes"] if c in sel]
    print(sel[cols].round(3).to_string(index=False))
    print(f"\n{len(sel)} parties écrites dans {sortie}")


if __name__ == "__main__":
    main()
