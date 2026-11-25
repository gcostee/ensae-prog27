"""
Comparaison : 200 parties (10 blocs de 20) tirées uniformément au hasard parmi TOUTES les parties du dump mensuel
(~92 millions pour janvier 2026), sans aucun filtre. Seule l'adresse de la partie (en-tête Site) est conservée :
elle s'ouvre directement dans le navigateur.

Ces blocs servent de référence : comparer à l'œil les 20 parties de l'échantillon final à 20 parties quelconques.

Tirage : échantillonnage par réservoir en un seul passage sur le dump. Le dump est décompressé par gros blocs et
seules les lignes [Site "…"] des parties tirées sont lues ; la durée est celle de la décompression du fichier
(de l'ordre de quelques minutes à une demi-heure selon la machine et le disque).

Sortie : <dossier>/comparaison/comparaison_200.csv (colonnes bloc, position, Site).

Exemple :
    python src/comparaison.py
    python src/comparaison.py --dump lichess_db_standard_rated_2026-01.pgn.zst --graine 1
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd

try:                                                 # permet d'importer outils.py, placé à côté de ce fichier
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
from outils import DOSSIER, DUMP, TAILLE_BLOC, chemin, compter_sites, lire_blocs

RE_SITE = re.compile(rb'^\[Site "([^"]*)"\]', re.M)


def tirer_sites(chemin_dump, k, graine, afficher_tous=20_000_000, taille=TAILLE_BLOC):
    """Réservoir (algorithme R vectorisé par bloc) : k adresses tirées uniformément ; renvoie (adresses, total)."""
    rng = np.random.default_rng(graine)
    reservoir, vus, reste, palier, t0 = [], 0, b"", afficher_tous, time.perf_counter()
    for bloc in lire_blocs(chemin_dump, taille):
        tampon = reste + bloc
        coupe = tampon.rfind(b"\n") + 1                        # on ne traite que des lignes complètes
        tampon, reste = tampon[:coupe], tampon[coupe:]
        m = compter_sites(tampon)
        if not m:
            continue
        rang = np.arange(vus, vus + m)                           # rang global des parties de ce bloc
        j = rng.integers(0, rang + 1)                            # position tirée dans [0, rang]
        utiles = np.flatnonzero((rang < k) | (j < k))
        if utiles.size:
            cible = iter(utiles.tolist())
            pos = next(cible)
            for i, mt in enumerate(RE_SITE.finditer(tampon)):
                if i == pos:
                    site = mt.group(1).decode()
                    if rang[i] < k:
                        reservoir.append(site)                   # remplissage initial
                    else:
                        reservoir[j[i]] = site                   # remplacement
                    pos = next(cible, None)
                    if pos is None:
                        break
        vus += m
        if vus >= palier:
            print(f"  {vus:,} parties lues ({time.perf_counter() - t0:,.0f} s)")
            palier += afficher_tous
    reste_m = compter_sites(reste)
    if reste_m:                                                  # dernière ligne sans saut de ligne final
        for mt in RE_SITE.finditer(reste):
            rang_r = vus
            jj = rng.integers(0, rang_r + 1)
            if rang_r < k:
                reservoir.append(mt.group(1).decode())
            elif jj < k:
                reservoir[jj] = mt.group(1).decode()
            vus += 1
    return reservoir, vus


def main(argv=None):
    a = argparse.ArgumentParser(description="Tire au hasard 10 blocs de 20 parties dans tout le dump (Site seulement).")
    a.add_argument("--dossier", default=DOSSIER, help="dossier des données (défaut : %(default)s)")
    a.add_argument("--dump", default=DUMP, help="dump .pgn.zst (défaut : %(default)s)")
    a.add_argument("--blocs", type=int, default=10, help="nombre de blocs (défaut : %(default)s)")
    a.add_argument("--taille", type=int, default=20, help="parties par bloc (défaut : %(default)s)")
    a.add_argument("--sortie", default="comparaison/comparaison_200.csv")
    a.add_argument("--graine", type=int, default=0, help="graine du tirage (reproductibilité)")
    o = a.parse_args(argv)

    src = chemin(o.dossier, o.dump)
    if not os.path.exists(src):
        sys.exit(f"Dump introuvable : {src}")
    k = o.blocs * o.taille
    t0 = time.perf_counter()
    print(f"Tirage de {k} parties parmi toutes celles de {src}...")
    sites, total = tirer_sites(src, k, o.graine)
    np.random.default_rng(o.graine + 1).shuffle(sites)          # ordre aléatoire avant le découpage en blocs
    n = len(sites)
    df = pd.DataFrame({"bloc": np.arange(n) // o.taille + 1, "position": np.arange(n) % o.taille + 1, "Site": sites})
    sortie = chemin(o.dossier, o.sortie)
    os.makedirs(os.path.dirname(sortie), exist_ok=True)
    df.to_csv(sortie, index=False)
    print(f"{total:,} parties dans le dump ; {n} tirées, en {df['bloc'].nunique()} blocs, écrites dans {sortie} "
          f"({time.perf_counter() - t0:,.0f} s)")
    for b, g in df.groupby("bloc"):
        print(f"\nBloc {b} :")
        print("\n".join(g["Site"]))


if __name__ == "__main__":
    main()
