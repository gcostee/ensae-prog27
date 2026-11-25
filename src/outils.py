"""
Outils communs aux scripts : chemins par défaut, lecture rapide des dumps Lichess, extraction de lignes de parquet.

Lecture rapide d'un dump : le fichier .pgn.zst est décompressé par gros blocs (64 Mo) et découpé en parties sur
la séquence « ligne vide + [Event » ; aucune boucle Python ligne par ligne. Sur un dump mensuel (~90 millions de
parties, ~200 Go décompressés), la durée est dominée par la décompression.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

RACINE = Path(__file__).resolve().parent.parent
DOSSIER = os.environ.get("LICHESS_DATA", str(RACINE / "donnees"))      # dossier des données (hors dépôt git)
DUMP = "lichess_db_standard_rated_2026-01.pgn.zst"                     # nom par défaut du dump mensuel

TAILLE_BLOC = 64 * 1024 * 1024
SEP = b"\n\n[Event "
RE_ENTETE = re.compile(rb'^\[(\w+) "(.*)"\]\s*$', re.M)


def chemin(dossier, nom):
    """Chemin absolu de `nom` : tel quel s'il est absolu, sinon relatif au dossier des données."""
    return nom if os.path.isabs(nom) else os.path.join(dossier, nom)


# ----------------------------------------------------------------------------
# Lecture d'un dump (.pgn.zst ou .pgn)
# ----------------------------------------------------------------------------
def lire_blocs(chemin_pgn, taille=TAILLE_BLOC):
    """Blocs d'octets successifs du fichier (décompressé à la volée si .zst)."""
    with open(chemin_pgn, "rb") as fh:
        flux = zstd.ZstdDecompressor().stream_reader(fh, read_size=1 << 20) if chemin_pgn.endswith(".zst") else fh
        while True:
            bloc = flux.read(taille)
            if not bloc:
                return
            yield bloc


def iter_parties(chemin_pgn):
    """Chaque partie du fichier, en octets (en-têtes + coups). Le découpage se fait sur « \\n\\n[Event »."""
    reste = b""
    for bloc in lire_blocs(chemin_pgn):
        morceaux = (reste + bloc).split(SEP)
        for k in range(1, len(morceaux)):
            morceaux[k] = b"[Event " + morceaux[k]
        reste = morceaux.pop()
        for m in morceaux:
            if m.strip():
                yield m
    if reste.strip():
        yield reste


def decouper(partie):
    """(dictionnaire des en-têtes, texte des coups) d'une partie en octets."""
    i = partie.find(b"\n\n")
    tete, coups = (partie, b"") if i < 0 else (partie[:i], partie[i + 2:])
    h = {k.decode(): v.decode("utf-8", "replace") for k, v in RE_ENTETE.findall(tete)}
    return h, coups.decode("utf-8", "replace").strip()


def compter_sites(bloc):
    """Nombre de parties commençant dans un bloc (une ligne [Site "…"] par partie)."""
    return bloc.count(b'\n[Site "') + bloc.startswith(b'[Site "')


# ----------------------------------------------------------------------------
# Parquet : extraire quelques lignes sans tout charger
# ----------------------------------------------------------------------------
def extraire_lignes(chemin_parquet, indices, colonnes, taille_lot=200_000):
    """Lignes `indices` (numéros de ligne, dans un ordre quelconque) des `colonnes` présentes, lues lot par lot :
    la mémoire reste bornée même si le fichier contient des millions de parties. Renvoie un DataFrame dont la
    ligne i correspond à indices[i]."""
    pf = pq.ParquetFile(chemin_parquet)
    colonnes = [c for c in colonnes if c in pf.schema_arrow.names]
    indices = np.asarray(indices, dtype=np.int64)
    if not colonnes or indices.size == 0:
        return pd.DataFrame(index=range(indices.size))
    ordre = np.argsort(indices, kind="stable")
    tri = indices[ordre]
    morceaux, debut, j = [], 0, 0
    for lot in pf.iter_batches(batch_size=taille_lot, columns=colonnes):
        fin = debut + lot.num_rows
        k = int(np.searchsorted(tri, fin, side="left"))
        if k > j:
            morceaux.append(pa.Table.from_batches([lot]).take(pa.array(tri[j:k] - debut)))
            j = k
        debut = fin
        if j >= tri.size:
            break
    t = pa.concat_tables(morceaux).to_pandas()
    t.index = ordre[: len(t)]
    return t.sort_index().reset_index(drop=True)
