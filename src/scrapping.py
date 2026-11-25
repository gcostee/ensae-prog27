"""
Étape 3a : récupère les « Lichess Game of the Month » (parties du mois) publiées sur le blog du compte Lichess
et les écrit dans un fichier PGN au format de la base Lichess (en-têtes + coups de la ligne principale).

Fonctionnement :
  1. liste des billets : pages du blog https://lichess.org/@/Lichess/blog?page=1, 2, … (on garde les billets dont
     l'adresse contient « game-of-the-month »), complétée par les liens trouvés dans le billet de départ
     (--depart) et, si besoin, par une liste fournie à la main (--urls fichier.txt, une adresse par ligne) ;
  2. dans chaque billet : liens vers des études (/study/XXXXXXXX, dont les intégrations /study/embed/…) et
     vers des parties (/XXXXXXXX, /embed/game/XXXXXXXX) ;
  3. téléchargement par l'API publique de Lichess : https://lichess.org/api/study/{id}.pgn (tous les chapitres)
     et https://lichess.org/game/export/{id} ; chaque chapitre ou partie de plus de --min-plies demi-coups et au
     résultat décisif devient une partie du fichier ;
  4. doublons supprimés (même suite de coups) ; les billets des mois exclus (--exclure, par défaut janvier et
     février 2026, mois de la partie du mois indiqué dans le titre du billet) sont ignorés. Quand un chapitre d'étude
     n'indique pas le résultat (« * »), il est lu dans la partie Lichess d'origine si l'en-tête Site la désigne.

Sorties (dans <dossier>/pu/) :
  gotm.pgn          les parties du mois, en-têtes [GOTM "AAAA-MM"] et [GOTMSource "<billet>"] ajoutés
  gotm_index.csv    une ligne par partie : mois, billet, source (étude ou partie), joueurs, résultat, demi-coups

L'API de Lichess demande une seule requête à la fois : une pause (--pause, 1 s) sépare les requêtes et une
réponse 429 (trop de requêtes) déclenche une attente d'une minute.

Exemple :
    python src/scraping.py
    python src/scraping.py --urls mes_billets.txt --exclure 2026-01,2026-02
"""
import argparse
import csv
import hashlib
import io
import os
import re
import sys
import time

import chess
import chess.pgn

try:                                                 # permet d'importer outils.py, placé à côté de ce fichier
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
from outils import DOSSIER, chemin

SITE = "https://lichess.org"
BLOG = SITE + "/@/Lichess/blog"
DEPART = SITE + "/@/Lichess/blog/lichess-game-of-the-month-august-26/ekVdAfDN"
EXCLURE = "2026-01,2026-02"          # mois exclus (mois de la partie du mois, AAAA-MM)
MIN_PLIES = 20
PAUSE = 1.0
MAX_PAGES = 100
AGENT = "lichess-features/1.0 (projet d'etude ; https://github.com)"

MOIS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
        "nov": 11, "dec": 12}
RE_BILLET = re.compile(r'(?:https://lichess\.org)?(/@/Lichess/blog/([a-z0-9-]+)/([A-Za-z0-9]{8}))(?![A-Za-z0-9])')
RE_MOIS = re.compile(r"game-of-the-month-([a-z]+)-(\d{2,4})(?![0-9])")
RE_ETUDE = re.compile(r'/study/(?:embed/)?([A-Za-z0-9]{8})(?![A-Za-z0-9])')
RE_PARTIE = re.compile(r'(?:href|src)="(?:https://lichess\.org)?/(?:embed/(?:game/)?)?([A-Za-z0-9]{8})'
                       r'(?:[A-Za-z0-9]{4})?(?:/(?:white|black))?(?:#\d+)?(?:\?[^"]*)?"')
RE_SITE_PARTIE = re.compile(r"lichess\.org/([A-Za-z0-9]{8})(?![A-Za-z0-9])")


# ----------------------------------------------------------------------------
# Réseau
# ----------------------------------------------------------------------------
class Client:
    """Requêtes HTTP polies : une à la fois, pause entre deux requêtes, attente d'une minute sur 429."""

    def __init__(self, pause=PAUSE):
        import requests
        self.s = requests.Session()
        self.s.headers["User-Agent"] = AGENT
        self.pause, self.dernier = pause, 0.0

    def get(self, url, accept=None, essais=4):
        for k in range(essais):
            attente = self.pause - (time.monotonic() - self.dernier)
            if attente > 0:
                time.sleep(attente)
            try:
                r = self.s.get(url, headers={"Accept": accept} if accept else None, timeout=60)
            except Exception as e:                                  # réseau indisponible, délai dépassé…
                print(f"    erreur réseau ({e.__class__.__name__}) sur {url} ; nouvel essai")
                time.sleep(5 * (k + 1))
                continue
            finally:
                self.dernier = time.monotonic()
            if r.status_code == 429:
                print("    trop de requêtes (429) : attente de 61 s")
                time.sleep(61)
                continue
            if r.status_code == 404:
                return None
            if r.ok:
                return r.text
            print(f"    réponse {r.status_code} pour {url}")
            time.sleep(3)
        return None


# ----------------------------------------------------------------------------
# Analyse des pages
# ----------------------------------------------------------------------------
def mois_du_billet(url):
    """« …/lichess-game-of-the-month-december-24/… » -> "2024-12" (None si non reconnu)."""
    m = RE_MOIS.search(url)
    if not m:
        return None
    mois = MOIS.get(m.group(1)[:3])
    if not mois:
        return None
    an = int(m.group(2))
    return f"{an + 2000 if an < 100 else an:04d}-{mois:02d}"


def billets_gotm(html):
    """Adresses des billets « game of the month » cités dans une page."""
    return {SITE + ch for ch, slug, _ in RE_BILLET.findall(html) if "game-of-the-month" in slug}


def tous_les_billets(html):
    return {ch for ch, _, _ in RE_BILLET.findall(html)}


def corps_du_billet(html):
    """Le contenu rédigé du billet (pour ne pas ramasser les liens de navigation) ; la page entière à défaut."""
    i = html.find("ublog-post__markup")
    if i < 0:
        return html
    fins = [j for j in (html.find(m, i) for m in ("ublog-post__footer", "ublog-post__like", "</article>")) if j > 0]
    return html[i:min(fins)] if fins else html[i:]


def liens_du_billet(html):
    """(identifiants d'études, identifiants de parties) cités dans le corps du billet."""
    corps = corps_du_billet(html)
    etudes = list(dict.fromkeys(RE_ETUDE.findall(corps)))
    parties = [g for g in dict.fromkeys(RE_PARTIE.findall(corps))
               if any(c.isupper() or c.isdigit() for c in g)]           # écarte /training, /practice, /analysis…
    return etudes, [g for g in parties if g not in etudes]


def lister_billets(client, depart, max_pages, urls_manuelles):
    """Tous les billets « game of the month » : pages du blog + billet de départ + liste manuelle."""
    billets, vus_blog = set(urls_manuelles), set()
    for page in range(1, max_pages + 1):
        html = client.get(f"{BLOG}?page={page}")
        tous = tous_les_billets(html or "")
        if not tous or tous <= vus_blog:                            # page vide ou déjà vue : fin du blog
            break
        vus_blog |= tous
        nouveaux = billets_gotm(html) - billets
        billets |= nouveaux
        print(f"  page {page} du blog : {len(tous)} billets, {len(nouveaux)} « game of the month » nouveaux")
    a_voir, vus = [depart] if depart else [], set()
    while a_voir:                                                   # liens d'un billet à l'autre
        url = a_voir.pop()
        if url in vus:
            continue
        vus.add(url)
        html = client.get(url)
        if html:
            nouveaux = billets_gotm(html) - billets - {url}
            billets |= nouveaux | {url}
            a_voir.extend(nouveaux)
    return sorted(billets, key=lambda u: mois_du_billet(u) or "")


# ----------------------------------------------------------------------------
# Parties
# ----------------------------------------------------------------------------
def lire_pgn(texte):
    """Toutes les parties d'un texte PGN (chapitres d'une étude, ou une partie)."""
    flux, parties = io.StringIO(texte), []
    while True:
        g = chess.pgn.read_game(flux)
        if g is None:
            return parties
        parties.append(g)


def resultat(g, board):
    r = g.headers.get("Result", "*")
    if r in ("1-0", "0-1", "1/2-1/2"):
        return r
    if board.is_checkmate():
        return "0-1" if board.turn == chess.WHITE else "1-0"
    return "*"


def id_partie_origine(h):
    """Identifiant de la partie Lichess d'origine d'un chapitre d'étude (en-tête Site ou GameURL), ou None."""
    for k in ("Site", "GameURL"):
        m = RE_SITE_PARTIE.search(h.get(k, ""))
        if m and "/study/" not in h.get(k, ""):
            return m.group(1)
    return None


def partie_propre(g, mois, billet, source, min_plies, garder_nulles, client=None):
    """Ligne principale d'une partie (sans variantes ni commentaires) avec les en-têtes utiles, ou None."""
    h = g.headers
    fen = h.get("FEN")
    if (h.get("Variant", "Standard").lower() not in ("standard", "")
            or (fen and fen.split()[:4] != chess.STARTING_FEN.split()[:4])):
        return None                                                 # variante ou position de départ non standard
    board, sans = g.board(), []
    for coup in g.mainline_moves():
        sans.append(board.san(coup))
        board.push(coup)
    if len(sans) < min_plies:
        return None                                                 # chapitre d'introduction, fragment…
    res = resultat(g, board)
    gid = id_partie_origine(h)
    if (res == "*" or "WhiteElo" not in h) and gid and client is not None:   # complété par la partie d'origine
        orig = lire_pgn(client.get(f"{SITE}/game/export/{gid}?clocks=false&evals=false",
                                   accept="application/x-chess-pgn") or "")
        if orig:
            for k, v in orig[0].headers.items():
                if k not in h or h[k] in ("", "?", "*", "????.??.??"):
                    h[k] = v
            res = resultat(g, board) if res == "*" else res
    if res == "*" or (res == "1/2-1/2" and not garder_nulles):
        return None
    site = f"{SITE}/{gid}" if gid else h.get("Site", "")
    nouv = chess.pgn.Game()
    nouv.headers.clear()
    nouv.headers.update({
        "Event": f"Lichess Game of the Month {mois}", "Site": site or source,
        "Date": h.get("UTCDate", h.get("Date", "????.??.??")), "White": h.get("White", "?"),
        "Black": h.get("Black", "?"), "Result": res})
    for k in ("WhiteElo", "BlackElo", "WhiteTitle", "BlackTitle", "ECO", "Opening", "TimeControl", "Termination"):
        if h.get(k) and h[k] != "?":
            nouv.headers[k] = h[k]
    nouv.headers.update({"GOTM": mois, "GOTMSource": billet, "Source": source})
    noeud = nouv
    for coup in g.mainline_moves():
        noeud = noeud.add_variation(coup)
    return dict(jeu=nouv, sans=sans, res=res, date=nouv.headers["Date"])


def recuperer(client, billets, exclus, min_plies, garder_nulles):
    parties, vues = [], set()
    for billet in billets:
        mois = mois_du_billet(billet)
        if mois is None:
            print(f"  ? mois non reconnu, billet ignoré : {billet}")
            continue
        if mois in exclus:
            print(f"  {mois} exclu : {billet}")
            continue
        html = client.get(billet)
        if not html:
            print(f"  {mois} billet inaccessible : {billet}")
            continue
        etudes, jeux = liens_du_billet(html)
        n_avant = len(parties)
        sources = ([(f"{SITE}/study/{e}", f"{SITE}/api/study/{e}.pgn") for e in etudes]
                   + [(f"{SITE}/{g}", f"{SITE}/game/export/{g}?clocks=false&evals=false") for g in jeux])
        for source, url in sources:
            texte = client.get(url, accept="application/x-chess-pgn")
            for g in lire_pgn(texte or ""):
                p = partie_propre(g, mois, billet, source, min_plies, garder_nulles, client)
                if p is None:
                    continue
                cle = hashlib.md5((" ".join(p["sans"]) + p["res"]).encode()).hexdigest()
                if cle in vues:
                    continue
                vues.add(cle)
                parties.append({**p, "mois": mois, "billet": billet, "source": source})
        print(f"  {mois} : {len(etudes)} étude(s), {len(jeux)} partie(s) citée(s) -> "
              f"{len(parties) - n_avant} partie(s) retenue(s)")
    return parties


def ecrire(parties, dossier_sortie):
    os.makedirs(dossier_sortie, exist_ok=True)
    chemin_pgn = os.path.join(dossier_sortie, "gotm.pgn")
    with open(chemin_pgn, "w", encoding="utf-8") as f:
        for p in parties:
            f.write(p["jeu"].accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False)))
            f.write("\n\n")
    with open(os.path.join(dossier_sortie, "gotm_index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mois", "billet", "source", "Site", "White", "Black", "Result", "n_plies"])
        for p in parties:
            h = p["jeu"].headers
            w.writerow([p["mois"], p["billet"], p["source"], h["Site"], h["White"], h["Black"], p["res"],
                        len(p["sans"])])
    return chemin_pgn


def main(argv=None):
    a = argparse.ArgumentParser(description="Récupère les parties du mois Lichess dans un fichier PGN.")
    a.add_argument("--dossier", default=DOSSIER, help="dossier des données (défaut : %(default)s)")
    a.add_argument("--sortie", default="pu", help="sous-dossier de sortie (défaut : %(default)s)")
    a.add_argument("--depart", default=DEPART, help="billet de départ (ses liens vers d'autres billets sont suivis)")
    a.add_argument("--urls", default=None, help="fichier texte : une adresse de billet par ligne (complète la liste)")
    a.add_argument("--exclure", default=EXCLURE, help="mois exclus AAAA-MM séparés par des virgules (défaut : %(default)s)")
    a.add_argument("--max-pages", type=int, default=MAX_PAGES, help="pages du blog parcourues au plus")
    a.add_argument("--min-plies", type=int, default=MIN_PLIES, help="demi-coups minimum d'une partie retenue")
    a.add_argument("--garder-nulles", action="store_true", help="garder aussi les parties nulles")
    a.add_argument("--pause", type=float, default=PAUSE, help="secondes entre deux requêtes (défaut : %(default)s)")
    o = a.parse_args(argv)

    t0 = time.perf_counter()
    client = Client(o.pause)
    exclus = {m.strip() for m in o.exclure.split(",") if m.strip()}
    manuelles = []
    if o.urls:
        with open(o.urls, encoding="utf-8") as f:
            manuelles = [l.strip() for l in f if l.strip().startswith("http")]
    print("Recherche des billets « Lichess Game of the Month »...")
    billets = lister_billets(client, o.depart, o.max_pages, manuelles)
    print(f"{len(billets)} billets trouvés ({', '.join(sorted({m for m in map(mois_du_billet, billets) if m}))})")
    if not billets:
        sys.exit("Aucun billet trouvé : vérifier la connexion, ou fournir la liste des billets avec --urls.")
    print("Téléchargement des parties...")
    parties = recuperer(client, billets, exclus, o.min_plies, o.garder_nulles)
    chemin_pgn = ecrire(parties, chemin(o.dossier, o.sortie))
    mois = sorted({p["mois"] for p in parties})
    print(f"\n{len(parties)} parties du mois écrites dans {chemin_pgn}")
    print(f"Mois couverts : {len(mois)} ({mois[0] if mois else '-'} à {mois[-1] if mois else '-'}) ; "
          f"exclus : {', '.join(sorted(exclus)) or 'aucun'}")
    print(f"Temps écoulé : {time.perf_counter() - t0:,.0f} s")


if __name__ == "__main__":
    main()
