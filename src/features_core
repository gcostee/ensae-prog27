"""
Calcul des features d'une partie en UN SEUL passage sur la liste des coups (SAN).

Choix de performance :
  * pas de python-chess.pgn (pas d'arbre de nœuds) : on découpe le texte des coups et on appelle
    board.parse_san + board.push, seules opérations qui demandent le plateau ;
  * captures / échecs / roques / coups de roi / promotions repérés par le TEXTE du coup SAN
    ('x', '+', '#', 'O', 'K', '=') ou par move.promotion : aucun appel supplémentaire à python-chess ;
  * matériel suivi de façon incrémentale (valeur de la pièce capturée), jamais recompté ;
  * les features qui ont besoin d'un historique (sacrifices, échanges inégaux) lisent un petit
    dictionnaire {demi-coup: valeur capturée} après la boucle ;
  * pions passés : calculés une seule fois, sur la position finale, avec des masques précalculés ;
  * features d'ÉVALUATION : les évaluations arrivent sous forme d'octets (int16, lecture sans copie par
    np.frombuffer, aucun texte à analyser) ; tout est ensuite vectorisé avec NumPy : quelques dizaines
    d'opérations sur des tableaux d'une centaine d'éléments, sans boucle Python sur les demi-coups.

Hypothèses : partie jouée depuis la position initiale ; SAN avec '+' / '#' (format Lichess).
Les features 'elo' et 'rarete_ouverture' sont calculées dans le script principal (colonnes du DataFrame).

Format de la colonne 'Evals' (écrite par filtrer_parties.py) : un tableau d'int16 par partie, un par
demi-coup, du point de vue des BLANCS :  centipions (bornés à ±9999) ;  mat en k coups pour les blancs
= +(10000 + k), pour les noirs = -(10000 + k) ;  -32768 = évaluation absente.
"""
import re
from array import array

import chess
import numpy as np

# ----------------------------------------------------------------------------
# Paramètres : matériel
# ----------------------------------------------------------------------------
VALEUR = (0, 1, 3, 3, 5, 9, 0)   # valeur par type de pièce python-chess : pion=1, C=2, F=3, T=4, D=5, R=6
SEUIL_SIGNE = 2        # n_changements_signe : l'avantage compte à partir de |bilan| >= 2
SEUIL_DEFICIT = -2     # n_plies_deficit : déficit « significatif » du vainqueur : bilan <= -2
SEUIL_REMONTADA = -3   # remontada : déficit « important » : bilan <= -3
TOL_EQUIV = 1          # capture « équivalente » : |valeur perdue - valeur reprise| <= 1
SEUIL_ECHANGE = 2      # échange « inégal » : |solde matériel de la séquence| >= 2
ECART_ECHANGE = 1      # captures « rapprochées » : sur demi-coups consécutifs (2 = tolère 1 coup intermédiaire)

# ----------------------------------------------------------------------------
# Paramètres : évaluation
# ----------------------------------------------------------------------------
SENT = -32768          # évaluation absente
TAU_B = 0.0            # coup « meilleur » : perte <= TAU_B centipions (0 : l'évaluation ne baisse pas)
TAU_G = 50.0           # coup « brillant » : gain >= TAU_G centipions
GAFFE = 30.0           # gaffe : perte de chances de gain >= 30 points (sur 100)
FAUTE = 10.0           # faute : perte de chances de gain >= 10 points (les gaffes en font partie)
THETA = 50.0           # changement de signe de l'évaluation : on ignore |cp| < THETA
MANQ_MAX = 0.05        # couverture minimale des évaluations : 95 % des demi-coups
K_WIN = 0.00368208     # chances de gain = 100 / (1 + exp(-K_WIN * cp))   (formule de Lichess)
SEUIL_SAC_BAL = -2     # sacrifice validé : bilan matériel du vainqueur <= -2 ...
FENETRE_DAME = 4       # n_sacrifices_dame_valides : évaluation 4 demi-coups après la perte de la dame

NAN = float("nan")
_RE_COMMENT = re.compile(r"\{[^}]*\}")
_RE_COM = re.compile(r"\{([^}]*)\}")
_RE_VAR_OU_COM = re.compile(r"(\{[^}]*\})|(\([^()]*\))")   # commentaire {..} (conservé) ou variation (..) (supprimée)
_RESULTATS = {"1-0", "0-1", "1/2-1/2", "*"}
_S = np.tile(np.array([1.0, -1.0]), 2000)   # +1 si les blancs jouent au demi-coup t (t=1 : blancs)
_ARANGE = np.arange(4000)

# Features de matériel / structure (tuple renvoyé en premier)
FEATURES_MATERIEL = [
    "n_plies",
    "n_captures", "n_changements_signe", "amplitude_bal",
    "min_bal_gagnant", "n_plies_deficit", "remontada",
    "n_sacrifices_dame", "n_sacrifices_pieces", "n_echanges_inegaux",
    "n_echecs", "n_captures_avec_echec", "n_promotions", "n_sous_promotions",
    "n_coups_sans_capture_max", "n_coups_roque", "roques_opposes", "distance_roi_max",
    "n_coups_roi", "n_pions_passes_fin",
    "piece_mateuse",                       # catégorielle (texte)
]
# Features d'évaluation (tuple renvoyé ensuite, dans cet ordre)
FEATURES_EVAL = [
    # précision
    "acpl_gagnant", "acpl_perdant", "accuracy_gagnant", "accuracy_perdant", "part_fautes_perdant",
    # erreurs
    "n_gaffes_gagnant", "n_gaffes_perdant", "n_fautes_gagnant", "n_fautes_perdant", "perte_max_gagnant",
    # meilleurs coups et coups brillants
    "n_meilleurs_gagnant", "n_meilleurs_total", "n_meilleurs_gagnant_apres10", "serie_meilleurs_gagnant",
    "n_brillants_gagnant", "n_brillants_total",
    # retournements d'avantage
    "n_changements_signe_eval", "win_min_gagnant", "remontada_eval", "n_avantages_gaches",
    "n_retournements_majeurs", "ply_decisif_rel", "part_equilibree",
    # variations de l'évaluation
    "variance_eval", "volatilite_eval", "swing_max", "n_swings_majeurs", "variation_totale", "amplitude_eval",
    # sacrifices vérifiés et mat annoncé
    "n_plies_sacrifice_valide", "sacrifice_valide_max", "n_sacrifices_dame_valides",
    "n_brillants_avec_sacrifice", "n_plies_mat_gagnant", "mat_max_annonce",
    # phases
    "eval_sortie_ouverture", "acpl_gagnant_milieu", "acpl_gagnant_fin", "n_gaffes_fin",
]
FEATURES_PARTIE = FEATURES_MATERIEL + FEATURES_EVAL
# Features à valeurs décimales -> nombre de décimales conservées pour les histogrammes (regroupement)
DECIMALES = {
    "acpl_gagnant": 1, "acpl_perdant": 1, "accuracy_gagnant": 1, "accuracy_perdant": 1,
    "part_fautes_perdant": 2, "perte_max_gagnant": 1, "win_min_gagnant": 1, "ply_decisif_rel": 2,
    "part_equilibree": 2, "variance_eval": 1, "volatilite_eval": 1, "swing_max": 1,
    "variation_totale": 1, "amplitude_eval": 1, "acpl_gagnant_milieu": 1, "acpl_gagnant_fin": 1,
}
NAN_EVAL = (NAN,) * len(FEATURES_EVAL)
ECHEC = tuple(None if f == "piece_mateuse" else NAN for f in FEATURES_PARTIE)


# ----------------------------------------------------------------------------
# Lecture du texte PGN : variations, évaluations (utilisé par filtrer_parties.py et le diagnostic)
# ----------------------------------------------------------------------------
def sans_variations(texte):
    """Supprime les variations entre parenthèses (analyses alternatives) en conservant les commentaires {..},
    même s'ils contiennent des parenthèses. Gère l'imbrication."""
    if "(" not in texte:
        return texte
    prec = None
    while prec != texte:
        prec = texte
        texte = _RE_VAR_OU_COM.sub(lambda m: m.group(1) or "", texte)
    return texte


def nb_coups_texte(morceau):
    """Nombre de coups (ni numéros de coup, ni résultat) dans un morceau de texte PGN."""
    return sum(1 for t in morceau.split() if t[-1] != "." and t[0] != "$" and t not in _RESULTATS)   # "$n" = NAG


def nb_demi_coups(moves):
    """Nombre de demi-coups d'un texte de coups, selon la même règle que features_partie (sans plateau)."""
    n = 0
    for tok in moves.split():
        if "." in tok:
            tok = tok[tok.rfind(".") + 1:]
        if tok and tok[0] not in "0123456789*$":
            n += 1
    return n


def valeur_eval(commentaire):
    """Évaluation du commentaire { [%eval 0.17] [%clk ...] } en int16, du point de vue des blancs :
    centipions (bornés à ±9999) ; mat en k coups : +(10000+k) pour les blancs, -(10000+k) pour les noirs ;
    SENT si absente."""
    i = commentaire.find("[%eval ")
    if i < 0:
        return SENT
    s = commentaire[i + 7:commentaire.find("]", i)].strip()
    try:
        if s[0] == "#":
            k = int(s[1:])
            return SENT if k == 0 else (10000 + k if k > 0 else -10000 + k)
        return max(-9999, min(9999, round(float(s) * 100)))
    except (ValueError, IndexError):
        return SENT


def extraire_evals(ligne):
    """Une évaluation par demi-coup (SENT si le coup n'a pas d'évaluation), en octets int16 natifs.
    Chaque commentaire annote le coup qui le précède ; les commentaires supplémentaires (texte d'analyse)
    et les variations sont ignorés ; les coups sans commentaire reçoivent SENT."""
    ligne = sans_variations(ligne)
    vals, pos = [], 0
    for m in _RE_COM.finditer(ligne):
        n = nb_coups_texte(ligne[pos:m.start()])
        pos = m.end()
        if n:
            vals.extend([SENT] * (n - 1))
            vals.append(valeur_eval(m.group(1)))
    vals.extend([SENT] * nb_coups_texte(ligne[pos:]))
    return array("h", vals).tobytes()


# ----------------------------------------------------------------------------
# Masques de pions passés (précalculés une fois)
# ----------------------------------------------------------------------------
def _masque_passe(sq, blanc):
    """Cases devant le pion (même colonne et colonnes voisines) : s'il n'y a aucun pion adverse
    dessus, le pion est passé."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    m = 0
    for ff in (f - 1, f, f + 1):
        if 0 <= ff < 8:
            for rr in (range(r + 1, 8) if blanc else range(0, r)):
                m |= 1 << chess.square(ff, rr)
    return m


PASSE_B = [_masque_passe(s, True) for s in range(64)]
PASSE_N = [_masque_passe(s, False) for s in range(64)]


# ----------------------------------------------------------------------------
# Features d'évaluation (vectorisées)
# ----------------------------------------------------------------------------
def _serie_max(b):
    """Longueur de la plus longue suite de True dans un tableau booléen."""
    if not b.any():
        return 0
    f = np.flatnonzero(~b)
    if f.size == 0:
        return b.size
    return int((np.diff(f, prepend=-1, append=b.size) - 1).max())


def _moy(x):
    return x.mean() if x.size else NAN


def features_eval(buf, sgn, n, bals, capt, dames_sac, mat_fin):
    """Features d'évaluation d'une partie.
    buf : octets int16 (une évaluation par demi-coup) ; sgn : +1 si les blancs gagnent, -1 sinon ;
    n : nombre de demi-coups ; bals : bilan matériel (blancs - noirs) après chaque demi-coup ;
    capt : {demi-coup: valeur capturée} ; dames_sac : demi-coups où une dame a été perdue sans contrepartie ;
    mat_fin : la partie finit par un mat.
    Renvoie un tuple dans l'ordre de FEATURES_EVAL (tout à NaN si l'évaluation est inutilisable)."""
    if buf is None or not sgn or n < 22:
        return NAN_EVAL
    E = np.frombuffer(buf, dtype=np.int16)
    if E.size != n:                                     # évaluations non alignées sur les coups
        return NAN_EVAL
    E = E.astype(np.float64)

    # --- évaluations manquantes : couverture >= 95 %, sinon abandon ; le reste est comblé par la dernière valeur
    manq = E == SENT
    if manq.any():
        if manq.sum() > MANQ_MAX * n:
            return NAN_EVAL
        idx = np.where(manq, 0, _ARANGE[:n])
        np.maximum.accumulate(idx, out=idx)
        E = E[idx]
        E[E == SENT] = 0.0

    # --- décodage : centipions bornés à ±1000 (un mat vaut ±1000 : l'encodage ±(10000+k) est écrêté)
    cp = np.clip(E, -1000.0, 1000.0)
    if mat_fin:
        cp[-1] = sgn * 1000.0                           # le dernier coup est un mat : évaluation = victoire
    w = 100.0 / (1.0 + np.exp(-K_WIN * cp))             # chances de gain des blancs (0-100)
    ww = w if sgn == 1 else 100.0 - w                   # chances de gain du VAINQUEUR

    # --- variations d'un demi-coup au suivant, pour t = 2..n (indice j = t-2)
    S1 = _S[1:n]                                        # +1 si le camp qui joue est blanc
    dcp = (cp[1:] - cp[:-1]) * S1                       # > 0 : le coup améliore l'évaluation pour celui qui joue
    dwr = w[1:] - w[:-1]
    perte = np.maximum(-dcp, 0.0)                       # ℓ_t : centipions perdus par le coup
    gain = np.maximum(dcp, 0.0)                         # g_t : centipions gagnés par le coup
    lam = np.maximum(-dwr * S1, 0.0)                    # λ_t : chances de gain perdues par le coup
    adw = np.abs(dwr)

    # coups du vainqueur : t impair (j impair) s'il joue les blancs, t pair (j pair) sinon
    o = 1 if sgn == 1 else 0
    pw, pl = perte[o::2], perte[1 - o::2]
    lw, ll = lam[o::2], lam[1 - o::2]
    gw = gain[o::2]

    # ===== PRÉCISION =====
    # acpl_gagnant / acpl_perdant : moyenne des centipions perdus par coup (plus bas = plus précis)
    acpl_g, acpl_p = pw.mean(), pl.mean()
    # accuracy_* : formule par coup de Lichess sur les chances de gain perdues, moyenne simple, échelle 0-100
    acc_g = np.clip(103.1668 * np.exp(-0.04354 * lw) - 3.1669, 0.0, 100.0).mean()
    acc_p = np.clip(103.1668 * np.exp(-0.04354 * ll) - 3.1669, 0.0, 100.0).mean()
    # part_fautes_perdant : part des chances de gain perdues par le perdant (1 = victoire offerte)
    sl, sw = ll.sum(), lw.sum()
    part_p = sl / (sl + sw) if (sl + sw) > 0 else NAN

    # ===== ERREURS =====
    # n_gaffes_* : coups perdant >= GAFFE points de chances de gain ; n_fautes_* : >= FAUTE points
    # perte_max_gagnant : plus grande perte de chances de gain du vainqueur sur un coup
    ng_g, ng_p = np.count_nonzero(lw >= GAFFE), np.count_nonzero(ll >= GAFFE)
    nf_g, nf_p = np.count_nonzero(lw >= FAUTE), np.count_nonzero(ll >= FAUTE)
    perte_max = lw.max()

    # ===== MEILLEURS COUPS ET COUPS BRILLANTS =====
    # n_meilleurs_gagnant : coups du vainqueur dont la perte est <= TAU_B (l'évaluation ne baisse pas)
    bw = pw <= TAU_B
    nm_g = np.count_nonzero(bw)
    # n_meilleurs_total : idem pour les deux camps
    nm_t = np.count_nonzero(perte <= TAU_B)
    # n_meilleurs_gagnant_apres10 : coups du vainqueur après le 10e coup (t > 20)
    nm_10 = np.count_nonzero(perte[19 if o else 20::2] <= TAU_B)
    # serie_meilleurs_gagnant : plus longue suite de coups consécutifs du vainqueur qui sont des « meilleurs coups »
    serie = _serie_max(bw)
    # n_brillants_gagnant / total : coups dont le gain d'évaluation est >= TAU_G
    nb_g, nb_t = np.count_nonzero(gw >= TAU_G), np.count_nonzero(gain >= TAU_G)

    # ===== RETOURNEMENTS D'AVANTAGE =====
    # n_changements_signe_eval : changements de signe de l'évaluation, valeurs |cp| < THETA ignorées
    sg = np.sign(cp)
    sg[np.abs(cp) < THETA] = 0.0
    nz = sg[sg != 0.0]
    chg_signe = np.count_nonzero(nz[1:] != nz[:-1])
    # win_min_gagnant : plus basse chance de gain du vainqueur
    wmin = ww.min()
    # remontada_eval : le vainqueur est descendu à <= 15 % puis finit à >= 85 %
    rem = int(wmin <= 15.0 and ww[-1] >= 85.0)
    # n_avantages_gaches : passages du vainqueur de >= 80 % à <= 50 % (avec hystérésis)
    z = (ww >= 80.0).view(np.int8) - (ww <= 50.0).view(np.int8)
    z = z[z != 0]
    n_gach = np.count_nonzero((z[:-1] == 1) & (z[1:] == -1))
    # n_retournements_majeurs : alternances entre >= 70 % et <= 30 % de chances de gain
    z2 = (ww >= 70.0).view(np.int8) - (ww <= 30.0).view(np.int8)
    z2 = z2[z2 != 0]
    n_ret = np.count_nonzero(z2[1:] != z2[:-1])
    # ply_decisif_rel : demi-coup à partir duquel le vainqueur reste >= 80 %, divisé par n (1 = jamais décidée)
    bas = np.flatnonzero(ww < 80.0)
    decisif = min((bas[-1] + 2 if bas.size else 1) / n, 1.0)
    # part_equilibree : proportion de demi-coups entre 35 % et 65 % de chances de gain
    equi = np.count_nonzero((ww >= 35.0) & (ww <= 65.0)) / n

    # ===== VARIATIONS DE L'ÉVALUATION =====
    # variance_eval : variance des chances de gain ; volatilite_eval : écart-type de leurs variations
    var_e, vol_e = w.var(), dwr.std()
    # swing_max : plus grande variation de chances de gain en un coup ; n_swings_majeurs : variations >= 20 points
    swing, n_sw = adw.max(), np.count_nonzero(adw >= 20.0)
    # variation_totale : somme des |variations| par demi-coup ; amplitude_eval : max - min des chances de gain
    var_tot, ampl = adw.sum() / n, ww.max() - wmin

    # ===== SACRIFICES VÉRIFIÉS ET MAT ANNONCÉ =====
    bal_w = sgn * np.array(bals, dtype=np.float64)      # bilan matériel du vainqueur après chaque demi-coup
    capm = np.zeros(n, dtype=bool)                      # demi-coups qui sont des captures
    if capt:
        capm[np.fromiter(capt, dtype=np.int64, count=len(capt)) - 1] = True
    # n_plies_sacrifice_valide : demi-coups (positions calmes) où le vainqueur est à bilan <= -2 ET évaluation >= 0
    cond = (bal_w <= SEUIL_SAC_BAL) & (sgn * cp >= 0.0) & ~capm
    n_sv = np.count_nonzero(cond)
    # sacrifice_valide_max : plus grand déficit matériel atteint pendant ces demi-coups
    sv_max = -bal_w[cond].min() if n_sv else 0.0
    # n_sacrifices_dame_valides : dames du vainqueur perdues sans contrepartie, avec évaluation >= 0
    # pour lui 4 demi-coups plus tard
    perd_par = 0 if sgn == 1 else 1                     # parité des demi-coups joués par le perdant
    n_dv = 0
    for p in dames_sac:
        if (p & 1) == perd_par and sgn * cp[min(p + FENETRE_DAME, n) - 1] >= 0.0:
            n_dv += 1
    # n_brillants_avec_sacrifice : coups brillants du vainqueur joués alors qu'il est à bilan <= -2
    n_bs = np.count_nonzero((gw >= TAU_G) & (bal_w[:-1][o::2] <= SEUIL_SAC_BAL))
    # n_plies_mat_gagnant : demi-coups avec un mat annoncé en faveur du vainqueur ; mat_max_annonce : plus grand k
    mat = (np.abs(E) >= 10000.0) & (np.sign(E) == sgn)
    n_mat = np.count_nonzero(mat)
    mat_max = np.abs(E[mat]).max() - 10000.0 if n_mat else 0.0

    # ===== PHASES =====
    # eval_sortie_ouverture : évaluation (centipions, vue du vainqueur) après 10 coups (t = 20)
    ev20 = sgn * cp[19]
    # acpl_gagnant_milieu : précision du vainqueur aux demi-coups 21 à 60 ; acpl_gagnant_fin : sur les 20 derniers
    s = 19 if o else 20
    a = max(0, n - 21)
    s2 = a if (a & 1) == o else a + 1
    # n_gaffes_fin : gaffes des deux camps dans les 20 derniers demi-coups
    return tuple(np.array([
        acpl_g, acpl_p, acc_g, acc_p, part_p,
        ng_g, ng_p, nf_g, nf_p, perte_max,
        nm_g, nm_t, nm_10, serie, nb_g, nb_t,
        chg_signe, wmin, rem, n_gach, n_ret, decisif, equi,
        var_e, vol_e, swing, n_sw, var_tot, ampl,
        n_sv, sv_max, n_dv, n_bs, n_mat, mat_max,
        ev20, _moy(perte[s:59:2]), _moy(perte[s2:n - 1:2]), np.count_nonzero(lam[a:n - 1] >= GAFFE),
    ], dtype=np.float64).tolist())


# ----------------------------------------------------------------------------
# Features d'une partie
# ----------------------------------------------------------------------------
def features_partie(moves, result, evals=None):
    """moves : texte des coups ("1. e4 e5 2. Nf3 ..."), result : "1-0" / "0-1" / autre,
    evals : octets int16 des évaluations (ou None). Renvoie un tuple dans l'ordre de FEATURES_PARTIE."""
    if "(" in moves:                                   # variations éventuelles
        moves = sans_variations(moves)
    if "{" in moves:                                   # commentaires éventuels
        moves = _RE_COMMENT.sub("", moves)
    sgn = 1 if result == "1-0" else (-1 if result == "0-1" else 0)   # 0 : pas de vainqueur

    board = chess.Board()
    parse, push, type_en = board.parse_san, board.push, board.piece_type_at

    ply = 0                              # nombre de demi-coups joués
    dernier = ""                         # dernier coup joué (texte SAN) : sert à repérer le mat
    bal = bal_min = bal_max = 0          # bilan matériel RELATIF = matériel blanc - matériel noir
    bals = []                            # bilan matériel après chaque demi-coup (pour les features d'évaluation)
    bals_add = bals.append
    defi = n_def = 0                     # défi : le vainqueur est-il en déficit significatif ?
    n_chg_signe = 0
    cote = 0                             # camp ayant l'avantage (+1 blanc / -1 noir / 0 aucun) vu à ce jour
    capt = {}                            # {numéro de demi-coup: valeur de la pièce capturée}
    run = run_max = 0                    # série courante / maximale de demi-coups sans capture
    n_echecs = n_cap_echec = 0
    n_promo = n_sous_promo = 0
    n_roque = n_coups_roi = dist_roi_max = 0
    roque = {}                           # {couleur: 1 petit roque / 2 grand roque}
    roi_ref = {1: chess.E1, 0: chess.E8}  # case de référence du roi (réinitialisée après roque)

    for tok in moves.split():
        if "." in tok:                                 # "12." / "12..." / "12.e4" -> on garde le coup
            tok = tok[tok.rfind(".") + 1:]
        if not tok or tok[0] in "0123456789*$":        # numéro, résultat (1-0, 1/2-1/2, *), NAG
            continue
        if tok[-1] in "?!":
            tok = tok.rstrip("?!")

        move = parse(tok)
        ply += 1
        dernier = tok
        blanc = ply & 1                                # 1 si le camp qui joue est blanc
        cap = "x" in tok                               # capture (prise en passant incluse)
        d = 0                                          # variation du bilan (point de vue des blancs)

        # --- capture : valeur de la pièce prise, lue AVANT de jouer le coup
        if cap:
            vt = type_en(move.to_square)               # None = prise en passant -> un pion
            v = VALEUR[vt] if vt else 1
            capt[ply] = v
            d = v
            if run > run_max:
                run_max = run
            run = 0
        else:
            run += 1

        # --- promotion
        promo = move.promotion
        if promo:
            n_promo += 1
            if promo != chess.QUEEN:
                n_sous_promo += 1
            d += VALEUR[promo] - 1                     # le pion devient une pièce

        # --- bilan matériel
        if d:
            if not blanc:
                d = -d
            bal += d
            if bal > bal_max:
                bal_max = bal
            elif bal < bal_min:
                bal_min = bal
            s = 1 if bal >= SEUIL_SIGNE else (-1 if bal <= -SEUIL_SIGNE else 0)
            if s and s != cote:
                if cote:
                    n_chg_signe += 1                   # l'avantage change de camp
                cote = s
            defi = sgn * bal <= SEUIL_DEFICIT
        n_def += defi
        bals_add(bal)

        # --- échecs (suffixe SAN)
        if tok[-1] in "+#":
            n_echecs += 1
            if cap:
                n_cap_echec += 1

        # --- roi : roque ou coup de roi
        c0 = tok[0]
        if c0 == "O":
            n_roque += 1
            roque[blanc] = 2 if tok.startswith("O-O-O") else 1
            roi_ref[blanc] = move.to_square            # le roque n'est pas une « marche » du roi
        elif c0 == "K":
            n_coups_roi += 1
            sq, r = move.to_square, roi_ref[blanc]
            dd = max(abs((sq & 7) - (r & 7)), abs((sq >> 3) - (r >> 3)))
            if dd > dist_roi_max:
                dist_roi_max = dd

        push(move)

    if run > run_max:
        run_max = run

    # ------------------------------------------------------------------------
    # Analyse des captures (à partir du dictionnaire capt)
    # ------------------------------------------------------------------------
    n_sac_dame = n_sac_pieces = n_ech = 0
    dames_sac = []                                     # demi-coups où une dame est perdue sans contrepartie
    for p, v in capt.items():
        if v >= 3:                                     # dame, tour ou pièce mineure perdue
            # compensation immédiate : le demi-coup voisin capture une pièce de valeur équivalente
            if abs(capt.get(p - 1, -9) - v) > TOL_EQUIV and abs(capt.get(p + 1, -9) - v) > TOL_EQUIV:
                if v == 9:
                    n_sac_dame += 1
                    dames_sac.append(p)
                else:
                    n_sac_pieces += 1

    seq_len = seq_net = 0
    prev, par0 = -9, 0
    for p, v in capt.items():
        if p - prev <= ECART_ECHANGE:                  # même séquence de captures rapprochées
            seq_len += 1
            seq_net += v if (p & 1) == par0 else -v    # + si capture du camp qui a ouvert la séquence
        else:
            if seq_len >= 2 and abs(seq_net) >= SEUIL_ECHANGE:
                n_ech += 1
            seq_len, seq_net, par0 = 1, v, p & 1
        prev = p
    if seq_len >= 2 and abs(seq_net) >= SEUIL_ECHANGE:
        n_ech += 1

    # Pions passés dans la position finale (les deux camps)
    pb, pn = board.pieces_mask(chess.PAWN, True), board.pieces_mask(chess.PAWN, False)
    n_passes = (sum(1 for s in chess.scan_forward(pb) if not PASSE_B[s] & pn)
                + sum(1 for s in chess.scan_forward(pn) if not PASSE_N[s] & pb))

    # piece_mateuse : pièce qui joue le coup du mat, lue dans le SAN du dernier coup (suffixe '#').
    # 'Q','R','B','N','K' = lettre du coup ; pion = 'P' (ou la pièce de promotion : "e8=Q#" -> 'Q') ;
    # roque = 'R' (c'est la tour qui donne échec). '' = la partie ne finit pas par un mat.
    # NB : en cas d'échec à la découverte, c'est la pièce qui bouge, pas celle qui donne échec.
    if dernier[-1:] == "#":
        c = dernier[0]
        piece_mat = (c if c in "NBRQK" else "R" if c == "O"
                     else dernier[dernier.index("=") + 1] if "=" in dernier else "P")
    else:
        piece_mat = ""

    # Features liées au vainqueur
    if sgn:
        min_w = bal_min if sgn == 1 else -bal_max      # plus mauvais bilan du vainqueur (<= 0)
        remontada = int(min_w <= SEUIL_REMONTADA and sgn * bal >= 0)
        n_def_v = n_def
    else:
        min_w = remontada = n_def_v = NAN

    # Features d'évaluation (NaN si pas d'évaluation exploitable)
    ev = features_eval(evals, sgn, ply, bals, capt, dames_sac, bool(piece_mat))

    return (
        # ===== A. Complexité =====
        # n_plies : nombre total de demi-coups joués
        ply,
        # ===== B. Combats et fluctuations matérielles =====
        # n_captures : nombre total de captures, prises en passant comprises
        len(capt),
        # n_changements_signe : changements de camp ayant l'avantage matériel, écarts < SEUIL_SIGNE ignorés
        n_chg_signe,
        # amplitude_bal : max - min du bilan matériel relatif (position initiale incluse)
        bal_max - bal_min,
        # min_bal_gagnant : plus mauvais bilan du vainqueur après un coup (<= 0, position initiale incluse)
        min_w,
        # n_plies_deficit : demi-coups où le bilan du vainqueur est <= SEUIL_DEFICIT
        n_def_v,
        # remontada : le vainqueur est passé à un bilan <= SEUIL_REMONTADA puis finit à bilan >= 0
        remontada,
        # ===== C. Sacrifices et déséquilibres =====
        # n_sacrifices_dame : dames perdues sans capture de dame adverse au demi-coup voisin
        n_sac_dame,
        # n_sacrifices_pieces : pièces mineures / tours perdues sans capture de valeur équivalente (±TOL_EQUIV)
        n_sac_pieces,
        # n_echanges_inegaux : séquences de >= 2 captures rapprochées dont le solde matériel est >= SEUIL_ECHANGE
        n_ech,
        # ===== D. Activité tactique =====
        # n_echecs : échecs donnés par les deux camps (suffixe '+' ou '#')
        n_echecs,
        # n_captures_avec_echec : captures qui donnent échec simultanément
        n_cap_echec,
        # n_promotions : promotions de toute nature
        n_promo,
        # n_sous_promotions : promotions en autre chose qu'une dame
        n_sous_promo,
        # ===== E. Dynamisme =====
        # n_coups_sans_capture_max : plus longue série de demi-coups sans capture
        run_max,
        # n_coups_roque : roques réalisés (0, 1 ou 2)
        n_roque,
        # roques_opposes : 1 si les deux rois ont roqué, sur des ailes opposées
        int(len(roque) == 2 and roque[1] != roque[0]),
        # distance_roi_max : distance (en cases, au sens du roi) à la case de départ ; roque non compté
        dist_roi_max,
        # n_coups_roi : demi-coups joués par un roi (roques exclus, déjà dans n_coups_roque)
        n_coups_roi,
        # n_pions_passes_fin : pions passés (les deux camps) dans la position finale
        n_passes,
        # piece_mateuse : pièce qui donne le mat ('' si pas de mat) -- voir plus haut
        piece_mat,
    ) + ev


def traiter_bloc(bloc):
    """bloc : liste de (moves, result, evals). Renvoie (liste de tuples, nombre d'erreurs).
    Fonction de niveau module : utilisable par multiprocessing."""
    sortie, erreurs = [], 0
    for moves, result, evals in bloc:
        try:
            sortie.append(features_partie(moves, result, evals))
        except Exception:                              # coup illisible / illégal
            sortie.append(ECHEC)
            erreurs += 1
    return sortie, erreurs