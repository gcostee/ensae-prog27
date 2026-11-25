# ensae-prog27

Recherche de parties d'échecs remarquables (du type « partie du mois » de Lichess) dans un dump mensuel de la base
Lichess (~100 millions de parties) : filtrage, calcul d'une soixantaine de features par partie, sélection statistique
des parties atypiques, puis PU learning appris sur les parties du mois pour ne garder que 20 parties.

```
dump Lichess (.pgn.zst, ~100 M parties)
   │  1. filtrer_parties.py      parties exploitables (Elo ≥ 1500, 20-120 coups, décisives, évaluées)
   │  1. analyse_features.py     ~60 features par partie + statistiques et graphiques
   ▼
features.parquet (quelques millions de parties)
   │  2. selection_pot.py        Mahalanobis robuste + POT, par code ECO (≥ 5 000 parties) -> base unique
   ▼
base_selection.parquet (quelques milliers de candidates)
   │  3a. scraping.py            parties du mois du blog Lichess        (positifs P)
   │  3b. temoin.py              5 000 parties tirées au hasard          (non étiquetées U)
   │  3c. augmentation.py        Stockfish à profondeur fixe + features, pour P, U et les candidates
   │  3d. pu_learning.py         classifieur P contre U (Elkan-Noto) -> candidates étiquetées positives
   │  3e. echantillon_final.py   20 parties (grandes distances + plus communes)
   ▼
echantillon_final.csv (20 parties)          comparaison.py : 200 parties quelconques (10 blocs de 20)
```

Tout se pilote depuis le notebook **`main.ipynb`**, qui lance les scripts dans l'ordre, affiche tableaux et graphiques
et laisse de la place pour les commentaires. Chaque script peut aussi être lancé seul en ligne de commande.

## Structure du dépôt

```
ensae-prog27/
├── main.ipynb               notebook principal (toutes les étapes, tableaux, graphiques, commentaires)
├── README.md
├── requirements.txt
├── .gitignore               exclut les données, les parquets, les résultats, .venv
├── src/
│   ├── features_core.py     calcul des features d'une partie (module, ne se lance pas)
│   ├── outils.py            chemins, lecture rapide des dumps, extraction de lignes (module)
│   ├── filtrer_parties.py   étape 1 : filtrage du dump
│   ├── analyse_features.py  étape 1 : features, statistiques, graphiques
│   ├── selection_pot.py     étape 2 : sélection Mahalanobis + POT
│   ├── scraping.py          étape 3a : parties du mois
│   ├── temoin.py            étape 3b : groupe témoin
│   ├── augmentation.py      étape 3c : Stockfish + features
│   ├── pu_learning.py       étape 3d : PU learning
│   ├── echantillon_final.py étape 3e : 20 parties
│   └── comparaison.py       200 parties tirées au hasard
├── exemples/
│   ├── mini_dump.pgn.zst    700 parties SYNTHÉTIQUES (coups au hasard) pour tester l'installation
│   └── gotm_exemple.pgn     40 « parties du mois » SYNTHÉTIQUES (test sans réseau)
├── tests/                   tests hors ligne (pytest)
└── donnees/                 vide dans le dépôt : vos données (ou ailleurs, voir plus bas)
```

## Installation

Python 3.10 ou plus récent.

```powershell
# Windows (PowerShell), à la racine du dépôt
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Stockfish** (étape 3c) ne s'installe pas avec pip : télécharger l'exécutable sur <https://stockfishchess.org/download/>
(Windows : version « x86-64-avx2 » pour un processeur récent), le décompresser hors du dépôt, puis indiquer son chemin :
variable d'environnement `STOCKFISH_PATH`, variable `STOCKFISH` du notebook, ou option `--stockfish`.

## Données

Les données ne sont jamais versionnées. Par défaut elles sont lues et écrites dans `donnees/` à la racine du dépôt ;
pour un autre emplacement (conseillé : un disque hors OneDrive), définir la variable `LICHESS_DATA`, ou régler
`DONNEES` dans le notebook, ou passer `--dossier` aux scripts.

1. Télécharger le dump mensuel sur <https://database.lichess.org> (`lichess_db_standard_rated_2026-01.pgn.zst`).
2. Le placer dans le dossier des données, sans le décompresser.

Fichiers produits dans le dossier des données :

| Étape | Fichiers |
|---|---|
| 1 | `parties_filtrees.parquet`, `comptes_par_tranche.csv` |
| 1 | `resultats/features.parquet`, `resume_statistiques.csv`, `distributions_*.png`, `matrice_correlation.png`, `nuages_de_points_*.png` |
| 2 | `resultats/selection/base_selection.parquet` (+ `.csv`), `resume_par_groupe.csv` |
| 3a | `pu/gotm.pgn`, `pu/gotm_index.csv` |
| 3b | `pu/temoin.parquet` |
| 3c | `pu/features_augmentees.parquet`, `pu/cache_stockfish_d12/` |
| 3d | `pu/pu_candidats.parquet` (+ `.csv`), `pu/pu_resume.json`, `pu/pu_scores.png`, `pu/pu_coefficients.csv` |
| 3e | `pu/echantillon_final.csv` |
| Comparaison | `comparaison/comparaison_200.csv` |

## Exécution

### Avec le notebook (recommandé)

1. Ouvrir le dossier du dépôt dans VS Code, ouvrir `main.ipynb`, choisir le noyau du `.venv` (en haut à droite).
2. **Test de l'installation** : dans la cellule de configuration, `MODE = "exemple"`, renseigner `STOCKFISH`, puis
   *Run All* (quelques minutes). Les résultats vont dans `exemples/travail/` (non versionné).
3. **Vos données** : `MODE = "complet"`, vérifier `DONNEES`, `DUMP` et `STOCKFISH`, puis exécuter les cellules
   dans l'ordre. Une étape déjà faite se saute avec `LANCER["…"] = False` (ses fichiers sont relus et affichés).

Chaque script tourne dans un processus séparé : la sortie s'affiche en direct dans la cellule, et le calcul
parallèle fonctionne aussi sous Windows.

### En ligne de commande

Depuis la racine du dépôt, dans l'ordre (le dossier des données vient de `LICHESS_DATA`, ou ajouter `--dossier …`) :

```bash
python src/filtrer_parties.py --dump lichess_db_standard_rated_2026-01.pgn.zst
python src/analyse_features.py
python src/selection_pot.py                       # --n-min 5000 --p-seuil 1e-3
python src/scraping.py                            # internet requis
python src/temoin.py                              # --n 5000
python src/augmentation.py --stockfish CHEMIN     # --profondeur 12
python src/pu_learning.py                         # --seuil 0.5
python src/echantillon_final.py                   # --n 20
python src/comparaison.py --dump lichess_db_standard_rated_2026-01.pgn.zst
```

`python src/<script>.py --help` liste les options de chaque script.

### Durées indicatives (dump d'un mois, PC portable 4 à 8 cœurs)

| Étape | Durée | Remarque |
|---|---|---|
| filtrer_parties | 1 à 2 h | lit tout le dump, une seule fois par mois |
| analyse_features | 10 min à 1 h | selon le nombre de parties retenues |
| selection_pot | quelques minutes | |
| scraping | quelques minutes | une requête par seconde (règles de l'API Lichess) |
| temoin | quelques secondes | (`--source dump` : un passage complet sur le dump) |
| augmentation | 1 à 3 h | Stockfish ; reprend là où il s'est arrêté |
| pu_learning, echantillon_final | moins d'une minute | |
| comparaison | 5 à 30 min | décompression de tout le dump |

## Reproductibilité

* Tirages aléatoires à graine fixe (`--graine`, 0 par défaut) : témoin, comparaison, validation croisée.
* Stockfish à profondeur fixe (et non à temps fixe) : mêmes évaluations d'une machine à l'autre pour une même
  version de Stockfish ; noter la version utilisée. Le cache `pu/cache_stockfish_d<profondeur>/` garde les évaluations.
* Les parties du mois évoluent chaque mois : conserver `pu/gotm.pgn` et `pu/gotm_index.csv` utilisés pour une analyse.
* Paramètres de chaque exécution : dans la cellule de configuration du notebook (et `pu/pu_resume.json`).
* Tests hors ligne : `python -m pytest tests`.

## Limites

* Toutes les parties retenues à l'étape 1 sont des parties analysées à la demande d'un joueur : population biaisée
  vers des parties que leurs joueurs trouvaient intéressantes.
* Peu de positifs (une partie par mois depuis 2023) : le PU learning reste indicatif ; regarder l'AUC hors
  échantillon et la part du témoin étiquetée positive dans `pu_resume.json`.
* Le scraping dépend de la mise en page du blog Lichess ; si elle change, fournir la liste des billets avec
  `python src/scraping.py --urls billets.txt`.
* Les fichiers d'`exemples/` sont synthétiques (coups joués au hasard) : ils vérifient le fonctionnement, leurs
  résultats n'ont aucun sens échiquéen.

