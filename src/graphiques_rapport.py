import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

RACINE = Path(__file__).resolve().parent.parent
DOSSIER = Path(os.environ.get("LICHESS_DATA", str(RACINE / "donnees")))
OUT = DOSSIER / "resultats" / "selection"

# ------------------------------------------------------------------
# GRAPHIQUE 3 : Comparaison de 5 grandes ouvertures (justifie le tri par ECO)
# ------------------------------------------------------------------
df_feat = pq.read_table(
    DOSSIER / "resultats" / "features.parquet",
    columns=["ECO", "n_captures", "n_echecs"]
).to_pandas()

top5_eco = df_feat["ECO"].value_counts().head(5).index.tolist()
sub = df_feat[df_feat["ECO"].isin(top5_eco)]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sub.boxplot(column="n_captures", by="ECO", ax=axes[0], showfliers=False)
axes[0].set_title("Nombre de captures par ouverture (Top 5 ECO)")
axes[0].set_xlabel("Code ECO")
axes[0].set_ylabel("n_captures")

sub.boxplot(column="n_echecs", by="ECO", ax=axes[1], showfliers=False)
axes[1].set_title("Nombre d'échecs par ouverture (Top 5 ECO)")
axes[1].set_xlabel("Code ECO")
axes[1].set_ylabel("n_echecs")

plt.suptitle("")
plt.tight_layout()
plt.savefig(OUT / "graph3_comparaison_eco.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# GRAPHIQUE 4 : Queue de distribution de d² et ajustement EVT / POT
# ------------------------------------------------------------------
scores = pq.read_table(OUT / "scores.parquet", columns=["ECO", "d2", "rarete"]).to_pandas()
resume = pd.read_csv(OUT / "resume_par_groupe.csv")

top_row = resume[resume["statut"] == "ok"].sort_values("n", ascending=False).iloc[0]
eco_cible = str(top_row["groupe"])
u, xi, sigma = top_row["seuil_u"], top_row["xi"], top_row["sigma"]

if eco_cible == "GLOBAL":
    masque = pd.Series(True, index=scores.index)
elif eco_cible.endswith("*"):
    masque = scores["ECO"].astype(str).str.startswith(eco_cible[0])
else:
    masque = scores["ECO"] == eco_cible

d2_eco = scores.loc[masque, "d2"].dropna().to_numpy()
exc = np.sort(d2_eco[d2_eco > u] - u)
n_tot = len(d2_eco)
ne = len(exc)

p_emp = (ne - np.arange(ne)) / n_tot
if abs(xi) < 1e-8:
    p_gpd = (ne / n_tot) * np.exp(-exc / sigma)
else:
    p_gpd = (ne / n_tot) * np.maximum(1 + xi * exc / sigma, 1e-12) ** (-1.0 / xi)

plt.figure(figsize=(8, 5))
plt.semilogy(exc + u, p_emp, ".", markersize=4, alpha=0.6, label="Queue empirique (données)")
plt.semilogy(exc + u, p_gpd, "r-", linewidth=2, label=f"Loi de Pareto (GPD, $\\xi$={xi:.2f})")
plt.axhline(1e-3, color="green", linestyle="--", label="Seuil de sélection ($p = 10^{-3}$)")
plt.axvline(u, color="gray", linestyle=":", label=f"Seuil POT ($u$ = {u:.1f})")
plt.title(f"Ajustement EVT/POT sur la distance de Mahalanobis ($d^2$) — Groupe {eco_cible}")
plt.xlabel("Distance de Mahalanobis au carré ($d^2$)")
plt.ylabel("Probabilité de dépassement (échelle log)")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "graph4_evt_pot.png", dpi=150)
plt.close()

print("Graphiques 3 et 4 générés avec succès !")
