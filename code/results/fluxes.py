import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import variation
import os

from scipy.stats import pearsonr

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

RESULTS_DIR = "../data/models/c_ohadii"
CONDITIONS = ["50", "100", "300", "600", "1000", "1500", "2000"]
MINFLUX = 2e-1
MAXFLUX = 20
TOP_N = 8



data = {c: pd.read_csv(f"{RESULTS_DIR}/estimation_results_{c}/estimated_fluxes_ci.csv",
                       index_col=0)
        for c in CONDITIONS}

for c in CONDITIONS:
    data[c] = data[c].clip(lower=0)

flux = pd.DataFrame({c: d.iloc[:, 0] for c, d in data.items()}).dropna()

remove_list= ['G-3PGA.c', 'G-3PGA.p', 'G-ACA.m', 'G-ACA.p', 'G-G1P.c', 'G-G1P.p', 'G-G6P.c', 'G-G6P.p', 'G-GLY.c', 'G-GLY.p', 'G-PEP.c', 'G-PEP.p', 'G-PYR.c', 'G-PYR.p', 'G-SER.c', 'G-SER.P', 'G-TP.c', 'G-TP.p', 'G-F6P.c', 'G-F6P.p', 'G-P5P.c', 'G-P5P.p' ]  # reactions you don't want

flux = flux.drop(index=remove_list, errors="ignore")

mask = ((flux > MINFLUX) & (flux < MAXFLUX)).all(axis=1)
kept = flux[mask]


cv = variation(kept, axis=1)
top = kept.index[np.argsort(cv)[::-1][:TOP_N]].tolist()



records = []
for rxn in top:
    vals = flux.loc[rxn]
    fc = vals.max() / vals.min()
    records.append((rxn, fc, vals.idxmax(), vals.idxmin()))

records.sort(key=lambda t: t[1], reverse=True)

for rxn, fc, hi_c, lo_c in records[:2]:
    print(f"{rxn}: fold-change={fc:.1f}, max at {hi_c} µE, min at {lo_c} µE")

# colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors = ['#FECD66', '#FF9138', '#EF314A', '#000000', '#BEE22A', '#4CC402', '#338032']
dodge = np.linspace(-0.3, 0.3, len(CONDITIONS))
x = np.arange(len(top))

fig, ax = plt.subplots(figsize=(12, 6))
for k, c in enumerate(CONDITIONS):
    val = data[c].loc[top].iloc[:, 0]
    lb  = data[c].loc[top].iloc[:, 1]
    ub  = data[c].loc[top].iloc[:, 2]
    lo, hi = np.minimum(lb, ub), np.maximum(lb, ub)
    yerr = np.clip([val - lo, hi - val], 0, None)
    ax.errorbar(x + dodge[k], val, yerr=yerr,
                fmt="o", ms=4, capsize=2, color=colors[k], label=f"{c} µE")

ax.set_yscale("log")
ax.set_ylim(bottom=MINFLUX)
ax.set_xticks(x)
ax.set_xticklabels(top, rotation=90)
ax.set_ylabel("Flux (µmol g$^{-1}$ DW s$^{-1}$)")
ax.set_xlabel("Reaction")
ax.legend(title="Condition", frameon=True, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
# ax.text(0.0, 1.02, "A.", transform=ax.transAxes, fontsize=14, fontweight="bold")
fig.savefig(os.path.join(RESULTS_DIR, "fluxes.png"), dpi=300, bbox_inches="tight")


from scipy.stats import pearsonr, spearmanr

STARCH = [17.13329496, 49.4810233, 87.82824639, 75.85133228, 102.4811334, 108.4646493, 116.7136155]
GROWTH = [0.00457426967, 0.01959489523, 0.02746937508, 0.03654856763, 0.04464759916, 0.04934585466, 0.05124255122]
PEP = [20.53061246,48.0925921,94.35885282,95.67774888,113.0963118,139.4592767,167.4471461]


starch = pd.Series(STARCH, index=CONDITIONS)
growth = pd.Series(GROWTH, index=CONDITIONS)
pep = pd.Series(PEP, index=CONDITIONS)

# kept columns are already in CONDITIONS order; ensure alignment
kept_aligned = kept[CONDITIONS]

def correlate_against(target):
    rows = []
    for rxn in kept_aligned.index:
        v = kept_aligned.loc[rxn]
        if v.std() == 0:  # skip constant series (correlation undefined)
            continue
        r, p = pearsonr(v.values, target.values)
        rows.append((rxn, r, p))
    return pd.DataFrame(rows, columns=["reaction", "r", "p"]).set_index("reaction")

growth_corr = correlate_against(growth)

# Growth: report reaction with highest/lowest correlation
hi = growth_corr.loc[growth_corr["r"].idxmax()]
lo = growth_corr.loc[growth_corr["r"].idxmin()]
print("\n=== GROWTH ===")
print(f"Highest correlation: {growth_corr['r'].idxmax()}: r={hi['r']:.3f}, p={hi['p']:.3g}")
print(f"Lowest correlation:  {growth_corr['r'].idxmin()}: r={lo['r']:.3f}, p={lo['p']:.3g}")


sp = flux.loc["Starch_produce", CONDITIONS]
r, p= spearmanr(sp.values, starch.values)
# r, p= pearsonr(sp.values, starch.values)
print("\n=== STARCH (Starch_produce) ===")
print(f"r={r:.3f}, p={p:.3g}")

pep_est_flux = flux.loc["ENO.c", CONDITIONS] + flux.loc["Ppdk.p", CONDITIONS]
rho, p = spearmanr(pep_est_flux.values, pep.values)
print("\n=== ENO.c + Ppdk.p ===")
print(f"Spearman rho={rho:.3f}, p={p:.3g}")