import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr, pearsonr
import os


RESULTS_DIR = "../data/models/c_ohadii"
CONDITIONS = ["50", "100", "300", "600", "1000", "1500", "2000"]
MEASURED_FILE = "../data/models/c_ohadii/FastLabelling_Levels.csv"

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12



read = pd.read_excel if str(MEASURED_FILE).lower().endswith((".xlsx", ".xls")) else pd.read_csv
raw = read(MEASURED_FILE, skiprows=1)
raw.columns = ["repeat", "light", "time"] + list(raw.columns[3:])
mets = list(raw.columns[3:])
raw[mets] = raw[mets].apply(pd.to_numeric, errors="coerce") * 1e-3

per_rep = raw.groupby(["light", "repeat"])[mets].mean()
meas_mean = per_rep.groupby("light").mean()
meas_std = per_rep.groupby("light").std()
meas_mean.index = meas_mean.index.astype(float)
meas_std.index = meas_std.index.astype(float)

# colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors = ['#FECD66', '#FF9138', '#EF314A', '#000000', '#BEE22A', '#4CC402', '#338032']
fig, ax = plt.subplots(figsize=(6.5, 6))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.001, 1000)
ax.set_ylim(0.001, 10)

print(f"{'cond':>6} {'under':>6} {'over':>6} {'equal':>6} {'r_log':>8} {'n':>4}")

for k, c in enumerate(CONDITIONS):
    est = pd.read_csv(f"{RESULTS_DIR}/estimation_results_{c}/estimated_pool_sizes_ci.csv",
                      index_col=0)
    meas = pd.DataFrame({"meas": meas_mean.loc[float(c)],
                         "meas_std": meas_std.loc[float(c)]})
    df = est.join(meas, how="inner")

    lo_band = df["meas"] - df["meas_std"]
    hi_band = df["meas"] + df["meas_std"]
    over  = int((df["LB"] > hi_band).sum())
    under = int((df["UB"] < lo_band).sum())
    equal = int(((df["LB"] <= hi_band) & (df["UB"] >= lo_band)).sum())


    plot_this = c in ("1500", "2000")
    xe, lb, ub = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    ym, ystd = df["meas"], df["meas_std"]
    lo, hi = np.minimum(lb, ub), np.maximum(lb, ub)
    xerr = np.vstack([np.minimum(np.clip(xe - lo, 0, None), xe.to_numpy() * 0.999),
                      np.clip(hi - xe, 0, None)])
    yerr = np.vstack([np.minimum(np.clip(ystd, 0, None), ym.to_numpy() * 0.999),
                      np.clip(ystd, 0, None)])
    if plot_this:
        ax.errorbar(xe, ym, xerr=xerr, yerr=yerr,
                    fmt="o", ms=2, capsize=1.5, elinewidth=0.6,
                    color=colors[k], label=f"{c} µE", alpha=0.6)

    xv, yv = xe.to_numpy(), ym.to_numpy()
    mask = (xv > 0) & (yv > 0) & np.isfinite(xv) & np.isfinite(yv)
    r = np.nan
    if mask.sum() > 1:
        fit = linregress(np.log10(xv[mask]), np.log10(yv[mask]))
        r = fit.rvalue
        if plot_this:
            xs = np.logspace(np.log10(ax.get_xlim()[0]), np.log10(ax.get_xlim()[1]), 100)
            ax.plot(xs, 10 ** fit.intercept * xs ** fit.slope, "-",
                    color=colors[k], lw=1.2, zorder=5)

    print(f"{c:>6} {under:>6} {over:>6} {equal:>6} {r:>8.3f} {int(mask.sum()):>4}")

ax.set_xlabel("Estimated pool size (µmol g$^{-1}$ DW)")
ax.set_ylabel("Measured pool size (µmol g$^{-1}$ DW)")
ax.legend(frameon=True, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "met_pools.png"), dpi=300, bbox_inches="tight")