import pandas as pd
from scipy import stats

RESULTS_DIR = "../data/models/c_ohadii"
CONDITIONS = ["50", "100", "300", "600", "1000", "1500", "2000"]

data = {c: pd.read_csv(f"{RESULTS_DIR}/estimation_results_{c}/estimated_comp_alphas.csv",
                       index_col=0)
        for c in CONDITIONS}

# collect alpha values (as %) per compartment across all conditions
compartments = {"m": "Mitochondrion", "c": "Cytosol", "p": "Chloroplast"}
values = {label: [data[c].loc[k, "alpha"] * 100 for c in CONDITIONS]
          for k, label in compartments.items()}

# experimental reference: (mean %, sd %, n) from Chlorella fusca, Atkinson et al. 1974
reference = {
    "Mitochondrion": (2.688, 0.398, 5),
    "Cytosol":        (60.082, 3.887, 5),
    "Chloroplast":    (37.230, 4.260, 5),
}

for label, x in values.items():
    x = pd.Series(x)
    m1, s1, n1 = x.mean(), x.std(ddof=1), len(x)
    m2, s2, n2 = reference[label]

    se = ((s1**2 / n1) + (s2**2 / n2)) ** 0.5
    t = (m1 - m2) / se
    df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    p = 2 * stats.t.sf(abs(t), df)

    print(f"{label}: computed = {m1:.2f} ± {s1:.2f}% (n={n1}) | "
          f"experimental = {m2:.2f} ± {s2:.2f}% (n={n2})")
    print(f"   Welch's t = {t:.3f}, df = {df:.2f}, p = {p:.4f}\n")