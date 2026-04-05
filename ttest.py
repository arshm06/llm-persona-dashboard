import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

FILE_PATH = "simulated_ordering_experiment.csv"

# ================================
# LOAD DATA
# ================================
df = pd.read_csv(FILE_PATH)

# Clean column names just in case
df.columns = df.columns.str.strip()

# ================================
# BUILD BASE PERSONA ID
# ================================
# Example:
# P0_R0_Age_Gender_Nationality -> P0_R0
df["Base_ID"] = df["Sim_ID"].astype(str).str.extract(r"^(P\d+_R\d+)")

# Safety check
if df["Base_ID"].isna().any():
    print("Warning: Some Base_ID values could not be extracted.")
    print(df.loc[df["Base_ID"].isna(), ["Sim_ID"]].head())

# ================================
# COMPUTE EXTRAVERSION MEAN
# ================================
e_cols = [f"E{i}_score" for i in range(1, 11)]

missing = [c for c in e_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

df["E_mean"] = df[e_cols].mean(axis=1)

# ================================
# PIVOT BY BASE PERSONA + ORDERING
# ================================
df_pivot = df.pivot_table(
    index="Base_ID",
    columns="Ordering",
    values="E_mean",
    aggfunc="mean"
)

print("\nPivoted table:")
print(df_pivot)
print()

# ================================
# PAIRED T-TEST FUNCTION
# ================================
def paired_ttest(a, b):
    paired = pd.concat([a, b], axis=1).dropna()
    x = paired.iloc[:, 0]
    y = paired.iloc[:, 1]

    n = len(paired)

    if n < 2:
        return np.nan, np.nan, np.nan, n

    t_stat, p_val = stats.ttest_rel(x, y)

    diff = x - y
    diff_std = diff.std(ddof=1)

    if diff_std == 0 or np.isnan(diff_std):
        d = 0.0
    else:
        d = diff.mean() / diff_std

    return t_stat, p_val, d, n

# ================================
# RUN ALL PAIRWISE COMPARISONS
# ================================
orderings = df_pivot.columns.tolist()

print("==============================")
print("EXTRAVERSION: ORDERING EFFECTS")
print("==============================\n")

for o1, o2 in combinations(orderings, 2):
    t_stat, p_val, d, n = paired_ttest(df_pivot[o1], df_pivot[o2])

    print(f"Comparison: {o1}  vs  {o2}")
    print(f"Sample size (paired): {n}")

    if n < 2:
        print("T-statistic: NA")
        print("P-value: NA")
        print("Result: Not enough paired samples")
        print("Cohen's d: NA")
        print("-" * 50)
        continue

    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.6f}")

    sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"

    abs_d = abs(d)
    if abs_d < 0.2:
        effect = "Negligible"
    elif abs_d < 0.5:
        effect = "Small"
    elif abs_d < 0.8:
        effect = "Medium"
    else:
        effect = "Large"

    print(f"Result: {sig}")
    print(f"Cohen's d: {d:.4f} ({effect})")
    print("-" * 50)

print("\nDone.\n")