"""
Benjamini-Hochberg multiple hypothesis testing correction.

For each source (Human, GPT Explicit, GPT NLP, etc.), compares every
single-factor demographic subgroup against the global Human baseline using
Welch's t-test. Applies BH FDR correction across all tests within each
source, then reports what survives at q < 0.05.

Usage:
    python scripts/analysis/bh_significance.py

Output:
    output/dashboard/bh_results_all.csv     - all tests with raw + adjusted p
    output/dashboard/bh_results_significant.csv - only FDR-significant findings
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).parent.parent.parent

TRAITS = ['E', 'A', 'C', 'N', 'O']
TRAIT_NAMES = {
    'E': 'Extroversion',
    'A': 'Agreeableness',
    'C': 'Conscientiousness',
    'N': 'Emotional Stability',
    'O': 'Openness'
}
DEMO_COLS = ['Country', 'Race', 'Age', 'Age_Group', 'Gender']
MIN_N = 30


def welch_t_from_stats(m1, s1, n1, m2, s2, n2):
    """Welch's t-test and Cohen's d from summary statistics."""
    _, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    pooled_var = (((n1 - 1) * s1**2) + ((n2 - 1) * s2**2)) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 0
    d = (m1 - m2) / pooled_std if pooled_std > 0 else 0.0
    return p, d


def get_single_factor_label(row):
    """Returns a human-readable label for a single-factor subgroup row."""
    non_all = [(col, row[col]) for col in DEMO_COLS if str(row[col]) != 'All']
    if len(non_all) != 1:
        return None
    col, val = non_all[0]
    return f"{col}={val}"


def run_bh_analysis(df, source, human_baseline):
    """
    For a given source, compare each single-factor subgroup to the global
    Human baseline. Returns a DataFrame of all test results.
    """
    source_df = df[df['Source'] == source]
    results = []

    for _, row in source_df.iterrows():
        label = get_single_factor_label(row)
        if label is None:
            continue

        for t in TRAITS:
            m_sub = row[f"{t}_Trait_mean"]
            s_sub = row[f"{t}_Trait_std"]
            n_sub = row[f"{t}_Trait_count"]

            if pd.isna(m_sub) or pd.isna(s_sub) or n_sub < MIN_N:
                continue

            m_base = human_baseline[f"{t}_Trait_mean"]
            s_base = human_baseline[f"{t}_Trait_std"]
            n_base = human_baseline[f"{t}_Trait_count"]

            p_raw, d = welch_t_from_stats(m_sub, s_sub, n_sub, m_base, s_base, n_base)

            results.append({
                'Source': source,
                'Subgroup': label,
                'Trait': TRAIT_NAMES[t],
                'Trait_Code': t,
                'Subgroup_Mean': round(m_sub, 3),
                'Baseline_Mean': round(m_base, 3),
                'Difference': round(m_sub - m_base, 3),
                'Cohens_d': round(d, 3),
                'N_Subgroup': int(n_sub),
                'P_Raw': p_raw,
            })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    pvals = results_df['P_Raw'].values

    reject_bh,  p_adj_bh,  _, _ = multipletests(pvals, method='fdr_bh',    alpha=0.05)
    reject_bon, p_adj_bon, _, _ = multipletests(pvals, method='bonferroni', alpha=0.05)

    results_df['P_Raw']              = results_df['P_Raw'].round(4)
    results_df['P_Adjusted_BH']      = p_adj_bh.round(4)
    results_df['P_Adjusted_Bonferroni'] = p_adj_bon.round(4)
    results_df['Significant_BH']         = reject_bh
    results_df['Significant_Bonferroni'] = reject_bon

    return results_df


# ==========================================
# MAIN
# ==========================================
print("Loading data...")
df = pd.read_csv(ROOT / "output/dashboard/dashboard_precalc_stats_singular.csv", low_memory=False)

# Global Human baseline: all demo cols = 'All', source = 'Human'
human_baseline = df[
    (df['Source'] == 'Human') &
    (df[DEMO_COLS] == 'All').all(axis=1)
].iloc[0]

print(f"Human global baseline (n={int(human_baseline['E_Trait_count'])}):")
for t in TRAITS:
    print(f"  {TRAIT_NAMES[t]}: {human_baseline[f'{t}_Trait_mean']:.3f} ± {human_baseline[f'{t}_Trait_std']:.3f}")

sources = [s for s in df['Source'].unique()]
print(f"\nRunning BH analysis for {len(sources)} sources: {sources}")

all_results = []
for source in sources:
    res = run_bh_analysis(df, source, human_baseline)
    if not res.empty:
        all_results.append(res)
        n_tests = len(res)
        n_sig = res['Significant_BH'].sum()
        print(f"  {source}: {n_tests} tests → {n_sig} survive BH correction")

combined = pd.concat(all_results, ignore_index=True)
combined = combined.sort_values('Cohens_d', key=abs, ascending=False)

sig_bh  = combined[combined['Significant_BH']].copy()
sig_bon = combined[combined['Significant_Bonferroni']].copy()

out_all = ROOT / "output/dashboard/bh_results_all.csv"
out_bh  = ROOT / "output/dashboard/bh_results_significant.csv"
out_bon = ROOT / "output/dashboard/bonferroni_results_significant.csv"
combined.to_csv(out_all, index=False)
sig_bh.to_csv(out_bh, index=False)
sig_bon.to_csv(out_bon, index=False)

print(f"\n{'='*60}")
print(f"TOTAL TESTS:                {len(combined)}")
print(f"SURVIVING BH (q<0.05):      {len(sig_bh)}")
print(f"SURVIVING Bonferroni (p<0.05): {len(sig_bon)}")
print(f"{'='*60}")

print("\nPer-source survival:")
for src in combined['Source'].unique():
    sub = combined[combined['Source'] == src]
    n   = len(sub)
    bh  = sub['Significant_BH'].sum()
    bon = sub['Significant_Bonferroni'].sum()
    print(f"  {src:<30} BH: {bh}/{n} ({100*bh/n:.1f}%)   Bonferroni: {bon}/{n} ({100*bon/n:.1f}%)")

print(f"\nFull results:        {out_all}")
print(f"BH significant:      {out_bh}")
print(f"Bonferroni significant: {out_bon}")
