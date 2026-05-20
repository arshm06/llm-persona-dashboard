"""
Alignment metrics: how well does each model's demographic pattern correlate with humans?

For each factor x trait:
  - Builds vectors of per-subgroup means for Human and each model
  - Computes Pearson and Spearman r between the human vector and each model's vector
  - Repeats for demographic effect vectors (subgroup_mean - global_mean)

Outputs:
  output/dashboard/alignment_correlations.csv  - correlations per (factor, trait, source)
  output/figures/fig_alignment_heatmap.png     - Pearson r heatmap per factor

Usage:
    python scripts/analysis/alignment_metrics.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

TRAITS = ['E', 'A', 'C', 'N', 'O']
TRAIT_NAMES = {
    'E': 'Extroversion', 'A': 'Agreeableness', 'C': 'Conscientiousness',
    'N': 'Emotional Stability', 'O': 'Openness',
}
TRAIT_ORDER  = [TRAIT_NAMES[t] for t in TRAITS]
DEMO_COLS    = ['Country', 'Race', 'Age', 'Age_Group', 'Gender']
FACTORS      = ['Gender', 'Age_Group', 'Race', 'Country']
MIN_N        = 30
MIN_SUBGROUPS = 3   # minimum number of common subgroups needed for a valid correlation

HUMAN_SRC     = 'Human'
MODEL_SOURCES = ['AI_GPT4o_Explicit', 'AI_GPT4o_NLP', 'AI_Qwen_NLP']
SOURCE_LABELS = {
    'AI_GPT4o_Explicit': 'GPT Explicit',
    'AI_GPT4o_NLP':      'GPT NLP',
    'AI_Qwen_NLP':       'Qwen NLP',
}

FIG_DIR = ROOT / "output/figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def get_single_factor_rows(df, factor):
    mask = df[factor] != 'All'
    for other in DEMO_COLS:
        if other != factor:
            mask &= df[other] == 'All'
    return df[mask].copy()


def get_global_mean(df, source, trait):
    mask = (df['Source'] == source) & (df[DEMO_COLS] == 'All').all(axis=1)
    rows = df[mask]
    return float(rows.iloc[0][f"{trait}_Trait_mean"]) if not rows.empty else np.nan


# ── Load data ──────────────────────────────────────────────────────────────
print("Loading precalculated stats...")
df = pd.read_csv(ROOT / "output/dashboard/dashboard_precalc_stats_singular.csv", low_memory=False)
for col in DEMO_COLS:
    if col in df.columns:
        df[col] = df[col].astype(str)

available_models = [s for s in MODEL_SOURCES if s in df['Source'].unique()]
print(f"Model sources found: {available_models}")


# ── Compute correlations ────────────────────────────────────────────────────
records = []
for factor in FACTORS:
    factor_df = get_single_factor_rows(df, factor)

    for t in TRAITS:
        mean_col = f"{t}_Trait_mean"
        n_col    = f"{t}_Trait_count"

        # Build per-subgroup mean dict for each source (n >= MIN_N)
        src_vecs = {}
        for src in [HUMAN_SRC] + available_models:
            src_df = factor_df[
                (factor_df['Source'] == src) &
                (factor_df[n_col].astype(float) >= MIN_N)
            ]
            if src_df.empty:
                continue
            src_vecs[src] = src_df.set_index(factor)[mean_col].astype(float).to_dict()

        if HUMAN_SRC not in src_vecs:
            continue

        human_global = get_global_mean(df, HUMAN_SRC, t)

        for model_src in available_models:
            if model_src not in src_vecs:
                continue

            common = sorted(set(src_vecs[HUMAN_SRC]) & set(src_vecs[model_src]))
            if len(common) < MIN_SUBGROUPS:
                continue

            h_means = np.array([src_vecs[HUMAN_SRC][s]  for s in common])
            m_means = np.array([src_vecs[model_src][s]  for s in common])

            model_global = get_global_mean(df, model_src, t)
            h_effects    = h_means - human_global
            m_effects    = m_means - model_global

            # Pearson / Spearman on raw means
            r_means,   p_means   = stats.pearsonr(h_means,   m_means)
            rho_means, p_rho     = stats.spearmanr(h_means,  m_means)

            # Pearson on effect vectors (only meaningful if there's variance)
            if np.std(h_effects) > 1e-9 and np.std(m_effects) > 1e-9:
                r_eff, p_eff = stats.pearsonr(h_effects, m_effects)
            else:
                r_eff, p_eff = np.nan, np.nan

            records.append({
                'Factor':             factor,
                'Trait_Code':         t,
                'Trait':              TRAIT_NAMES[t],
                'Source':             model_src,
                'Source_Label':       SOURCE_LABELS[model_src],
                'N_Subgroups':        len(common),
                'Pearson_r_means':    round(r_means,   4),
                'Pearson_p_means':    round(p_means,   4),
                'Spearman_rho_means': round(rho_means, 4),
                'Spearman_p_means':   round(p_rho,     4),
                'Pearson_r_effects':  round(r_eff, 4) if not np.isnan(r_eff) else np.nan,
                'Pearson_p_effects':  round(p_eff, 4) if not np.isnan(p_eff) else np.nan,
            })

corr_df = pd.DataFrame(records)
out_csv = ROOT / "output/dashboard/alignment_correlations.csv"
corr_df.to_csv(out_csv, index=False)
print(f"Saved {out_csv} ({len(corr_df)} records)")

# Print summary
print("\nMean Pearson r (subgroup means vs human) by source:")
for src in available_models:
    sdf = corr_df[corr_df['Source'] == src]
    if sdf.empty:
        continue
    print(f"  {SOURCE_LABELS[src]}: r={sdf['Pearson_r_means'].mean():.3f} "
          f"(effect r={sdf['Pearson_r_effects'].mean():.3f}, "
          f"n_combos={len(sdf)})")

print("\nPearson r by factor:")
for factor in FACTORS:
    fdf = corr_df[corr_df['Factor'] == factor]
    for src in available_models:
        sdf = fdf[fdf['Source'] == src]
        if sdf.empty:
            continue
        print(f"  {factor:<12} {SOURCE_LABELS[src]:<14}: r={sdf['Pearson_r_means'].mean():.3f}")


# ── Figure: Pearson r heatmap (source x trait), faceted by factor ──────────
n_factors = len(FACTORS)
fig, axes = plt.subplots(1, n_factors, figsize=(4.5 * n_factors, 5), sharey=True)
if n_factors == 1:
    axes = [axes]

for ax, factor in zip(axes, FACTORS):
    fdf = corr_df[corr_df['Factor'] == factor]

    matrix = pd.DataFrame(index=available_models, columns=TRAIT_ORDER, dtype=float)
    n_mat  = pd.DataFrame(index=available_models, columns=TRAIT_ORDER, dtype=object)

    for _, row in fdf.iterrows():
        if row['Source'] in available_models and row['Trait'] in TRAIT_ORDER:
            matrix.loc[row['Source'], row['Trait']] = row['Pearson_r_means']
            n_mat.loc[row['Source'],  row['Trait']] = row['N_Subgroups']

    im = ax.imshow(matrix.values.astype(float), cmap='RdYlGn',
                   aspect='auto', vmin=-1, vmax=1)

    for i, src in enumerate(available_models):
        for j, trait in enumerate(TRAIT_ORDER):
            val = matrix.loc[src, trait]
            n   = n_mat.loc[src, trait]
            if pd.notna(val):
                txt_color = 'white' if abs(float(val)) > 0.7 else '#333'
                ax.text(j, i, f"{float(val):.2f}\n(n={n})",
                        ha='center', va='center', fontsize=9, color=txt_color)
            else:
                ax.text(j, i, '—', ha='center', va='center', fontsize=11, color='#aaa')

    short_traits = [t[:4] for t in TRAIT_ORDER]
    ax.set_xticks(range(len(TRAIT_ORDER)))
    ax.set_xticklabels(short_traits, fontsize=10)
    ax.set_title(factor.replace('_', ' '), fontsize=12, fontweight='bold')

    if ax is axes[0]:
        ax.set_yticks(range(len(available_models)))
        ax.set_yticklabels([SOURCE_LABELS[s] for s in available_models], fontsize=10)

fig.colorbar(im, ax=axes[-1], shrink=0.8, label='Pearson r')
fig.suptitle(
    "Alignment: Model Subgroup Means vs Human Subgroup Means (Pearson r)\n"
    "n = number of shared subgroups; Qwen NLP has fewer subgroups due to smaller dataset",
    fontsize=13, fontweight='bold', y=1.02,
)
fig.tight_layout()
out_fig = FIG_DIR / "fig_alignment_heatmap.png"
fig.savefig(out_fig, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved {out_fig}")

# Second figure: Pearson r on effect vectors (same layout)
fig2, axes2 = plt.subplots(1, n_factors, figsize=(4.5 * n_factors, 5), sharey=True)
if n_factors == 1:
    axes2 = [axes2]

for ax, factor in zip(axes2, FACTORS):
    fdf = corr_df[corr_df['Factor'] == factor]

    matrix = pd.DataFrame(index=available_models, columns=TRAIT_ORDER, dtype=float)
    n_mat  = pd.DataFrame(index=available_models, columns=TRAIT_ORDER, dtype=object)

    for _, row in fdf.iterrows():
        if row['Source'] in available_models and row['Trait'] in TRAIT_ORDER:
            matrix.loc[row['Source'], row['Trait']] = row['Pearson_r_effects']
            n_mat.loc[row['Source'],  row['Trait']] = row['N_Subgroups']

    im2 = ax.imshow(matrix.values.astype(float), cmap='RdYlGn',
                    aspect='auto', vmin=-1, vmax=1)

    for i, src in enumerate(available_models):
        for j, trait in enumerate(TRAIT_ORDER):
            val = matrix.loc[src, trait]
            n   = n_mat.loc[src, trait]
            if pd.notna(val):
                txt_color = 'white' if abs(float(val)) > 0.7 else '#333'
                ax.text(j, i, f"{float(val):.2f}\n(n={n})",
                        ha='center', va='center', fontsize=9, color=txt_color)
            else:
                ax.text(j, i, '—', ha='center', va='center', fontsize=11, color='#aaa')

    ax.set_xticks(range(len(TRAIT_ORDER)))
    ax.set_xticklabels([t[:4] for t in TRAIT_ORDER], fontsize=10)
    ax.set_title(factor.replace('_', ' '), fontsize=12, fontweight='bold')

    if ax is axes2[0]:
        ax.set_yticks(range(len(available_models)))
        ax.set_yticklabels([SOURCE_LABELS[s] for s in available_models], fontsize=10)

fig2.colorbar(im2, ax=axes2[-1], shrink=0.8, label='Pearson r')
fig2.suptitle(
    "Alignment: Model Demographic Effect Vectors vs Human (Pearson r)\n"
    "Effect = subgroup_mean − global_mean",
    fontsize=13, fontweight='bold', y=1.02,
)
fig2.tight_layout()
out_fig2 = FIG_DIR / "fig_alignment_effects_heatmap.png"
fig2.savefig(out_fig2, dpi=180, bbox_inches='tight')
plt.close(fig2)
print(f"Saved {out_fig2}")

print("\nDone.")
