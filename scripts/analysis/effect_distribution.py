"""
Effect size distribution and variance collapse analysis.

1. Per-factor violin plots of Cohen's d for Human, GPT-Explicit, GPT-NLP, Qwen-NLP
2. Variance collapse: ratio of within-subgroup std (model / human) per subgroup x trait
   — values < 1 indicate the model produces less within-group personality diversity than humans

Outputs:
  output/dashboard/variance_collapse.csv                 - std ratios per (source, factor, subgroup, trait)
  output/figures/fig_effect_distribution_{factor}.png   - per-factor violin plots
  output/figures/fig_variance_collapse.png              - heatmap of mean std ratios

Usage:
    python scripts/analysis/effect_distribution.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

TRAITS = ['E', 'A', 'C', 'N', 'O']
TRAIT_NAMES = {
    'E': 'Extroversion', 'A': 'Agreeableness', 'C': 'Conscientiousness',
    'N': 'Emotional Stability', 'O': 'Openness',
}
TRAIT_ORDER = [TRAIT_NAMES[t] for t in TRAITS]
DEMO_COLS   = ['Country', 'Race', 'Age', 'Age_Group', 'Gender']
FACTORS     = ['Gender', 'Age_Group', 'Race', 'Country']
MIN_N       = 30

SOURCES = ['Human', 'AI_GPT4o_Explicit', 'AI_GPT4o_NLP', 'AI_Qwen_NLP']
SOURCE_LABELS = {
    'Human':             'Human',
    'AI_GPT4o_Explicit': 'GPT Explicit',
    'AI_GPT4o_NLP':      'GPT NLP',
    'AI_Qwen_NLP':       'Qwen NLP',
}
PALETTE = {
    'Human':             '#2C7BB6',
    'AI_GPT4o_Explicit': '#D7191C',
    'AI_GPT4o_NLP':      '#E87D2B',
    'AI_Qwen_NLP':       '#888888',
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


# ── Load BH results (has Cohen's d per subgroup x source) ─────────────────
print("Loading BH results...")
bh_df = pd.read_csv(ROOT / "output/dashboard/bh_results_all.csv")
available = [s for s in SOURCES if s in bh_df['Source'].unique()]
bh_df = bh_df[bh_df['Source'].isin(available)].copy()
bh_df['Factor'] = bh_df['Subgroup'].str.split('=').str[0]
bh_df = bh_df[bh_df['Factor'].isin(FACTORS)].copy()
print(f"  {len(bh_df)} records | sources: {bh_df['Source'].unique().tolist()}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE SET 1: Per-factor violin plots of Cohen's d
# ══════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(42)

for factor in FACTORS:
    fdf = bh_df[bh_df['Factor'] == factor].copy()
    if fdf.empty:
        continue

    fig, axes = plt.subplots(1, len(TRAITS), figsize=(3.5 * len(TRAITS), 5), sharey=False)

    for ax, t in zip(axes, TRAITS):
        tdf = fdf[fdf['Trait_Code'] == t]
        data_by_src, labels, colors, medians = [], [], [], []

        for src in available:
            vals = tdf[tdf['Source'] == src]['Cohens_d'].dropna().values
            if len(vals) >= 2:
                data_by_src.append(vals)
                labels.append(SOURCE_LABELS[src])
                colors.append(PALETTE[src])
                medians.append(np.median(vals))

        if not data_by_src:
            ax.set_visible(False)
            continue

        positions = list(range(1, len(data_by_src) + 1))
        vp = ax.violinplot(data_by_src, positions=positions,
                           showmedians=True, showextrema=False, widths=0.65)
        for body, color in zip(vp['bodies'], colors):
            body.set_facecolor(color)
            body.set_alpha(0.75)
        vp['cmedians'].set_color('white')
        vp['cmedians'].set_linewidth(2)

        for pos, vals, color in zip(positions, data_by_src, colors):
            jitter = rng.uniform(-0.1, 0.1, len(vals))
            ax.scatter(pos + jitter, vals, color=color, alpha=0.3, s=10, linewidths=0)

        # Annotate median
        for pos, med in zip(positions, medians):
            ax.text(pos, ax.get_ylim()[0] if ax.get_ylim()[1] > 0 else 0,
                    f"{med:.2f}", ha='center', va='bottom', fontsize=8, color='#555')

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_title(TRAIT_NAMES[t], fontsize=11)
        if t == TRAITS[0]:
            ax.set_ylabel("Cohen's d (vs human global baseline)", fontsize=10)

    legend_patches = [mpatches.Patch(color=PALETTE[s], label=SOURCE_LABELS[s])
                      for s in available]
    fig.legend(handles=legend_patches, loc='upper right', frameon=False,
               fontsize=10, ncol=len(available))
    fig.suptitle(
        f"Effect Size Distribution by Prompting Method — Factor: {factor.replace('_', ' ')}\n"
        f"(each point = one subgroup; Cohen's d vs human global mean)",
        fontsize=12, fontweight='bold', y=1.01,
    )
    fig.tight_layout()
    out_fig = FIG_DIR / f"fig_effect_distribution_{factor.lower()}.png"
    fig.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_fig}")


# Also produce a combined overview across all factors
fig, axes = plt.subplots(len(FACTORS), len(TRAITS),
                         figsize=(3.2 * len(TRAITS), 3.0 * len(FACTORS)),
                         sharey='row')

for fi, factor in enumerate(FACTORS):
    fdf = bh_df[bh_df['Factor'] == factor].copy()
    for ti, t in enumerate(TRAITS):
        ax  = axes[fi][ti]
        tdf = fdf[fdf['Trait_Code'] == t]
        data_by_src, colors = [], []
        for src in available:
            vals = tdf[tdf['Source'] == src]['Cohens_d'].dropna().values
            if len(vals) >= 2:
                data_by_src.append(vals)
                colors.append(PALETTE[src])

        if not data_by_src:
            ax.set_visible(False)
            continue

        positions = list(range(1, len(data_by_src) + 1))
        vp = ax.violinplot(data_by_src, positions=positions,
                           showmedians=True, showextrema=False, widths=0.65)
        for body, color in zip(vp['bodies'], colors):
            body.set_facecolor(color)
            body.set_alpha(0.75)
        vp['cmedians'].set_color('white')
        vp['cmedians'].set_linewidth(1.5)

        ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
        ax.set_xticks([])
        if ti == 0:
            ax.set_ylabel(factor.replace('_', ' '), fontsize=9)
        if fi == 0:
            ax.set_title(TRAIT_NAMES[t][:5], fontsize=9)

legend_patches = [mpatches.Patch(color=PALETTE[s], label=SOURCE_LABELS[s])
                  for s in available]
fig.legend(handles=legend_patches, loc='lower center', ncol=len(available),
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Effect Size Distributions: Factor × Trait", fontsize=12, fontweight='bold')
fig.tight_layout()
out_fig = FIG_DIR / "fig_effect_distribution_overview.png"
fig.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out_fig}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Variance collapse
# Within-subgroup std ratio (model / human) per (source, factor, subgroup, trait)
# ══════════════════════════════════════════════════════════════════════════
print("\nComputing variance collapse...")
stats_df = pd.read_csv(ROOT / "output/dashboard/dashboard_precalc_stats_singular.csv",
                       low_memory=False)
for col in DEMO_COLS:
    if col in stats_df.columns:
        stats_df[col] = stats_df[col].astype(str)

model_sources = [s for s in SOURCES if s != 'Human' and s in stats_df['Source'].unique()]

collapse_records = []
for factor in FACTORS:
    fdf = get_single_factor_rows(stats_df, factor)
    human_fdf = fdf[fdf['Source'] == 'Human']

    for t in TRAITS:
        std_col = f"{t}_Trait_std"
        n_col   = f"{t}_Trait_count"

        h_data = (human_fdf[[factor, std_col, n_col]]
                  .copy()
                  .rename(columns={std_col: 'h_std', n_col: 'h_n'}))
        h_data = h_data[h_data['h_n'].astype(float) >= MIN_N].dropna(subset=['h_std'])

        for model_src in model_sources:
            m_fdf = fdf[fdf['Source'] == model_src]
            m_data = (m_fdf[[factor, std_col, n_col]]
                      .copy()
                      .rename(columns={std_col: 'm_std', n_col: 'm_n'}))
            m_data = m_data[m_data['m_n'].astype(float) >= MIN_N].dropna(subset=['m_std'])

            merged = h_data.merge(m_data, on=factor)
            for _, row in merged.iterrows():
                h_std = float(row['h_std'])
                m_std = float(row['m_std'])
                ratio = m_std / h_std if h_std > 0 else np.nan
                collapse_records.append({
                    'Factor':       factor,
                    'Subgroup':     row[factor],
                    'Trait_Code':   t,
                    'Trait':        TRAIT_NAMES[t],
                    'Source':       model_src,
                    'Source_Label': SOURCE_LABELS[model_src],
                    'Human_Std':    round(h_std, 4),
                    'Model_Std':    round(m_std, 4),
                    'Std_Ratio':    round(ratio, 4) if not np.isnan(ratio) else np.nan,
                })

collapse_df = pd.DataFrame(collapse_records)
out_csv = ROOT / "output/dashboard/variance_collapse.csv"
collapse_df.to_csv(out_csv, index=False)
print(f"Saved {out_csv} ({len(collapse_df)} records)")

print("\nMean std ratio (model / human) — values < 1 = variance collapse:")
for src in model_sources:
    sdf = collapse_df[collapse_df['Source'] == src]
    print(f"  {SOURCE_LABELS[src]:<14}: {sdf['Std_Ratio'].mean():.3f}  "
          f"(median {sdf['Std_Ratio'].median():.3f})")

# Heatmap: mean std ratio by (source x trait), one row per source
pivot = (collapse_df
         .groupby(['Source_Label', 'Trait'])['Std_Ratio']
         .mean()
         .unstack()
         .reindex(index=[SOURCE_LABELS[s] for s in model_sources],
                  columns=TRAIT_ORDER))

fig, ax = plt.subplots(figsize=(10, 4))
vmax = max(1.5, float(np.nanmax(pivot.values)))
vmin = min(0.5, float(np.nanmin(pivot.values)))
im = ax.imshow(pivot.values.astype(float), cmap='RdYlGn',
               aspect='auto', vmin=vmin, vmax=vmax)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.iloc[i, j]
        if pd.notna(val):
            txt_color = 'white' if abs(float(val) - 1.0) > 0.25 else '#333'
            ax.text(j, i, f"{float(val):.2f}",
                    ha='center', va='center', fontsize=12, color=txt_color)

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=11)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=11)
ax.set_title(
    "Variance Collapse: Mean (Model Within-Subgroup Std / Human Within-Subgroup Std)\n"
    "< 1 = model produces less within-group diversity; averaged across all factors & subgroups",
    fontsize=12, fontweight='bold',
)
fig.colorbar(im, ax=ax, shrink=0.8, label='Std Ratio (model / human)')
fig.tight_layout()
out_fig = FIG_DIR / "fig_variance_collapse.png"
fig.savefig(out_fig, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out_fig}")

# Per-factor breakdown of variance collapse
fig2, axes2 = plt.subplots(1, len(FACTORS), figsize=(4.0 * len(FACTORS), 4), sharey=True)
for ax, factor in zip(axes2, FACTORS):
    fdf = collapse_df[collapse_df['Factor'] == factor]
    pivot_f = (fdf.groupby(['Source_Label', 'Trait'])['Std_Ratio']
                  .mean()
                  .unstack()
                  .reindex(index=[SOURCE_LABELS[s] for s in model_sources],
                           columns=TRAIT_ORDER))
    im2 = ax.imshow(pivot_f.values.astype(float), cmap='RdYlGn',
                    aspect='auto', vmin=vmin, vmax=vmax)
    for i in range(len(pivot_f.index)):
        for j in range(len(pivot_f.columns)):
            val = pivot_f.iloc[i, j]
            if pd.notna(val):
                txt_color = 'white' if abs(float(val) - 1.0) > 0.25 else '#333'
                ax.text(j, i, f"{float(val):.2f}",
                        ha='center', va='center', fontsize=9, color=txt_color)
    ax.set_xticks(range(len(TRAIT_ORDER)))
    ax.set_xticklabels([t[:4] for t in TRAIT_ORDER], fontsize=9)
    ax.set_title(factor.replace('_', ' '), fontsize=11, fontweight='bold')
    if ax is axes2[0]:
        ax.set_yticks(range(len(pivot_f.index)))
        ax.set_yticklabels(pivot_f.index, fontsize=10)

fig2.colorbar(im2, ax=axes2[-1], shrink=0.8, label='Std Ratio')
fig2.suptitle("Variance Collapse by Factor (Std Ratio model / human)",
              fontsize=12, fontweight='bold', y=1.02)
fig2.tight_layout()
out_fig2 = FIG_DIR / "fig_variance_collapse_by_factor.png"
fig2.savefig(out_fig2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"Saved {out_fig2}")

print("\nDone.")
