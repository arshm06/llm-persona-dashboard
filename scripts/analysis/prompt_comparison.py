"""
Prompt comparison: how does NLP vs Explicit prompting compare to human demographic patterns?

For each trait x demographic factor:
  - Extracts per-subgroup means from Human, GPT-Explicit, GPT-NLP, Qwen-NLP
  - Computes demographic effect = subgroup_mean - global_mean for each source
  - Quantifies whether NLP exaggerates or compresses demographic effects vs Explicit

Outputs:
  output/dashboard/prompt_comparison.csv          - effect vectors per (source, factor, subgroup, trait)
  output/dashboard/prompt_comparison_summary.csv  - directional comparison per subgroup
  output/figures/fig_prompt_comparison_{factor}.png

Usage:
    python scripts/analysis/prompt_comparison.py
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
DEMO_COLS = ['Country', 'Race', 'Age', 'Age_Group', 'Gender']
FACTORS = ['Gender', 'Age_Group', 'Race', 'Country']
MIN_N = 30

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


def get_global_row(df, source):
    mask = (df['Source'] == source) & (df[DEMO_COLS] == 'All').all(axis=1)
    rows = df[mask]
    return rows.iloc[0] if not rows.empty else None


# ── Load data ──────────────────────────────────────────────────────────────
print("Loading precalculated stats...")
df = pd.read_csv(ROOT / "output/dashboard/dashboard_precalc_stats_singular.csv", low_memory=False)
for col in DEMO_COLS:
    if col in df.columns:
        df[col] = df[col].astype(str)

available_sources = [s for s in SOURCES if s in df['Source'].unique()]
print(f"Available sources: {available_sources}")

# Precompute global means per source
global_rows = {src: get_global_row(df, src) for src in available_sources}


# ── Build effect records ───────────────────────────────────────────────────
effect_records = []
for factor in FACTORS:
    factor_df = get_single_factor_rows(df, factor)

    for source in available_sources:
        gm = global_rows[source]
        if gm is None:
            continue
        src_df = factor_df[factor_df['Source'] == source]

        for _, row in src_df.iterrows():
            for t in TRAITS:
                mean_col = f"{t}_Trait_mean"
                n_col    = f"{t}_Trait_count"
                if mean_col not in row or pd.isna(row[mean_col]):
                    continue
                n = int(row[n_col]) if n_col in row and pd.notna(row[n_col]) else 0
                if n < MIN_N:
                    continue
                effect_records.append({
                    'Factor':       factor,
                    'Subgroup':     row[factor],
                    'Trait_Code':   t,
                    'Trait':        TRAIT_NAMES[t],
                    'Source':       source,
                    'Source_Label': SOURCE_LABELS[source],
                    'Subgroup_Mean': round(float(row[mean_col]), 4),
                    'Global_Mean':   round(float(gm[mean_col]), 4),
                    'Effect':        round(float(row[mean_col]) - float(gm[mean_col]), 4),
                    'N':             n,
                })

effects_df = pd.DataFrame(effect_records)
print(f"Effect records: {len(effects_df)}")

out_csv = ROOT / "output/dashboard/prompt_comparison.csv"
effects_df.to_csv(out_csv, index=False)
print(f"Saved {out_csv}")


# ── Directional summary: does NLP bring model closer to human than Explicit? ──
ref_src  = 'Human'
explicit = 'AI_GPT4o_Explicit'
nlp_src  = 'AI_GPT4o_NLP'
qnlp_src = 'AI_Qwen_NLP'

summary_records = []
for factor in FACTORS:
    fdf = effects_df[effects_df['Factor'] == factor]
    for t in TRAITS:
        tdf = fdf[fdf['Trait_Code'] == t]
        pivot = tdf.pivot(index='Subgroup', columns='Source', values='Effect')
        needed = [ref_src, explicit, nlp_src]
        if not all(s in pivot.columns for s in needed):
            continue
        pivot = pivot[needed + ([qnlp_src] if qnlp_src in pivot.columns else [])].dropna(
            subset=needed)

        for sub, row in pivot.iterrows():
            h_eff  = row[ref_src]
            ex_eff = row[explicit]
            gn_eff = row[nlp_src]
            qn_eff = row.get(qnlp_src, np.nan)

            ex_dist = abs(ex_eff - h_eff)
            gn_dist = abs(gn_eff - h_eff)

            summary_records.append({
                'Factor':                      factor,
                'Subgroup':                    sub,
                'Trait_Code':                  t,
                'Trait':                       TRAIT_NAMES[t],
                'Human_Effect':                round(h_eff,  4),
                'GPT_Explicit_Effect':         round(ex_eff, 4),
                'GPT_NLP_Effect':              round(gn_eff, 4),
                'Qwen_NLP_Effect':             round(qn_eff, 4) if not np.isnan(qn_eff) else np.nan,
                'GPT_Explicit_Dist_Human':     round(ex_dist, 4),
                'GPT_NLP_Dist_Human':          round(gn_dist, 4),
                'GPT_NLP_closer_than_Explicit': gn_dist < ex_dist,
                'Human_exaggerates':           False,
                'GPT_Explicit_exaggerates':    abs(ex_eff) > abs(h_eff),
                'GPT_NLP_exaggerates':         abs(gn_eff) > abs(h_eff),
                'Qwen_NLP_exaggerates':        abs(qn_eff) > abs(h_eff) if not np.isnan(qn_eff) else np.nan,
            })

summary_df = pd.DataFrame(summary_records)
out_summary = ROOT / "output/dashboard/prompt_comparison_summary.csv"
summary_df.to_csv(out_summary, index=False)
print(f"Saved {out_summary}")

# Print high-level numbers
total = len(summary_df)
closer = summary_df['GPT_NLP_closer_than_Explicit'].sum()
print(f"\nGPT NLP closer to human than Explicit: {closer}/{total} ({100*closer/total:.1f}%)")
print("By trait:")
for t in TRAITS:
    tdf = summary_df[summary_df['Trait_Code'] == t]
    g = tdf['GPT_NLP_closer_than_Explicit'].sum()
    ex_ex = tdf['GPT_Explicit_exaggerates'].sum()
    gn_ex = tdf['GPT_NLP_exaggerates'].sum()
    print(f"  {TRAIT_NAMES[t]:<22} NLP closer: {g}/{len(tdf)} | "
          f"Explicit exaggerates: {ex_ex}/{len(tdf)} | NLP exaggerates: {gn_ex}/{len(tdf)}")


# ── Figures: grouped bar plots per factor ──────────────────────────────────
TRAIT_ORDER = ['E', 'A', 'C', 'N', 'O']

for factor in FACTORS:
    fdf = effects_df[effects_df['Factor'] == factor].copy()

    # Common subgroups across all available sources with data
    src_subs = {
        src: set(fdf[fdf['Source'] == src]['Subgroup'])
        for src in available_sources
    }
    common = sorted(set.intersection(*src_subs.values()))

    # For Country, keep only the 15 largest human subgroups to stay readable
    if factor == 'Country' and len(common) > 15:
        human_n = (fdf[fdf['Source'] == ref_src]
                   .groupby('Subgroup')['N'].first()
                   .reindex(common)
                   .fillna(0))
        common = human_n.nlargest(15).index.tolist()

    if len(common) < 2:
        print(f"Skipping {factor}: <2 common subgroups")
        continue

    n_subs = len(common)
    n_src  = len(available_sources)
    width  = 0.8 / n_src
    x      = np.arange(n_subs)

    fig, axes = plt.subplots(len(TRAIT_ORDER), 1,
                             figsize=(max(10, n_subs * 0.9 + 2), 3.2 * len(TRAIT_ORDER)),
                             sharex=True)

    for ax, t in zip(axes, TRAIT_ORDER):
        for si, src in enumerate(available_sources):
            src_fdf = fdf[(fdf['Source'] == src) & (fdf['Trait_Code'] == t)]
            sub_map = src_fdf.set_index('Subgroup')['Effect'].to_dict()
            vals    = [sub_map.get(s, np.nan) for s in common]
            offset  = (si - (n_src - 1) / 2) * width
            ax.bar(x + offset, vals, width=width * 0.92,
                   color=PALETTE[src], label=SOURCE_LABELS[src], alpha=0.85,
                   edgecolor='white', linewidth=0.4)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
        ax.set_ylabel(TRAIT_NAMES[t], fontsize=10)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(common, rotation=40, ha='right', fontsize=9)

    legend_patches = [mpatches.Patch(color=PALETTE[s], label=SOURCE_LABELS[s])
                      for s in available_sources]
    fig.legend(handles=legend_patches, loc='upper right', frameon=False,
               fontsize=10, ncol=len(available_sources))
    fig.suptitle(
        f"Demographic Effect Size by Prompting Method — Factor: {factor.replace('_', ' ')}\n"
        f"(bars = subgroup mean − global mean; Qwen NLP note: n=1,200 vs ~20k for others)",
        fontsize=12, fontweight='bold', y=1.01,
    )
    fig.tight_layout()
    out_fig = FIG_DIR / f"fig_prompt_comparison_{factor.lower()}.png"
    fig.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_fig}")

print("\nDone.")
