"""
ANOVA pipeline: one-way and two-way analysis of Big Five traits.
Focuses on GPT Explicit vs GPT NLP vs Human baseline.

Outputs:
  output/dashboard/anova_oneway.csv        - one-way results (F, p, eta²)
  output/dashboard/anova_twoway.csv        - two-way results incl. interactions
  output/dashboard/anova_source_interaction.csv - factor × source interactions
  output/figures/fig4_oneway_eta2.png
  output/figures/fig5_source_interaction.png

Usage:
    python scripts/analysis/anova_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pingouin as pg
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent

SCORING_KEYS = {
    'E': {'E1':1,'E2':-1,'E3':1,'E4':-1,'E5':1,'E6':-1,'E7':1,'E8':-1,'E9':1,'E10':-1},
    'A': {'A1':-1,'A2':1,'A3':-1,'A4':1,'A5':-1,'A6':1,'A7':-1,'A8':1,'A9':1,'A10':1},
    'C': {'C1':1,'C2':-1,'C3':1,'C4':-1,'C5':1,'C6':-1,'C7':1,'C8':-1,'C9':1,'C10':1},
    'N': {'N1':-1,'N2':1,'N3':-1,'N4':1,'N5':-1,'N6':-1,'N7':-1,'N8':-1,'N9':-1,'N10':-1},
    'O': {'O1':1,'O2':-1,'O3':1,'O4':-1,'O5':1,'O6':-1,'O7':1,'O8':1,'O9':1,'O10':1},
}
TRAITS      = list(SCORING_KEYS.keys())
TRAIT_NAMES = {'E':'Extroversion','A':'Agreeableness','C':'Conscientiousness',
               'N':'Emotional Stability','O':'Openness'}
RACE_MAP    = {1:'Mixed Race',2:'Arctic',3:'Caucasian (European)',4:'Caucasian (Indian)',
               5:'Caucasian (Middle East)',6:'Caucasian (North African)',
               7:'Indigenous Australian',8:'Native American',9:'North East Asian',
               10:'Pacific Islander',11:'South East Asian',12:'West African',13:'Other'}
GENDER_MAP  = {1:'Male',2:'Female',3:'Other'}
AGE_BINS    = [0,17,25,34,50,120]
AGE_LABELS  = ['<18','18-25','26-34','35-50','50+']

FACTORS = ['Gender', 'Age_Group', 'Race', 'Country']
MIN_GROUP_N = 50   # minimum per group for ANOVA


# ── Helpers ────────────────────────────────────────────────────────────────
def age_group(age):
    for lo, hi, label in zip(AGE_BINS, AGE_BINS[1:], AGE_LABELS):
        if lo < age <= hi:
            return label
    return np.nan


def compute_traits(df, is_simulated=True):
    df = df.copy()
    for trait, keys in SCORING_KEYS.items():
        cols = []
        for q_id, sign in keys.items():
            col = f"{q_id}_score" if is_simulated else q_id
            df[col] = pd.to_numeric(df[col], errors='coerce')
            scored = f"_{q_id}_s"
            df[scored] = df[col] if sign == 1 else 6 - df[col]
            cols.append(scored)
        df[f"{trait}_Trait"] = df[cols].mean(axis=1)
        df.drop(columns=cols, inplace=True)
    return df


def filter_groups(df, factor, min_n=MIN_GROUP_N):
    """Drop factor levels with fewer than min_n observations."""
    counts = df[factor].value_counts()
    keep   = counts[counts >= min_n].index
    return df[df[factor].isin(keep)].copy()


def eta_squared_from_pg(aov_row):
    """Extract eta-squared from a pingouin anova result row."""
    if 'np2' in aov_row:
        return float(aov_row['np2'])
    return np.nan


# ── Load & prepare data ────────────────────────────────────────────────────
print("Loading data...")

# Human
h = pd.read_csv(ROOT / "data/human_data.csv", sep='\t', low_memory=False)
iso = pd.read_csv(ROOT / "data/iso.csv")
iso_map = dict(zip(iso['alpha-2'], iso['name']))
for col in TRAITS:
    for i in range(1, 11):
        h[f"{col}{i}"] = pd.to_numeric(h[f"{col}{i}"], errors='coerce')
h = h[h[['race','gender'] + [f"{t}{i}" for t in TRAITS for i in range(1,11)]
        ].notna().all(axis=1)].copy()
h = h[(pd.to_numeric(h['age'], errors='coerce').between(13, 100))].copy()
h['Age']      = pd.to_numeric(h['age'], errors='coerce')
h['Gender']   = pd.to_numeric(h['gender'], errors='coerce').map(GENDER_MAP)
h['Race']     = pd.to_numeric(h['race'], errors='coerce').map(RACE_MAP)
h['Country']  = h['country'].map(iso_map)
h['Age_Group'] = h['Age'].apply(age_group)
h['Source']   = 'Human'
h = compute_traits(h, is_simulated=False)

# GPT Explicit (first run only)
ex = pd.read_csv(ROOT / "output/simulated/simulated_human_data_isolated.csv", low_memory=False)
ex = ex[ex['Sim_ID'].str.contains('_R0')].copy()
ex['Age_Group'] = ex['Age'].apply(age_group)
ex['Source']    = 'GPT_Explicit'
ex = compute_traits(ex, is_simulated=True)

# GPT NLP
nlp = pd.read_csv(ROOT / "output/simulated/simulated_human_data_nlp.csv", low_memory=False)
nlp['Age_Group'] = nlp['Age'].apply(age_group)
nlp['Source']    = 'GPT_NLP'
nlp = compute_traits(nlp, is_simulated=True)

KEEP_COLS = ['Source', 'Gender', 'Age_Group', 'Race', 'Country'] + [f"{t}_Trait" for t in TRAITS]
combined  = pd.concat([h[KEEP_COLS], ex[KEEP_COLS], nlp[KEEP_COLS]], ignore_index=True)
combined  = combined.dropna(subset=['Gender', 'Age_Group'])

print(f"  Human: {len(h):,}  |  GPT Explicit: {len(ex):,}  |  GPT NLP: {len(nlp):,}")
SOURCES = ['Human', 'GPT_Explicit', 'GPT_NLP']


# ══════════════════════════════════════════════════════════════════════════
# 1. ONE-WAY ANOVA
# ══════════════════════════════════════════════════════════════════════════
print("\nRunning one-way ANOVA...")
oneway_results = []

for source in SOURCES:
    src_df = combined[combined['Source'] == source].copy()
    for factor in FACTORS:
        fdf = filter_groups(src_df.dropna(subset=[factor]), factor)
        if fdf[factor].nunique() < 2:
            continue
        for t in TRAITS:
            dv = f"{t}_Trait"
            try:
                aov = pg.anova(data=fdf, dv=dv, between=factor, detailed=True)
                row = aov[aov['Source'] == factor].iloc[0]
                oneway_results.append({
                    'Source':  source,
                    'Factor':  factor,
                    'Trait':   TRAIT_NAMES[t],
                    'Trait_Code': t,
                    'F':       round(float(row['F']), 3),
                    'p_raw':   float(row['p_unc']),
                    'eta2':    round(float(row['np2']), 4),
                    'n_groups': int(fdf[factor].nunique()),
                    'n_total':  int(len(fdf)),
                })
            except Exception as e:
                print(f"    WARN {source}/{factor}/{t}: {e}")

oneway_df = pd.DataFrame(oneway_results)

# BH correction within each source
adj_p = []
for source in SOURCES:
    mask = oneway_df['Source'] == source
    if mask.sum() == 0:
        continue
    _, p_adj, _, _ = multipletests(oneway_df.loc[mask, 'p_raw'], method='fdr_bh')
    adj_p.extend(list(zip(oneway_df[mask].index, p_adj)))
adj_map = dict(adj_p)
oneway_df['p_adj_bh'] = oneway_df.index.map(adj_map).round(4)
oneway_df['p_raw']    = oneway_df['p_raw'].round(4)
oneway_df['sig_bh']   = oneway_df['p_adj_bh'] < 0.05

oneway_df.to_csv(ROOT / "output/dashboard/anova_oneway.csv", index=False)
print(f"  One-way done: {len(oneway_df)} tests, {oneway_df['sig_bh'].sum()} survive BH")


# ══════════════════════════════════════════════════════════════════════════
# 2. TWO-WAY ANOVA (key factor pairs, within each source)
# ══════════════════════════════════════════════════════════════════════════
print("\nRunning two-way ANOVA...")
FACTOR_PAIRS = [('Gender', 'Age_Group'), ('Gender', 'Race'), ('Age_Group', 'Race')]
twoway_results = []

for source in SOURCES:
    src_df = combined[combined['Source'] == source].copy()
    for f1, f2 in FACTOR_PAIRS:
        fdf = src_df.dropna(subset=[f1, f2]).copy()
        # Filter both factors to groups with enough data
        for f in [f1, f2]:
            counts = fdf[f].value_counts()
            fdf = fdf[fdf[f].isin(counts[counts >= MIN_GROUP_N].index)]
        if fdf[f1].nunique() < 2 or fdf[f2].nunique() < 2:
            continue
        for t in TRAITS:
            dv = f"{t}_Trait"
            try:
                formula = f"{dv} ~ C({f1}) + C({f2}) + C({f1}):C({f2})"
                model   = ols(formula, data=fdf).fit()
                aov_tbl = anova_lm(model, typ=2)
                ss_total = aov_tbl['sum_sq'].sum()

                for effect_label, row_key in [
                    (f1,          f"C({f1})"),
                    (f2,          f"C({f2})"),
                    ('Interaction', f"C({f1}):C({f2})"),
                ]:
                    if row_key not in aov_tbl.index:
                        continue
                    r = aov_tbl.loc[row_key]
                    eta2 = r['sum_sq'] / ss_total if ss_total > 0 else np.nan
                    twoway_results.append({
                        'Source':    source,
                        'Factor1':   f1,
                        'Factor2':   f2,
                        'Effect':    effect_label,
                        'Trait':     TRAIT_NAMES[t],
                        'Trait_Code': t,
                        'F':         round(float(r['F']), 3),
                        'p_raw':     round(float(r['PR(>F)']), 4),
                        'eta2':      round(float(eta2), 4),
                    })
            except Exception as e:
                print(f"    WARN {source}/{f1}×{f2}/{t}: {e}")

twoway_df = pd.DataFrame(twoway_results)
twoway_df.to_csv(ROOT / "output/dashboard/anova_twoway.csv", index=False)
print(f"  Two-way done: {len(twoway_df)} effects")


# ══════════════════════════════════════════════════════════════════════════
# 3. SOURCE × FACTOR INTERACTION
#    Does the demographic effect differ between Human / GPT_Explicit / GPT_NLP?
# ══════════════════════════════════════════════════════════════════════════
print("\nRunning source × factor interaction ANOVA...")
source_int_results = []

for factor in ['Gender', 'Age_Group', 'Race']:
    fdf = combined.dropna(subset=[factor]).copy()
    for f in [factor]:
        counts = fdf[f].value_counts()
        fdf = fdf[fdf[f].isin(counts[counts >= MIN_GROUP_N].index)]
    if fdf[factor].nunique() < 2:
        continue
    for t in TRAITS:
        dv = f"{t}_Trait"
        try:
            formula = f"{dv} ~ C({factor}) + C(Source) + C({factor}):C(Source)"
            model   = ols(formula, data=fdf).fit()
            aov_tbl = anova_lm(model, typ=2)
            ss_total = aov_tbl['sum_sq'].sum()

            for effect_label, row_key in [
                (factor,       f"C({factor})"),
                ('Source',     'C(Source)'),
                ('Interaction', f"C({factor}):C(Source)"),
            ]:
                if row_key not in aov_tbl.index:
                    continue
                r = aov_tbl.loc[row_key]
                eta2 = r['sum_sq'] / ss_total if ss_total > 0 else np.nan
                source_int_results.append({
                    'Factor':    factor,
                    'Effect':    effect_label,
                    'Trait':     TRAIT_NAMES[t],
                    'Trait_Code': t,
                    'F':         round(float(r['F']), 3),
                    'p_raw':     round(float(r['PR(>F)']), 4),
                    'eta2':      round(float(eta2), 4),
                })
        except Exception as e:
            print(f"    WARN {factor}/{t}: {e}")

src_int_df = pd.DataFrame(source_int_results)
src_int_df.to_csv(ROOT / "output/dashboard/anova_source_interaction.csv", index=False)
print(f"  Source interaction done: {len(src_int_df)} effects")


# ══════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════
FIG_DIR = ROOT / "output/figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 12,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

SOURCE_LABELS = {'Human':'Human', 'GPT_Explicit':'GPT Explicit', 'GPT_NLP':'GPT NLP'}
TRAIT_ORDER   = [TRAIT_NAMES[t] for t in TRAITS]
FACTOR_ORDER  = ['Gender', 'Age_Group', 'Race', 'Country']


# ── Fig 4: One-way eta² heatmap ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

vmax = oneway_df['eta2'].max()

for ax, source in zip(axes, SOURCES):
    sub = oneway_df[oneway_df['Source'] == source].copy()
    pivot = sub.pivot(index='Trait', columns='Factor', values='eta2')
    pivot = pivot.reindex(index=TRAIT_ORDER, columns=FACTOR_ORDER)

    sig_pivot = sub.pivot(index='Trait', columns='Factor', values='sig_bh')
    sig_pivot = sig_pivot.reindex(index=TRAIT_ORDER, columns=FACTOR_ORDER)

    im = ax.imshow(pivot.values.astype(float), cmap='Oranges',
                   aspect='auto', vmin=0, vmax=vmax)

    for i, trait in enumerate(TRAIT_ORDER):
        for j, factor in enumerate(FACTOR_ORDER):
            val = pivot.iloc[i, j] if pivot.iloc[i, j] is not None else np.nan
            sig = sig_pivot.iloc[i, j] if sig_pivot.iloc[i, j] is not None else False
            if np.isnan(float(val)) if val is not None else True:
                ax.text(j, i, '—', ha='center', va='center', fontsize=11, color='#aaa')
            else:
                star = '*' if sig else ''
                txt_color = 'white' if float(val) > vmax * 0.6 else '#333'
                ax.text(j, i, f"{float(val):.3f}{star}", ha='center', va='center',
                        fontsize=11, color=txt_color)

    ax.set_xticks(range(len(FACTOR_ORDER)))
    ax.set_xticklabels(FACTOR_ORDER, fontsize=11, rotation=20, ha='right')
    ax.set_yticks(range(len(TRAIT_ORDER)))
    ax.set_yticklabels(TRAIT_ORDER, fontsize=11)
    ax.set_title(SOURCE_LABELS[source])
    fig.colorbar(im, ax=ax, shrink=0.7, label='η² (eta-squared)')

fig.suptitle("One-Way ANOVA: Eta-Squared by Trait × Demographic Factor\n"
             "(*  = significant after BH correction)", fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_oneway_eta2.png", dpi=180, bbox_inches='tight')
plt.close(fig)
print("\nSaved fig4_oneway_eta2.png")


# ── Fig 5: Source × Factor interaction (does factor effect differ by source?) ──
int_df = src_int_df[src_int_df['Effect'] == 'Interaction'].copy()
int_df['p_sig'] = int_df['p_raw'] < 0.05
FACTOR_INT_ORDER = ['Gender', 'Age_Group', 'Race']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: eta² of the interaction term
pivot_eta = int_df.pivot(index='Trait', columns='Factor', values='eta2')
pivot_eta = pivot_eta.reindex(index=TRAIT_ORDER, columns=FACTOR_INT_ORDER)
pivot_sig = int_df.pivot(index='Trait', columns='Factor', values='p_sig')
pivot_sig = pivot_sig.reindex(index=TRAIT_ORDER, columns=FACTOR_INT_ORDER)

vmax2 = float(pivot_eta.max().max()) if not pivot_eta.empty else 0.1
im = axes[0].imshow(pivot_eta.values.astype(float), cmap='Purples',
                    aspect='auto', vmin=0, vmax=max(vmax2, 0.01))
for i, trait in enumerate(TRAIT_ORDER):
    for j, factor in enumerate(FACTOR_INT_ORDER):
        val = pivot_eta.iloc[i, j]
        sig = pivot_sig.iloc[i, j]
        if pd.isna(val):
            axes[0].text(j, i, '—', ha='center', va='center', fontsize=11, color='#aaa')
        else:
            star = '*' if sig else ''
            txt_color = 'white' if float(val) > vmax2 * 0.6 else '#333'
            axes[0].text(j, i, f"{float(val):.4f}{star}", ha='center', va='center',
                         fontsize=11, color=txt_color)
axes[0].set_xticks(range(len(FACTOR_INT_ORDER)))
axes[0].set_xticklabels(FACTOR_INT_ORDER, fontsize=11)
axes[0].set_yticks(range(len(TRAIT_ORDER)))
axes[0].set_yticklabels(TRAIT_ORDER, fontsize=11)
axes[0].set_title("Factor × Source Interaction (η²)\n"
                  "(* = p < 0.05; does demographic effect differ between sources?)")
fig.colorbar(im, ax=axes[0], shrink=0.8, label='η²')

# Panel B: main effect of Source (overall GPT vs Human offset per trait × factor)
src_df_main = src_int_df[src_int_df['Effect'] == 'Source'].copy()
pivot_src   = src_df_main.pivot(index='Trait', columns='Factor', values='eta2')
pivot_src   = pivot_src.reindex(index=TRAIT_ORDER, columns=FACTOR_INT_ORDER)
pivot_src_sig = src_df_main.pivot(index='Trait', columns='Factor', values='p_raw') < 0.05
pivot_src_sig = pivot_src_sig.reindex(index=TRAIT_ORDER, columns=FACTOR_INT_ORDER)

vmax3 = float(pivot_src.max().max()) if not pivot_src.empty else 0.1
im2 = axes[1].imshow(pivot_src.values.astype(float), cmap='Reds',
                     aspect='auto', vmin=0, vmax=max(vmax3, 0.01))
for i, trait in enumerate(TRAIT_ORDER):
    for j, factor in enumerate(FACTOR_INT_ORDER):
        val = pivot_src.iloc[i, j]
        sig = pivot_src_sig.iloc[i, j]
        if pd.isna(val):
            axes[1].text(j, i, '—', ha='center', va='center', fontsize=11, color='#aaa')
        else:
            star = '*' if sig else ''
            txt_color = 'white' if float(val) > vmax3 * 0.6 else '#333'
            axes[1].text(j, i, f"{float(val):.4f}{star}", ha='center', va='center',
                         fontsize=11, color=txt_color)
axes[1].set_xticks(range(len(FACTOR_INT_ORDER)))
axes[1].set_xticklabels(FACTOR_INT_ORDER, fontsize=11)
axes[1].set_yticks(range(len(TRAIT_ORDER)))
axes[1].set_yticklabels(TRAIT_ORDER, fontsize=11)
axes[1].set_title("Main Effect of Source (η²)\n"
                  "(overall GPT vs Human offset, controlling for demographics)")
fig.colorbar(im2, ax=axes[1], shrink=0.8, label='η²')

fig.suptitle("Does the Demographic Effect Differ Between Human / GPT Explicit / GPT NLP?\n"
             "Two-way ANOVA: Trait ~ Factor + Source + Factor×Source",
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_source_interaction.png", dpi=180, bbox_inches='tight')
plt.close(fig)
print("Saved fig5_source_interaction.png")

print("\nDone. All outputs in output/dashboard/ and output/figures/")
