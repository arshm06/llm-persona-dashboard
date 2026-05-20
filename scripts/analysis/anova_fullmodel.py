"""
Full additive ANOVA model: Big Five traits ~ demographics + prompt type + model.

Treats demographic factors (Gender, Age_Group, Race) alongside methodological
factors (PromptType, Model) as co-equal predictors in a single OLS model.
Partial η² for each predictor directly answers:
  "Does using GPT vs Qwen explain more trait variance than knowing someone's gender?"

Sources:
  Human          → Model=Human,  PromptType=Survey
  GPT4o Explicit → Model=GPT4o,  PromptType=Explicit
  GPT4o NLP      → Model=GPT4o,  PromptType=NLP
  Qwen NLP       → Model=Qwen,   PromptType=NLP

Usage:
    python scripts/analysis/anova_fullmodel.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

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
TRAIT_ORDER = [TRAIT_NAMES[t] for t in TRAITS]

RACE_MAP   = {1:'Mixed Race',2:'Arctic',3:'Caucasian (European)',4:'Caucasian (Indian)',
              5:'Caucasian (Middle East)',6:'Caucasian (North African)',
              7:'Indigenous Australian',8:'Native American',9:'North East Asian',
              10:'Pacific Islander',11:'South East Asian',12:'West African',13:'Other'}
GENDER_MAP = {1:'Male',2:'Female',3:'Other'}
AGE_BINS   = [0,17,25,34,50,120]
AGE_LABELS = ['<18','18-25','26-34','35-50','50+']
MIN_GROUP_N = 30


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


def filter_small_groups(df, factor, min_n=MIN_GROUP_N):
    counts = df[factor].value_counts()
    keep   = counts[counts >= min_n].index
    return df[df[factor].isin(keep)].copy()


# ── Load data ──────────────────────────────────────────────────────────────
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
h = h[pd.to_numeric(h['age'], errors='coerce').between(13, 100)].copy()
h['Age']       = pd.to_numeric(h['age'], errors='coerce')
h['Gender']    = pd.to_numeric(h['gender'], errors='coerce').map(GENDER_MAP)
h['Race']      = pd.to_numeric(h['race'], errors='coerce').map(RACE_MAP)
h['Country']   = h['country'].map(iso_map)
h['Age_Group'] = h['Age'].apply(age_group)
h['PromptType'] = 'Survey'
h['Model']      = 'Human'
h = compute_traits(h, is_simulated=False)

# GPT4o Explicit (R0 only to avoid pseudo-replication)
ex = pd.read_csv(ROOT / "output/simulated/simulated_human_data_isolated.csv", low_memory=False)
ex = ex[ex['Sim_ID'].str.contains('_R0')].copy()
ex['Age_Group']  = ex['Age'].apply(age_group)
ex['PromptType'] = 'Explicit'
ex['Model']      = 'GPT4o'
ex = compute_traits(ex, is_simulated=True)

# GPT4o NLP
nlp = pd.read_csv(ROOT / "output/simulated/simulated_human_data_nlp.csv", low_memory=False)
nlp['Age_Group']  = nlp['Age'].apply(age_group)
nlp['PromptType'] = 'NLP'
nlp['Model']      = 'GPT4o'
nlp = compute_traits(nlp, is_simulated=True)

# Qwen NLP
qwen = pd.read_csv(ROOT / "output/simulated/simulated_human_data_nlp_qwen.csv", low_memory=False)
qwen['Age_Group']  = qwen['Age'].apply(age_group)
qwen['PromptType'] = 'NLP'
qwen['Model']      = 'Qwen'
qwen = compute_traits(qwen, is_simulated=True)

KEEP_COLS = ['PromptType', 'Model', 'Gender', 'Age_Group', 'Race'] + \
            [f"{t}_Trait" for t in TRAITS]

combined = pd.concat([h[KEEP_COLS], ex[KEEP_COLS], nlp[KEEP_COLS], qwen[KEEP_COLS]],
                     ignore_index=True)
combined = combined.dropna(subset=['Gender', 'Age_Group', 'Race'])

print(f"  Human: {len(h):,} | GPT4o Explicit: {len(ex):,} | "
      f"GPT4o NLP: {len(nlp):,} | Qwen NLP: {len(qwen):,}")
print(f"  Combined: {len(combined):,} rows")

# Filter small groups per factor
for factor in ['Gender', 'Age_Group', 'Race']:
    combined = filter_small_groups(combined, factor)
print(f"  After min-group filter: {len(combined):,} rows")


# ── Full additive model ────────────────────────────────────────────────────
# Trait ~ C(Gender) + C(Age_Group) + C(Race) + C(PromptType) + C(Model)
# Partial η² = SS_effect / (SS_effect + SS_residual)   [Type II SS]
# ──────────────────────────────────────────────────────────────────────────
PREDICTORS = ['Gender', 'Age_Group', 'Race', 'PromptType', 'Model']
PRED_LABELS = ['Gender', 'Age Group', 'Race', 'Prompt Type', 'Model']

print("\nFitting full additive models...")
results = []

for t in TRAITS:
    dv = f"{t}_Trait"
    formula = (f"{dv} ~ C(Gender) + C(Age_Group) + C(Race) "
               f"+ C(PromptType) + C(Model)")
    try:
        model   = ols(formula, data=combined).fit()
        aov_tbl = anova_lm(model, typ=2)
        ss_res  = aov_tbl.loc['Residual', 'sum_sq']

        for pred, label in zip(PREDICTORS, PRED_LABELS):
            key = f"C({pred})"
            if key not in aov_tbl.index:
                continue
            r = aov_tbl.loc[key]
            partial_eta2 = r['sum_sq'] / (r['sum_sq'] + ss_res)
            results.append({
                'Trait':        TRAIT_NAMES[t],
                'Trait_Code':   t,
                'Predictor':    label,
                'Predictor_Key': pred,
                'F':            round(float(r['F']), 3),
                'p_raw':        round(float(r['PR(>F)']), 6),
                'partial_eta2': round(float(partial_eta2), 5),
            })
        print(f"  {TRAIT_NAMES[t]}: R²={model.rsquared:.3f}")
    except Exception as e:
        print(f"  ERROR {TRAIT_NAMES[t]}: {e}")

results_df = pd.DataFrame(results)

# BH correction across all 25 tests (5 traits × 5 predictors)
_, p_adj, _, _ = multipletests(results_df['p_raw'], method='fdr_bh')
results_df['p_adj_bh'] = p_adj.round(6)
results_df['sig_bh']   = results_df['p_adj_bh'] < 0.05

out_path = ROOT / "output/dashboard/anova_fullmodel.csv"
results_df.to_csv(out_path, index=False)
print(f"\nSaved {out_path}")
print(results_df[['Trait','Predictor','partial_eta2','p_adj_bh','sig_bh']].to_string(index=False))


# ── Figure 6: Partial η² heatmap ──────────────────────────────────────────
FIG_DIR = ROOT / "output/figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 12,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

pivot_eta = results_df.pivot(index='Trait', columns='Predictor', values='partial_eta2')
pivot_eta = pivot_eta.reindex(index=TRAIT_ORDER, columns=PRED_LABELS)

pivot_sig = results_df.pivot(index='Trait', columns='Predictor', values='sig_bh')
pivot_sig = pivot_sig.reindex(index=TRAIT_ORDER, columns=PRED_LABELS)

vmax = float(pivot_eta.values.max())

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(pivot_eta.values.astype(float), cmap='YlOrRd',
               aspect='auto', vmin=0, vmax=vmax)

for i, trait in enumerate(TRAIT_ORDER):
    for j, pred in enumerate(PRED_LABELS):
        val = pivot_eta.iloc[i, j]
        sig = pivot_sig.iloc[i, j]
        if pd.isna(val):
            ax.text(j, i, '—', ha='center', va='center', fontsize=11, color='#aaa')
        else:
            star = '*' if sig else ''
            txt_color = 'white' if float(val) > vmax * 0.55 else '#333'
            ax.text(j, i, f"{float(val):.4f}{star}",
                    ha='center', va='center', fontsize=11, color=txt_color)

ax.set_xticks(range(len(PRED_LABELS)))
ax.set_xticklabels(PRED_LABELS, fontsize=12)
ax.set_yticks(range(len(TRAIT_ORDER)))
ax.set_yticklabels(TRAIT_ORDER, fontsize=12)
ax.set_title("Full Model: Partial η² per Predictor\n"
             "Trait ~ Gender + Age Group + Race + Prompt Type + Model\n"
             "(* = significant after BH correction, q < 0.05)",
             fontsize=13, fontweight='bold')

fig.colorbar(im, ax=ax, shrink=0.8, label='Partial η²')
fig.tight_layout()
out_fig = FIG_DIR / "fig6_fullmodel_partial_eta2.png"
fig.savefig(out_fig, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved {out_fig}")
