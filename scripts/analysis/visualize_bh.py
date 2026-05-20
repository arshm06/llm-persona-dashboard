"""
BH significance results — slide-ready figures.

Generates 3 figures:
  1. BH survival rate per source (variance collapse)
  2. Cohen's d distribution: Human vs GPT Explicit vs GPT NLP
  3. Human BH survivors: heatmap of trait × demographic factor

Usage:
    python scripts/analysis/visualize_bh.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
FIG_DIR = ROOT / "output/figures"
FIG_DIR.mkdir(exist_ok=True)

PALETTE = {
    "Human":            "#2C7BB6",
    "AI_GPT4o_Explicit":"#D7191C",
    "AI_GPT4o_NLP":     "#E87D2B",
    "AI_GPT4o_Order_AGN":"#C8586C",
    "AI_GPT4o_Order_GAN":"#C8586C",
    "AI_GPT4o_Order_NAG":"#C8586C",
    "AI_Qwen_NLP":      "#888888",
    "AI_Qwen_Order_AGN":"#AAAAAA",
    "AI_Qwen_Order_GAN":"#AAAAAA",
    "AI_Qwen_Order_NAG":"#AAAAAA",
}
SOURCE_LABELS = {
    "Human":            "Human",
    "AI_GPT4o_Explicit":"GPT Explicit",
    "AI_GPT4o_NLP":     "GPT NLP",
    "AI_GPT4o_Order_AGN":"GPT Order (AGN)",
    "AI_GPT4o_Order_GAN":"GPT Order (GAN)",
    "AI_GPT4o_Order_NAG":"GPT Order (NAG)",
    "AI_Qwen_NLP":      "Qwen NLP",
    "AI_Qwen_Order_AGN":"Qwen Order (AGN)",
    "AI_Qwen_Order_GAN":"Qwen Order (GAN)",
    "AI_Qwen_Order_NAG":"Qwen Order (NAG)",
}
TRAIT_ORDER = ["Extroversion", "Agreeableness", "Conscientiousness", "Emotional Stability", "Openness"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Load data ──────────────────────────────────────────────────────────────
df      = pd.read_csv(ROOT / "output/dashboard/bh_results_all.csv")
df_bh   = pd.read_csv(ROOT / "output/dashboard/bh_results_significant.csv")
df_bon  = pd.read_csv(ROOT / "output/dashboard/bonferroni_results_significant.csv")
print(f"Loaded {len(df)} tests | BH significant: {len(df_bh)} | Bonferroni significant: {len(df_bon)}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — BH vs Bonferroni Survival Rate (side-by-side)
# ══════════════════════════════════════════════════════════════════════════════
summary = (df.groupby("Source")
             .agg(total      =("P_Raw", "count"),
                  surv_bh    =("Significant_BH", "sum"),
                  surv_bon   =("Significant_Bonferroni", "sum"))
             .reset_index())
summary["pct_bh"]  = summary["surv_bh"]  / summary["total"] * 100
summary["pct_bon"] = summary["surv_bon"] / summary["total"] * 100
summary = summary.sort_values("pct_bh", ascending=True)
summary["label"] = summary["Source"].map(SOURCE_LABELS)
summary["color"] = summary["Source"].map(PALETTE)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

for ax, col, title, col_key in [
    (axes[0], "pct_bh",  "BH (FDR, q < 0.05)",       "surv_bh"),
    (axes[1], "pct_bon", "Bonferroni (FWER, p < 0.05)", "surv_bon"),
]:
    bars = ax.barh(summary["label"], summary[col],
                   color=summary["color"], edgecolor="white", height=0.6)
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{row[col]:.1f}%  ({int(row[col_key])}/{int(row['total'])})",
                va="center", fontsize=11, color="#333333")
    human_pct = summary.loc[summary["Source"] == "Human", col].values[0]
    ax.axvline(human_pct, color="#2C7BB6", linestyle="--", linewidth=1.4, alpha=0.6)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Tests surviving correction (%)", labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold")

fig.suptitle("Multiple Testing Correction: Survival Rate by Source\n"
             "(dashed = Human baseline)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
out1 = FIG_DIR / "fig1_survival_rate_bh_vs_bonferroni.png"
fig.savefig(out1, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Cohen's d Distribution: Human vs GPT Explicit vs GPT NLP
# ══════════════════════════════════════════════════════════════════════════════
focus_sources = ["Human", "AI_GPT4o_Explicit", "AI_GPT4o_NLP"]
focus_labels  = ["Human", "GPT Explicit", "GPT NLP"]
focus_colors  = [PALETTE[s] for s in focus_sources]
focus_df = df[df["Source"].isin(focus_sources)].copy()
focus_df["Source_Label"] = focus_df["Source"].map(SOURCE_LABELS)

fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=True)

for ax, trait in zip(axes, TRAIT_ORDER):
    tdata = focus_df[focus_df["Trait"] == trait]
    positions = [1, 2, 3]
    data_by_source = [tdata[tdata["Source"]==s]["Cohens_d"].dropna().values for s in focus_sources]

    vp = ax.violinplot(data_by_source, positions=positions, showmedians=True,
                       showextrema=False, widths=0.7)
    for body, color in zip(vp["bodies"], focus_colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
    vp["cmedians"].set_color("white")
    vp["cmedians"].set_linewidth(2)

    # Add jittered dots
    for pos, vals, color in zip(positions, data_by_source, focus_colors):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(pos + jitter, vals, color=color, alpha=0.25, s=8, linewidths=0)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels(focus_labels, fontsize=10, rotation=20, ha="right")
    ax.set_title(trait, fontsize=12)
    if ax == axes[0]:
        ax.set_ylabel("Cohen's d  (vs Human global baseline)", labelpad=8)

legend_patches = [mpatches.Patch(color=PALETTE[s], label=l)
                  for s, l in zip(focus_sources, focus_labels)]
fig.legend(handles=legend_patches, loc="upper right", frameon=False, fontsize=11)
fig.suptitle("Effect Size Distribution (Cohen's d) per Trait\n"
             "Human vs GPT Explicit vs GPT NLP — single-factor subgroups",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
out2 = FIG_DIR / "fig2_cohens_d_distribution.png"
fig.savefig(out2, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Human Survivors: BH vs Bonferroni Heatmaps (Trait × Factor)
# ══════════════════════════════════════════════════════════════════════════════
factor_order = ["Country", "Race", "Age_Group", "Age", "Gender"]

def build_pivot(sig_df, source="Human"):
    h = sig_df[sig_df["Source"] == source].copy()
    h["Factor"] = h["Subgroup"].str.split("=").str[0]
    counts = (h.groupby(["Trait","Factor"])
               .agg(n=("Cohens_d","count"), mean_d=("Cohens_d", lambda x: x.abs().mean()))
               .reset_index())
    pn = counts.pivot(index="Trait", columns="Factor", values="n").reindex(
        index=TRAIT_ORDER, columns=factor_order, fill_value=0).fillna(0).astype(int)
    pd_ = counts.pivot(index="Trait", columns="Factor", values="mean_d").reindex(
        index=TRAIT_ORDER, columns=factor_order, fill_value=0).fillna(0)
    return pn, pd_

pn_bh,  pd_bh  = build_pivot(df_bh)
pn_bon, pd_bon = build_pivot(df_bon)

vmax = max(pd_bh.values.max(), pd_bon.values.max())

fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))

for ax, pn, pd_, title in [
    (axes[0], pn_bh,  pd_bh,  "BH (FDR, q < 0.05)"),
    (axes[1], pn_bon, pd_bon, "Bonferroni (FWER, p < 0.05)"),
]:
    im = ax.imshow(pd_.values, cmap="Blues", aspect="auto", vmin=0, vmax=vmax)
    for i in range(len(TRAIT_ORDER)):
        for j in range(len(factor_order)):
            n = pn.values[i, j]
            d = pd_.values[i, j]
            txt = f"n={n}\nd={d:.2f}" if n > 0 else "—"
            color = "white" if d > vmax * 0.55 else "#333333"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10.5, color=color)
    ax.set_xticks(range(len(factor_order)))
    ax.set_xticklabels(factor_order, fontsize=12)
    ax.set_yticks(range(len(TRAIT_ORDER)))
    ax.set_yticklabels(TRAIT_ORDER, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Mean |Cohen's d|")

fig.suptitle("Human: Surviving Subgroup Effects by Correction Method\n"
             "(n = subgroups surviving; d = mean |Cohen's d|)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
out3 = FIG_DIR / "fig3_human_heatmap_bh_vs_bonferroni.png"
fig.savefig(out3, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out3}")

print("\nAll figures saved to output/figures/")
