"""
Generate a PDF report for professor discussion.

Usage:
    python scripts/generate_report.py
Output:
    output/report_llm_persona_analysis.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    HRFlowable, ListFlowable, ListItem,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import pandas as pd
from datetime import date

ROOT  = Path(__file__).parent.parent
FIGS  = ROOT / "output/figures"
DATA  = ROOT / "output/dashboard"
OUT   = ROOT / "output/report_llm_persona_analysis.pdf"

W, H  = A4
MARGIN = 2.0 * cm
CONTENT_W = W - 2 * MARGIN   # usable width in points

# ── Styles ───────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()
DARK = colors.HexColor("#1a1a2e")
BLUE = colors.HexColor("#2c5f8a")
GREY = colors.HexColor("#555555")

TITLE_S = ParagraphStyle("ts", parent=base["Title"],    fontSize=22, textColor=DARK,
                          spaceAfter=4, alignment=TA_CENTER)
SUB_S   = ParagraphStyle("ss", parent=base["Normal"],   fontSize=11, textColor=GREY,
                          spaceAfter=16, alignment=TA_CENTER)
H1_S    = ParagraphStyle("h1", parent=base["Heading1"], fontSize=13, textColor=DARK,
                          spaceBefore=14, spaceAfter=4)
CAP_S   = ParagraphStyle("cp", parent=base["Normal"],   fontSize=8,  textColor=GREY,
                          spaceAfter=6, alignment=TA_CENTER)
B_S     = ParagraphStyle("bs", parent=base["Normal"],   fontSize=9,  textColor=DARK,
                          leading=13, leftIndent=12, spaceAfter=2)


def rule():
    return HRFlowable(width="100%", thickness=0.8, color=BLUE, spaceAfter=6, spaceBefore=2)


MAX_FIG_H = 580   # max figure height in points — leaves room for caption + bullets

def fig(name, caption=None, width_frac=1.0):
    """Insert a figure, auto-computing height and capping to fit within one page."""
    p = FIGS / name
    if not p.exists():
        return [Paragraph(f"[missing: {name}]", CAP_S)]
    try:
        ir     = ImageReader(str(p))
        iw, ih = ir.getSize()
        w      = CONTENT_W * width_frac
        h      = w * (ih / iw)
        if h > MAX_FIG_H:
            h = MAX_FIG_H
            w = h * (iw / ih)
        elems = [Image(str(p), width=w, height=h)]
    except Exception:
        elems = [Paragraph(f"[error loading: {name}]", CAP_S)]
    if caption:
        elems.append(Paragraph(caption, CAP_S))
    return elems


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(t, B_S), leftIndent=8, bulletColor=BLUE) for t in items],
        bulletType="bullet", leftIndent=12, spaceBefore=4, spaceAfter=8,
    )


def h1(text):
    return [Paragraph(text, H1_S), rule()]


# ── Load stats ───────────────────────────────────────────────────────────────
bh     = pd.read_csv(DATA / "bh_results_all.csv")
am     = pd.read_csv(DATA / "anova_fullmodel.csv")
al     = pd.read_csv(DATA / "alignment_correlations.csv")
vc     = pd.read_csv(DATA / "variance_collapse.csv")
pc     = pd.read_csv(DATA / "prompt_comparison_summary.csv")

def bh_pct(src):
    s = bh[bh["Source"] == src]
    return 100 * s["Significant_BH"].mean()

vc_mean = vc[vc["Source"].isin(["AI_GPT4o_Explicit","AI_GPT4o_NLP"])]\
            .groupby("Source")["Std_Ratio"].mean()
al_mean = al[al["Source"].isin(["AI_GPT4o_Explicit","AI_GPT4o_NLP"])]\
            .groupby("Source")["Pearson_r_means"].mean()
nlp_closer = 100 * pc["GPT_NLP_closer_than_Explicit"].mean()
ex_exagg   = 100 * pc["GPT_Explicit_exaggerates"].mean()
nl_exagg   = 100 * pc["GPT_NLP_exaggerates"].mean()


# ── Build ─────────────────────────────────────────────────────────────────────
doc   = SimpleDocTemplate(str(OUT), pagesize=A4,
                           leftMargin=MARGIN, rightMargin=MARGIN,
                           topMargin=MARGIN, bottomMargin=MARGIN)
story = []

# ── cover ────────────────────────────────────────────────────────────────────
story += [
    Spacer(1, 2.5*cm),
    Paragraph("llm personality bias — analysis report", TITLE_S),
    Paragraph(f"professor discussion · {date.today().strftime('%b %d, %Y')}", SUB_S),
    rule(),
    Spacer(1, 0.5*cm),
    bullets([
        "19,720 human ipip-50 responses as baseline (age, gender, race, country)",
        "gpt-4o simulated in two conditions: explicit prompting and nl (backstory) prompting",
        "five big-five traits: extroversion, agreeableness, conscientiousness, emotional stability, openness",
        "pipeline: bh/bonferroni correction → factor-level anova → explicit vs nlp comparison → alignment metrics → variance collapse",
        "qwen nlp excluded from primary results (n=1,200 vs ~20k for gpt) — stronger models planned",
    ]),
    PageBreak(),
]

# ── 1. methodology ───────────────────────────────────────────────────────────
story += h1("1. methodology") + [
    bullets([
        "human data: ipip-50 questionnaire, 1–5 likert scale, 10 items per trait, reverse-scored where needed",
        "explicit condition: model given demographic attributes directly and asked to fill in ipip items as that person",
        "nlp condition: demographic profile converted to a natural-language backstory; model responds in character",
        "trait score = mean of 10 scored items per trait (emotional stability = reversed neuroticism)",
        "subgroup stats (mean, std, n) computed for each single-factor demographic slice (e.g. gender=female)",
        "significance: welch t-test vs global human baseline, corrected with benjamini–hochberg (q<0.05) and bonferroni",
        "anova: one-way per factor×source, two-way for factor pairs, full model: trait ~ gender + age + race + prompt type + model",
        "alignment: pearson r between human subgroup mean vector and model subgroup mean vector per factor×trait",
        "variance collapse: ratio of within-subgroup σ (model ÷ human) per subgroup × trait",
    ]),
    PageBreak(),
]

# ── 2. bh correction ─────────────────────────────────────────────────────────
story += h1("2. multiple hypothesis testing (bh + bonferroni)") + \
    fig("fig1_survival_rate_bh_vs_bonferroni.png",
        "bh and bonferroni survival rates per source. dashed line = human baseline.") + [
    bullets([
        f"human: {bh_pct('Human'):.0f}% of subgroup tests survive bh correction",
        f"gpt explicit: {bh_pct('AI_GPT4o_Explicit'):.0f}% survive — nearly all subgroup effects are statistically robust",
        f"gpt nlp: {bh_pct('AI_GPT4o_NLP'):.0f}% survive",
        "high survival rate for gpt does not mean alignment with humans — it means the model is consistently wrong in the same direction",
        "bonferroni is more conservative but the ordering across sources is identical",
    ]),
    PageBreak(),
]

# ── 3. anova ─────────────────────────────────────────────────────────────────
story += h1("3. factor-level anova") + \
    fig("fig6_fullmodel_partial_eta2.png",
        "full model partial η² per predictor × trait. * = significant after bh.") + [
    bullets([
        "full model: trait ~ gender + age group + race + prompt type + model",
        "prompt type and model are among the strongest predictors — in several traits they explain more variance than any demographic factor",
        "agreeableness most affected: η²=0.120 for model, η²=0.098 for prompt type",
        "conscientiousness shows the strongest age group effect (η²=0.112) — consistent across human and gpt",
        "race and gender effects are smaller but significant across most traits",
    ]),
    Spacer(1, 0.3*cm),
] + fig("fig4_oneway_eta2.png",
        "one-way anova η² per factor × source × trait.") + [
    bullets([
        "age group has the most consistent effect across human, gpt explicit, and gpt nlp",
        "gpt nlp generally produces larger factor effects than humans for most traits",
        "country effect is weakest for both humans and models — noisy, limited shared subgroups",
    ]),
    Spacer(1, 0.3*cm),
] + fig("fig5_source_interaction.png",
        "source × factor interaction: does the demographic effect differ across human/gpt explicit/gpt nlp?") + [
    bullets([
        "significant source × factor interactions found for most trait × factor combinations",
        "models and humans agree on the direction of effects more than on magnitude",
    ]),
    PageBreak(),
]

# ── 4. explicit vs nlp ───────────────────────────────────────────────────────
story += h1("4. explicit vs. nl prompting") + \
    fig("fig_prompt_comparison_gender.png",
        "effect sizes by prompting method — gender. bars = subgroup mean − global mean.") + [
    bullets([
        "both gpt conditions preserve the direction of human gender effects on agreeableness and conscientiousness",
        "gpt explicit systematically compresses extroversion differences; gpt nlp slightly exaggerates them",
        "emotional stability shows the largest divergence between prompting methods",
    ]),
    PageBreak(),
] + fig("fig_prompt_comparison_age_group.png",
        "effect sizes — age group.") + [
    bullets([
        "conscientiousness age gradient (rising with age) is captured by both conditions",
        "gpt nlp better tracks the human openness decline in older age groups",
        "emotional stability diverges substantially — both models show older = more stable, but the effect size is larger than in humans",
    ]),
    PageBreak(),
] + fig("fig_prompt_comparison_race.png",
        "effect sizes — race.") + [
    bullets([
        "race effects are most inconsistent — neither model reliably tracks the human pattern",
        "gpt explicit exaggerates several race effects that are near-zero in humans",
        "gpt nlp shows some improvement but still diverges for openness and conscientiousness",
    ]),
    PageBreak(),
] + fig("fig_prompt_comparison_country.png",
        "effect sizes — country (top 15 by human n).") + [
    bullets([
        "country-level effects are the weakest signal in both human and model data",
        "models do not reliably reproduce the small but real cross-national differences in human data",
        f"overall: gpt nlp is closer to human patterns in {nlp_closer:.0f}% of subgroups vs gpt explicit",
        f"gpt explicit exaggerates demographic effects in {ex_exagg:.0f}% of subgroups; gpt nlp in {nl_exagg:.0f}%",
        "neither prompting method consistently aligns with humans — nlp is marginally better",
    ]),
    PageBreak(),
]

# ── 5. alignment ─────────────────────────────────────────────────────────────
story += h1("5. alignment with human patterns (pearson r on subgroup means)") + \
    fig("fig_alignment_heatmap.png",
        "pearson r between model and human subgroup mean vectors. n = shared subgroups.") + [
    bullets([
        f"gpt explicit: mean r = {al_mean.get('AI_GPT4o_Explicit', 0):.3f} across all factor×trait combinations",
        f"gpt nlp: mean r = {al_mean.get('AI_GPT4o_NLP', 0):.3f}",
        "best alignment: gpt nlp × gender (r=0.84) and gpt explicit × age group (r=0.80)",
        "weakest: both models × country (r≈0.18–0.19) and race (r≈0.34–0.39)",
        "caveat: gender correlation uses only 3 subgroups — confidence intervals are wide",
    ]),
    Spacer(1, 0.3*cm),
] + fig("fig_alignment_effects_heatmap.png",
        "same but computed on effect vectors (subgroup mean − global mean) rather than raw means.") + [
    bullets([
        "effect-vector correlations are identical to mean correlations — as expected since subtracting a constant doesn't change r",
        "confirms models capture ordinal ranking of groups better than absolute magnitudes",
    ]),
    PageBreak(),
]

# ── 6. effect distribution ───────────────────────────────────────────────────
story += h1("6. effect size distribution") + \
    fig("fig_effect_distribution_overview.png",
        "violin plots of cohen's d (vs human global baseline) per factor × trait × source.") + [
    bullets([
        "human cohen's d distribution is narrow and centered near 0 — most subgroups don't differ much from the global mean",
        "gpt explicit and nlp both show wider tails — more extreme subgroup deviations",
        "gpt nlp consistently produces larger median effect sizes than explicit across most traits",
        "race and country factors show the widest spread for gpt conditions",
    ]),
    PageBreak(),
]

# ── 7. variance collapse ──────────────────────────────────────────────────────
story += h1("7. variance collapse") + \
    fig("fig_variance_collapse.png",
        "mean std ratio (model within-subgroup σ / human within-subgroup σ). < 1 = variance collapse.") + [
    bullets([
        f"gpt explicit: mean std ratio = {vc_mean.get('AI_GPT4o_Explicit', 0):.3f} — model diversity is ~17% of human diversity",
        f"gpt nlp: mean std ratio = {vc_mean.get('AI_GPT4o_NLP', 0):.3f} — slightly better but still severe collapse",
        "verified directly: gpt explicit female extroversion has only 15 unique values (range 2.3–3.7) vs 41 human values (range 1.0–5.0)",
        "models output near-identical scores for every individual with the same demographic profile — prototype stereotyping at the response level",
        "this is consistent across all traits and both prompting conditions",
    ]),
    Spacer(1, 0.3*cm),
] + fig("fig_variance_collapse_by_factor.png",
        "std ratio broken down by demographic factor. collapse is uniform across all factors.") + [
    bullets([
        "no factor escapes the collapse — gender, age, race, and country all show the same pattern",
        "gpt nlp slightly higher ratios than explicit (0.24 vs 0.17) but the difference is small",
        "important implication: llm personality simulations should not be used to model individual-level variation",
    ]),
    PageBreak(),
]

# ── 8. caveats + next steps ───────────────────────────────────────────────────
story += h1("8. caveats & next steps") + [
    bullets([
        "qwen nlp results excluded — n=1,200 is too small for reliable subgroup-level comparisons",
        "no qwen explicit condition yet — explicit vs nlp comparison is gpt-4o only",
        "gender alignment correlation (r=0.84) is based on 3 subgroups — statistically unreliable",
        "country alignment is weak partly due to limited shared subgroups (n≥30 threshold)",
        "planned: deepseek-r1-distill-qwen-7b and deepseek-r1-distill-llama-8b (explicit + nlp conditions)",
        "planned: literature-backed hypothesis testing (3–5 claims from prior work) — colleague handling",
    ]),
]

doc.build(story)
print(f"saved {OUT}")
