import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# --- PAGE CONFIG ---
st.set_page_config(page_title="Big Five: Human vs AI", layout="wide", page_icon="📊")

# --- STATISTICAL HELPERS ---
def calculate_stats(m1, s1, n1, m2, s2, n2):
    """Calculates Welch's T-Test and Cohen's d."""
    t_stat, p_val = stats.ttest_ind_from_stats(
        mean1=m1, std1=s1, nobs1=n1,
        mean2=m2, std2=s2, nobs2=n2,
        equal_var=False
    )
    pooled_var = (((n1 - 1) * s1**2) + ((n2 - 1) * s2**2)) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 0
    d = (m1 - m2) / pooled_std if pooled_std != 0 else 0
    return p_val, abs(d)

# --- SOURCE MAPPINGS ---
SOURCE_DISPLAY_MAP = {
    "Human": "🧑‍🤝‍🧑 Human",
    "AI_GPT4o_Explicit": "🤖 G4 Explicit",
    "AI_GPT4o_NLP": "🗣️ G4 NLP",
    "AI_Qwen_Explicit": "🧠 Qw Explicit",
    "AI_Qwen_NLP": "🗣️ Qw NLP",
    "AI_GPT4o_Order_AGN": "🔄 G4 (Age 1st)",
    "AI_GPT4o_Order_GAN": "🔄 G4 (Gen 1st)",
    "AI_GPT4o_Order_NAG": "🔄 G4 (Nat 1st)",
    "AI_Qwen_Order_AGN": "🔄 Qw (Age 1st)",
    "AI_Qwen_Order_GAN": "🔄 Qw (Gen 1st)",
    "AI_Qwen_Order_NAG": "🔄 Qw (Nat 1st)"
}

# --- NAVIGATION ---
page = st.sidebar.radio("Navigation", ["📊 Persona Analytics", "🏆 Key Discoveries", "📖 Relevant Literature"])

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dashboard_precalc_stats_singular.csv")
        for col in ['Country', 'Race', 'Age', 'Age_Group', 'Gender']:
            if col in df.columns: df[col] = df[col].astype(str)
        return df
    except FileNotFoundError:
        st.error("Missing data file. Please run trait_extraction.py first.")
        return pd.DataFrame()

df = load_data()

if page == "📊 Persona Analytics":
    st.title("🤖 LLM vs 🧑‍🤝‍🧑 Human: Persona Dashboard")
    
    if not df.empty:
        # --- SIDEBAR ---
        st.sidebar.header("Data Sources")
        available_sources = [s for s in SOURCE_DISPLAY_MAP.keys() if s in df['Source'].unique()]
        selected_sources = st.sidebar.multiselect("Models to Compare", options=available_sources, default=available_sources[:4], format_func=lambda x: SOURCE_DISPLAY_MAP[x])
        
        st.sidebar.markdown("---")
        def get_options(column):
            opts = sorted([x for x in df[column].unique() if x != 'All'])
            return ['All'] + opts
        
        def build_profile_sidebar(letter):
            st.sidebar.subheader(f"Profile {letter}")
            c = st.sidebar.selectbox(f"Country", get_options('Country'), key=f'c_{letter}')
            r = st.sidebar.selectbox(f"Race", get_options('Race'), key=f'r_{letter}')
            at = st.sidebar.radio(f"Age Logic", ["Range", "Specific Year"], key=f'at_{letter}')
            ag = st.sidebar.selectbox(f"Select Range", get_options('Age_Group'), key=f'ag_{letter}') if at == "Range" else "All"
            a = st.sidebar.selectbox(f"Select Year", get_options('Age'), key=f'a_{letter}') if at != "Range" else "All"
            g = st.sidebar.selectbox(f"Gender", get_options('Gender'), key=f'g_{letter}')
            return c, r, a, ag, g

        a_vals, b_vals = build_profile_sidebar("A"), build_profile_sidebar("B")
        trait_map = {"Extroversion": "E", "Agreeableness": "A", "Conscientiousness": "C", "Emotional Stability": "E", "Openness": "O"}
        sel_trait = st.sidebar.selectbox("Select Trait for Analysis", list(trait_map.keys()))
        t_pref = trait_map[sel_trait]

        data_a = df[(df['Country']==a_vals[0]) & (df['Race']==a_vals[1]) & (df['Age']==a_vals[2]) & (df['Age_Group']==a_vals[3]) & (df['Gender']==a_vals[4])]
        data_b = df[(df['Country']==b_vals[0]) & (df['Race']==b_vals[1]) & (df['Age']==b_vals[2]) & (df['Age_Group']==b_vals[3]) & (df['Gender']==b_vals[4])]

        st.markdown("---")
        
        # ==========================================
        # SECTION 1: INTER-PROFILE COMPARISON (A vs B, Same Model)
        # ==========================================
        st.header(f"⚖️ Section 1: Inter-Profile Comparison (Group A vs Group B)")
        st.markdown(f"Does the **same model** perceive a significant difference between these two demographic groups for **{sel_trait}**?")
        
        valid_both = [s for s in selected_sources if not data_a[data_a['Source'] == s].empty and not data_b[data_b['Source'] == s].empty]
        
        if valid_both:
            inter_results = []
            for src in valid_both:
                r1 = data_a[data_a['Source'] == src].iloc[0]
                r2 = data_b[data_b['Source'] == src].iloc[0]
                n1, n2 = int(r1[f"{t_pref}_Trait_count"]), int(r2[f"{t_pref}_Trait_count"])
                p, d = calculate_stats(r1[f"{t_pref}_Trait_mean"], r1[f"{t_pref}_Trait_std"], n1, 
                                       r2[f"{t_pref}_Trait_mean"], r2[f"{t_pref}_Trait_std"], n2)
                inter_results.append({
                    "Model": SOURCE_DISPLAY_MAP[src],
                    "n (A)": n1,
                    "n (B)": n2,
                    "P-Value": round(p, 4),
                    "Cohen's d": round(d, 2),
                    "Significant": "✅ Yes" if p < 0.05 else "❌ No"
                })
            st.table(pd.DataFrame(inter_results))
        else:
            st.warning("Select models that have data for both profiles.")

        # ==========================================
        # SECTION 2 & 3: INTRA-PROFILE METHODOLOGY GRIDS
        # ==========================================
        def render_intra_grid(data_subset, letter):
            st.header(f"🔍 Section {'2' if letter == 'A' else '3'}: Intra-Profile Methodology Grid (Group {letter})")
            st.markdown(f"How much do the different models and prompts disagree when describing **Group {letter}**?")
            
            valid_subset = [s for s in selected_sources if not data_subset[data_subset['Source'] == s].empty]
            if len(valid_subset) > 1:
                # Build names with (n=X)
                names = []
                for s in valid_subset:
                    n = int(data_subset[data_subset['Source'] == s].iloc[0][f"{t_pref}_Trait_count"])
                    names.append(f"{SOURCE_DISPLAY_MAP[s]} (n={n})")

                p_mat = pd.DataFrame(index=names, columns=names, dtype=float)
                d_mat = pd.DataFrame(index=names, columns=names, dtype=float)
                
                for i, s1 in enumerate(valid_subset):
                    r1 = data_subset[data_subset['Source'] == s1].iloc[0]
                    for j, s2 in enumerate(valid_subset):
                        r2 = data_subset[data_subset['Source'] == s2].iloc[0]
                        p, d = calculate_stats(r1[f"{t_pref}_Trait_mean"], r1[f"{t_pref}_Trait_std"], r1[f"{t_pref}_Trait_count"], 
                                               r2[f"{t_pref}_Trait_mean"], r2[f"{t_pref}_Trait_std"], r2[f"{t_pref}_Trait_count"])
                        p_mat.iloc[i, j], d_mat.iloc[i, j] = p, d

                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(px.imshow(p_mat, text_auto=".3f", color_continuous_scale="Blues_r", title=f"P-Values (Group {letter})"), use_container_width=True)
                with c2:
                    st.plotly_chart(px.imshow(d_mat, text_auto=".2f", color_continuous_scale="Reds", title=f"Effect Size (Group {letter})"), use_container_width=True)
            else:
                st.info(f"Select more models to see the disagreement grid for Group {letter}.")

        render_intra_grid(data_a, "A")
        st.markdown("---")
        render_intra_grid(data_b, "B")

        bar_data = []
        for d, n in zip([data_a, data_b], ["Group A", "Group B"]):
            for src in selected_sources:
                row = d[d['Source'] == src]
                if not row.empty:
                    bar_data.append({"Group": n, "Source": SOURCE_DISPLAY_MAP[src], "Mean": row.iloc[0][f"{t_pref}_Trait_mean"], "StdDev": row.iloc[0][f"{t_pref}_Trait_std"]})
        if bar_data: st.plotly_chart(px.bar(pd.DataFrame(bar_data), x='Group', y='Mean', color='Source', barmode='group', error_y='StdDev', text_auto='.2f', range_y=[1,5]), use_container_width=True)
# ==========================================
# PAGE 2 & 3: KEY DISCOVERIES / LITERATURE
# ==========================================
elif page == "🏆 Key Discoveries":
    st.title("🏆 Key Discoveries & Statistical Extremes")
    st.markdown("A macro-level view of the largest deviations and biases discovered in the dataset.")
    
    st.header("🚨 Top 20 AI Hallucinations (Explicit AI vs Human)")
    st.markdown("Demographics where the LLM's simulated personality diverged the most from actual human survey data.")
    try:
        ai_bias_df = pd.read_csv("top_20_ai_biases.csv")
        st.dataframe(ai_bias_df, use_container_width=True)
    except:
        st.info("Run the AI Bias script to generate 'top_20_ai_biases.csv'")
        
    st.divider()
    
    st.header("⚔️ Top 50 Human-vs-Human Personality Clashes")
    st.markdown("The largest recorded personality gaps between two distinct human demographic groups.")
    try:
        clash_df = pd.read_csv("top_50_human_clashes.csv")
        st.dataframe(clash_df, use_container_width=True)
    except:
        st.info("Run the Pairwise Clash script to generate 'top_50_human_clashes.csv'")


elif page == "📖 Relevant Literature":
    st.title("📖 Relevant Research & Papers")
    st.markdown("Essential reading on LLM personality bias and psychometric evaluation.")

    papers = [
        {"title": "Manipulating the Perceived Personality Traits of Language Models", "authors": "Caron & Srivastava (2022)", "link": "https://arxiv.org/abs/2212.10276"},
    ]

    for paper in papers:
        with st.expander(f"📄 {paper['title']}"):
            st.write(f"**Authors:** {paper['authors']}")
            st.link_button("Read Paper", paper["link"])
