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
    """Calculates Welch's T-Test and Cohen's d Effect Size."""
    # Welch's T-Test (does not assume equal variance)
    t_stat, p_val = stats.ttest_ind_from_stats(
        mean1=m1, std1=s1, nobs1=n1,
        mean2=m2, std2=s2, nobs2=n2,
        equal_var=False
    )
    
    # Cohen's d (Effect Size)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (m1 - m2) / pooled_std if pooled_std != 0 else 0
    
    # Interpretation
    if p_val < 0.05:
        sig_text = "Statistically Significant"
        if abs(d) < 0.2: d_text = "Negligible Effect"
        elif abs(d) < 0.5: d_text = "Small Effect"
        elif abs(d) < 0.8: d_text = "Medium Effect"
        else: d_text = "Large Effect"
    else:
        sig_text = "Not Significant"
        d_text = "N/A"
        
    return p_val, abs(d), sig_text, d_text

# --- NAVIGATION ---
page = st.sidebar.radio("Navigation", ["📊 Persona Analytics", "📖 Relevant Literature"])

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dashboard_precalc_stats.csv")
        demo_cols = ['Country', 'Race', 'Age', 'Age_Group', 'Gender']
        for col in demo_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df
    except FileNotFoundError:
        st.error("Missing 'dashboard_precalc_stats.csv'. Please run your aggregation script first.")
        return pd.DataFrame()

df = load_data()

# ==========================================
# PAGE 1: ANALYTICS
# ==========================================
if page == "📊 Persona Analytics":
    st.title("🤖 LLM vs 🧑‍🤝‍🧑 Human: Personality Persona Dashboard")
    
    if not df.empty:
        def get_options(column):
            opts = sorted([x for x in df[column].unique() if x != 'All'])
            return ['All'] + opts

        # --- SIDEBAR FILTERS ---
        st.sidebar.header("Filter Settings")
        
        def build_profile_sidebar(letter):
            st.sidebar.subheader(f"Profile {letter}")
            country = st.sidebar.selectbox(f"Country", get_options('Country'), key=f'c_{letter}')
            race = st.sidebar.selectbox(f"Race", get_options('Race'), key=f'r_{letter}')
            age_type = st.sidebar.radio(f"Age Logic", ["Range", "Specific Year"], key=f'at_{letter}')
            if age_type == "Range":
                age_group = st.sidebar.selectbox(f"Select Range", get_options('Age_Group'), key=f'ag_{letter}')
                age = "All"
            else:
                age = st.sidebar.selectbox(f"Select Year", get_options('Age'), key=f'a_{letter}')
                age_group = "All"
            gender = st.sidebar.selectbox(f"Gender", get_options('Gender'), key=f'g_{letter}')
            return country, race, age, age_group, gender

        a_vals = build_profile_sidebar("A")
        st.sidebar.markdown("---")
        b_vals = build_profile_sidebar("B")

        trait_map = {"Extroversion": "E", "Agreeableness": "A", "Conscientiousness": "C", "Neuroticism": "N", "Openness": "O"}
        sel_trait = st.sidebar.selectbox("Select Trait for Deep Dive", list(trait_map.keys()))
        t_pref = trait_map[sel_trait]

        # --- EXTRACT DATA ---
        def get_data(c, r, a, ag, g):
            return df[(df['Country']==c) & (df['Race']==r) & (df['Age']==a) & (df['Age_Group']==ag) & (df['Gender']==g)]

        data_a = get_data(*a_vals)
        data_b = get_data(*b_vals)

        if data_a.empty or data_b.empty:
            st.warning("Combination not found. Try resetting some filters to 'All'.")
            st.stop()

        # --- STATS SUMMARY ---
        st.header("📈 Statistical Significance")
        h_a = data_a[data_a['Source'] == 'Human'].iloc[0]
        h_b = data_b[data_b['Source'] == 'Human'].iloc[0]
        ai_a = data_a[data_a['Source'] == 'AI_Simulated'].iloc[0]

        p_h, d_h, txt_h, dt_h = calculate_stats(
            h_a[f"{t_pref}_Trait_mean"], h_a[f"{t_pref}_Trait_std"], h_a[f"{t_pref}_Trait_count"],
            h_b[f"{t_pref}_Trait_mean"], h_b[f"{t_pref}_Trait_std"], h_b[f"{t_pref}_Trait_count"]
        )
        
        p_ai, d_ai, txt_ai, dt_ai = calculate_stats(
            h_a[f"{t_pref}_Trait_mean"], h_a[f"{t_pref}_Trait_std"], h_a[f"{t_pref}_Trait_count"],
            ai_a[f"{t_pref}_Trait_mean"], ai_a[f"{t_pref}_Trait_std"], ai_a[f"{t_pref}_Trait_count"]
        )

        ai_b = data_b[data_b['Source'] == 'AI_Simulated'].iloc[0]

        # Calculate significance for AI Group A vs AI Group B
        p_ai_diff, d_ai_diff, txt_ai_diff, dt_ai_diff = calculate_stats(
            ai_a[f"{t_pref}_Trait_mean"], ai_a[f"{t_pref}_Trait_std"], ai_a[f"{t_pref}_Trait_count"],
            ai_b[f"{t_pref}_Trait_mean"], ai_b[f"{t_pref}_Trait_std"], ai_b[f"{t_pref}_Trait_count"]
        )

        # Update the columns to a 3-column layout
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Human A vs Human B")
            st.metric("P-Value", f"{p_h:.4f}")
            st.write(f"**Result:** {txt_h}\n\n**Effect:** {dt_h}")

        with c2:
            st.subheader("Human A vs AI A")
            st.metric("P-Value", f"{p_ai:.4f}")
            st.write(f"**Result:** {txt_ai}\n\n**Effect:** {dt_ai}")

        with c3:
            st.subheader("AI A vs AI B")
            st.metric("P-Value", f"{p_ai_diff:.4f}")
            st.write(f"**Result:** {txt_ai_diff}\n\n**Effect:** {dt_ai_diff}")
      
        # Deep Dive Bar
        bar_data = []
        for d, n in zip([data_a, data_b], ["Group A", "Group B"]):
            for s in ['Human', 'AI_Simulated']:
                row = d[d['Source'] == s]
                if not row.empty:
                    bar_data.append({"Group": n, "Source": s, "Mean": row.iloc[0][f"{t_pref}_Trait_mean"], "StdDev": row.iloc[0][f"{t_pref}_Trait_std"]})
        
        fig_bar = px.bar(pd.DataFrame(bar_data), x='Group', y='Mean', color='Source', barmode='group', error_y='StdDev', text_auto='.2f', range_y=[1,5], title=f"Comparison: {sel_trait}")
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# PAGE 2: LITERATURE
# ==========================================
elif page == "📖 Relevant Literature":
    st.title("📖 Relevant Research & Papers")
    st.markdown("Scientific foundations for LLM psychometric evaluation.")
    
    papers = [
        {"title": "The Silicon Sample: Using LLMs to Simulate Human Surveys", "authors": "Argyle et al. (2023)", "link": "https://doi.org/10.1017/psrm.2023.18", "summary": "A study on the alignment between LLM 'silicon' sub-populations and actual human survey data."},
        {"title": "Personality Traits in Large Language Models", "authors": "Caron & Srivastava (2022)", "link": "https://arxiv.org/abs/2207.00233", "summary": "An exploration into whether LLMs exhibit stable personality structures."},
        {"title": "On the Opportunities and Risks of Foundation Models", "authors": "Bommasani et al. (2021)", "link": "https://arxiv.org/abs/2108.07258", "summary": "A comprehensive review of the social biases inherent in large-scale foundation models."}
    ]

    for p in papers:
        with st.expander(f"📄 {p['title']}"):
            st.write(f"**Authors:** {p['authors']}\n\n**Summary:** {p['summary']}")
            st.link_button("Read Full Paper", p["link"])