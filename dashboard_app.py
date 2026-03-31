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
    t_stat, p_val = stats.ttest_ind_from_stats(
        mean1=m1, std1=s1, nobs1=n1,
        mean2=m2, std2=s2, nobs2=n2,
        equal_var=False
    )
    
    pooled_var = (((n1 - 1) * s1**2) + ((n2 - 1) * s2**2)) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 0
    d = (m1 - m2) / pooled_std if pooled_std != 0 else 0
    
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
page = st.sidebar.radio("Navigation", ["📊 Persona Analytics", "🏆 Key Discoveries", "📖 Relevant Literature"])

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dashboard_precalc_stats_all.csv")
        demo_cols = ['Country', 'Race', 'Age', 'Age_Group', 'Gender']
        for col in demo_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df
    except FileNotFoundError:
        st.error("Missing 'dashboard_precalc_stats_singular.csv'.")
        return pd.DataFrame()

df = load_data()

# ==========================================
# PAGE 1: ANALYTICS
# ==========================================
if page == "📊 Persona Analytics":
    st.title("🤖 LLM vs 🧑‍🤝‍🧑 Human: Personality Persona Dashboard")
    st.markdown("Compare empirical human personality against Explicit and Implicit (NLP) LLM personas.")
    
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

        # --- EXTRACT ROWS FOR MATH & UI ---
        h_a = data_a[data_a['Source'] == 'Human'].iloc[0] if not data_a[data_a['Source'] == 'Human'].empty else None
        ai_a = data_a[data_a['Source'] == 'AI_Simulated'].iloc[0] if not data_a[data_a['Source'] == 'AI_Simulated'].empty else None
        nlp_a = data_a[data_a['Source'] == 'AI_NLP'].iloc[0] if not data_a[data_a['Source'] == 'AI_NLP'].empty else None
        
        h_b = data_b[data_b['Source'] == 'Human'].iloc[0] if not data_b[data_b['Source'] == 'Human'].empty else None
        ai_b = data_b[data_b['Source'] == 'AI_Simulated'].iloc[0] if not data_b[data_b['Source'] == 'AI_Simulated'].empty else None
        nlp_b = data_b[data_b['Source'] == 'AI_NLP'].iloc[0] if not data_b[data_b['Source'] == 'AI_NLP'].empty else None
        # --- PROFILE OVERVIEW ---
        st.markdown("---")
        st.header("👥 Profile Overview")
        
        col_desc_a, col_desc_b = st.columns(2)
        
        def format_description(c, r, a, ag, g):
            desc = f"**Country:** {c} | **Race:** {r} | **Gender:** {g} <br>"
            desc += f"**Age:** {a}" if a != "All" else f"**Age Group:** {ag}"
            return desc

        with col_desc_a:
            st.subheader("Group A")
            st.markdown(format_description(*a_vals), unsafe_allow_html=True)
            n_h_a = int(h_a[f"{t_pref}_Trait_count"]) if h_a is not None else 0
            n_ai_a = int(ai_a[f"{t_pref}_Trait_count"]) if ai_a is not None else 0
            n_nlp_a = int(nlp_a[f"{t_pref}_Trait_count"]) if nlp_a is not None else 0
            st.info(f"📊 **Sample Size:** {n_h_a} Humans | {n_ai_a} AI (Explicit) | {n_nlp_a} AI (NLP)")

        with col_desc_b:
            st.subheader("Group B")
            st.markdown(format_description(*b_vals), unsafe_allow_html=True)

            n_h_b = int(h_b[f"{t_pref}_Trait_count"]) if h_b is not None else 0
            n_ai_b = int(ai_b[f"{t_pref}_Trait_count"]) if ai_b is not None else 0
            n_nlp_b = int(nlp_b[f"{t_pref}_Trait_count"]) if nlp_b is not None else 0

            st.info(f"📊 **Sample Size:** {n_h_b} Humans | {n_ai_b} AI (Explicit) | {n_nlp_b} AI (NLP)")
                    
        st.markdown("---")
        st.header("📈 Statistical Significance Grid")

        if h_a is not None and ai_a is not None and nlp_a is not None and h_b is not None:

            # --- GROUP A COMPARISONS ---
            st.subheader("Group A Comparisons")

            p_ha_ai, d_ha_ai, txt_ha_ai, dt_ha_ai = calculate_stats(
                h_a[f"{t_pref}_Trait_mean"], h_a[f"{t_pref}_Trait_std"], h_a[f"{t_pref}_Trait_count"],
                ai_a[f"{t_pref}_Trait_mean"], ai_a[f"{t_pref}_Trait_std"], ai_a[f"{t_pref}_Trait_count"]
            )

            p_ha_nlp, d_ha_nlp, txt_ha_nlp, dt_ha_nlp = calculate_stats(
                h_a[f"{t_pref}_Trait_mean"], h_a[f"{t_pref}_Trait_std"], h_a[f"{t_pref}_Trait_count"],
                nlp_a[f"{t_pref}_Trait_mean"], nlp_a[f"{t_pref}_Trait_std"], nlp_a[f"{t_pref}_Trait_count"]
            )

            p_ai_nlp_a, d_ai_nlp_a, txt_ai_nlp_a, dt_ai_nlp_a = calculate_stats(
                ai_a[f"{t_pref}_Trait_mean"], ai_a[f"{t_pref}_Trait_std"], ai_a[f"{t_pref}_Trait_count"],
                nlp_a[f"{t_pref}_Trait_mean"], nlp_a[f"{t_pref}_Trait_std"], nlp_a[f"{t_pref}_Trait_count"]
            )

            st.dataframe(pd.DataFrame([
                ["Human A vs AI A", p_ha_ai, txt_ha_ai, dt_ha_ai, d_ha_ai],
                ["Human A vs NLP A", p_ha_nlp, txt_ha_nlp, dt_ha_nlp, d_ha_nlp],
                ["AI A vs NLP A", p_ai_nlp_a, txt_ai_nlp_a, dt_ai_nlp_a, d_ai_nlp_a],
            ], columns=["Comparison", "P-Value", "Significance", "Effect Size Label", "Cohen's d"]))


            # --- GROUP B COMPARISONS ---
            st.subheader("Group B Comparisons")

            if ai_b is not None and nlp_b is not None:

                p_hb_ai, d_hb_ai, txt_hb_ai, dt_hb_ai = calculate_stats(
                    h_b[f"{t_pref}_Trait_mean"], h_b[f"{t_pref}_Trait_std"], h_b[f"{t_pref}_Trait_count"],
                    ai_b[f"{t_pref}_Trait_mean"], ai_b[f"{t_pref}_Trait_std"], ai_b[f"{t_pref}_Trait_count"]
                )

                p_hb_nlp, d_hb_nlp, txt_hb_nlp, dt_hb_nlp = calculate_stats(
                    h_b[f"{t_pref}_Trait_mean"], h_b[f"{t_pref}_Trait_std"], h_b[f"{t_pref}_Trait_count"],
                    nlp_b[f"{t_pref}_Trait_mean"], nlp_b[f"{t_pref}_Trait_std"], nlp_b[f"{t_pref}_Trait_count"]
                )

                p_ai_nlp_b, d_ai_nlp_b, txt_ai_nlp_b, dt_ai_nlp_b = calculate_stats(
                    ai_b[f"{t_pref}_Trait_mean"], ai_b[f"{t_pref}_Trait_std"], ai_b[f"{t_pref}_Trait_count"],
                    nlp_b[f"{t_pref}_Trait_mean"], nlp_b[f"{t_pref}_Trait_std"], nlp_b[f"{t_pref}_Trait_count"]
                )

                st.dataframe(pd.DataFrame([
                    ["Human B vs AI B", p_hb_ai, txt_hb_ai, dt_hb_ai, d_hb_ai],
                    ["Human B vs NLP B", p_hb_nlp, txt_hb_nlp, dt_hb_nlp, d_hb_nlp],
                    ["AI B vs NLP B", p_ai_nlp_b, txt_ai_nlp_b, dt_ai_nlp_b, d_ai_nlp_b],
                ], columns=["Comparison", "P-Value", "Significance", "Effect Size Label", "Cohen's d"]))

            else:
                st.warning("Missing AI/NLP data for Group B")

            st.subheader("Cross-Group Comparisons")

            rows = []

            # Human A vs Human B
            p_hh, d_hh, txt_hh, dt_hh = calculate_stats(
                h_a[f"{t_pref}_Trait_mean"], h_a[f"{t_pref}_Trait_std"], h_a[f"{t_pref}_Trait_count"],
                h_b[f"{t_pref}_Trait_mean"], h_b[f"{t_pref}_Trait_std"], h_b[f"{t_pref}_Trait_count"]
            )
            rows.append(["Human A vs Human B", p_hh, txt_hh, dt_hh, d_hh])

            # AI A vs AI B
            if ai_a is not None and ai_b is not None:
                p_ai, d_ai, txt_ai, dt_ai = calculate_stats(
                    ai_a[f"{t_pref}_Trait_mean"], ai_a[f"{t_pref}_Trait_std"], ai_a[f"{t_pref}_Trait_count"],
                    ai_b[f"{t_pref}_Trait_mean"], ai_b[f"{t_pref}_Trait_std"], ai_b[f"{t_pref}_Trait_count"]
                )
                rows.append(["AI A vs AI B", p_ai, txt_ai, dt_ai, d_ai])

            # NLP A vs NLP B
            if nlp_a is not None and nlp_b is not None:
                p_nlp, d_nlp, txt_nlp, dt_nlp = calculate_stats(
                    nlp_a[f"{t_pref}_Trait_mean"], nlp_a[f"{t_pref}_Trait_std"], nlp_a[f"{t_pref}_Trait_count"],
                    nlp_b[f"{t_pref}_Trait_mean"], nlp_b[f"{t_pref}_Trait_std"], nlp_b[f"{t_pref}_Trait_count"]
                )
                rows.append(["NLP A vs NLP B", p_nlp, txt_nlp, dt_nlp, d_nlp])

            st.dataframe(pd.DataFrame(
                rows,
                columns=["Comparison", "P-Value", "Significance", "Effect Size Label", "Cohen's d"]
            ))
        else:
            st.warning("Not enough data for statistical comparisons.")
     
        
        # Bar Chart
        bar_data = []
        for d, n in zip([data_a, data_b], ["Group A", "Group B"]):
            for s, s_label in zip(['Human', 'AI_Simulated', 'AI_NLP'], ['🧑‍🤝‍🧑 Human', '🤖 Explicit AI', '🗣️ NLP AI']):
                row = d[d['Source'] == s]
                if not row.empty:
                    bar_data.append({
                        "Group": n, 
                        "Source": s_label, 
                        "Mean": row.iloc[0][f"{t_pref}_Trait_mean"], 
                        "StdDev": row.iloc[0][f"{t_pref}_Trait_std"]
                    })
        
        if bar_data:
            fig_bar = px.bar(
                pd.DataFrame(bar_data), 
                x='Group', y='Mean', color='Source', 
                barmode='group', error_y='StdDev', text_auto='.2f', 
                range_y=[1,5], title=f"Comparison: {sel_trait}",
                color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96']
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# PAGE 2: KEY DISCOVERIES
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
    
    st.header("🌍 Top 20 Extreme Human Subgroups (vs Global Average)")
    st.markdown("Human demographics that possess the most distinct personality traits compared to the rest of the world.")
    try:
        human_extreme_df = pd.read_csv("top_20_human_extremes.csv")
        st.dataframe(human_extreme_df, use_container_width=True)
    except:
        st.info("Run the Human Variance script to generate 'top_20_human_extremes.csv'")

    st.divider()
    
    st.header("⚔️ Top 50 Human-vs-Human Personality Clashes")
    st.markdown("The largest recorded personality gaps between two distinct human demographic groups.")
    try:
        clash_df = pd.read_csv("top_50_human_clashes.csv")
        st.dataframe(clash_df, use_container_width=True)
    except:
        st.info("Run the Pairwise Clash script to generate 'top_50_human_clashes.csv'")


# ==========================================
# PAGE 3: LITERATURE
# ==========================================
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