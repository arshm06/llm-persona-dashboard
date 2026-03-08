import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Big Five: Human vs AI", layout="wide", page_icon="📊")

# --- NAVIGATION ---
# This adds the multi-page functionality
page = st.sidebar.radio("Navigation", ["📊 Persona Analytics", "📖 Relevant Literature"])

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dashboard_precalc_stats.csv")
        # Ensure all demographic columns are strings, including the new Age_Group
        demo_cols = ['Country', 'Race', 'Age', 'Age_Group', 'Gender']
        for col in demo_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df
    except FileNotFoundError:
        st.error("Missing Data File: Ensure 'dashboard_precalc_stats.csv' is in the directory.")
        return pd.DataFrame()

df = load_data()

# ==========================================
# PAGE 1: ANALYTICS
# ==========================================
if page == "📊 Persona Analytics":
    st.title("🤖 LLM vs 🧑‍🤝‍🧑 Human: Personality Persona Dashboard")
    st.markdown("Compare empirical human personality distributions against synthetic LLM personas.")

    if not df.empty:
        # --- HELPER: GET UNIQUE DEMOGRAPHICS ---
        def get_options(column):
            opts = sorted([x for x in df[column].unique() if x != 'All'])
            return ['All'] + opts

        # --- SIDEBAR CONTROLS ---
        st.sidebar.header("Filter Settings")
        
        # Function to build a profile UI
        def build_profile_sidebar(letter):
            st.sidebar.subheader(f"Profile {letter} Definition")
            country = st.sidebar.selectbox(f"Country ({letter})", get_options('Country'), key=f'c_{letter}')
            race = st.sidebar.selectbox(f"Race ({letter})", get_options('Race'), key=f'r_{letter}')
            
            # Age Filter Logic: Choose between Range or Specific Year
            age_type = st.sidebar.radio(f"Age Filter ({letter})", ["Range", "Specific Year"], key=f'at_{letter}')
            if age_type == "Range":
                age_group = st.sidebar.selectbox(f"Select Range ({letter})", get_options('Age_Group'), key=f'ag_{letter}')
                age = "All"
            else:
                age = st.sidebar.selectbox(f"Select Year ({letter})", get_options('Age'), key=f'a_{letter}')
                age_group = "All"
                
            gender = st.sidebar.selectbox(f"Gender ({letter})", get_options('Gender'), key=f'g_{letter}')
            return country, race, age, age_group, gender

        # Build Sidebars for A and B
        a_vals = build_profile_sidebar("A")
        st.sidebar.markdown("---")
        b_vals = build_profile_sidebar("B")

        # Trait Selection
        st.sidebar.markdown("---")
        trait_map = {"Extroversion": "E", "Agreeableness": "A", "Conscientiousness": "C", "Neuroticism": "N", "Openness": "O"}
        selected_trait_name = st.sidebar.selectbox("Select Trait for Deep Dive", list(trait_map.keys()))
        selected_trait_prefix = trait_map[selected_trait_name]

        # --- EXTRACT PROFILES ---
        def get_profile_data(country, race, age, age_group, gender):
            mask = (df['Country'] == country) & (df['Race'] == race) & \
                   (df['Age'] == age) & (df['Age_Group'] == age_group) & \
                   (df['Gender'] == gender)
            return df[mask]

        data_a = get_profile_data(*a_vals)
        data_b = get_profile_data(*b_vals)

        if data_a.empty or data_b.empty:
            st.warning("One or both combinations do not exist in the dataset. Try selecting 'All' for some filters.")
            st.stop()
        
        # --- SECTION 2: BAR CHARTS ---
        st.header(f"Trait Deep Dive: {selected_trait_name}")
        bar_data = []
        for profile_data, group_name in zip([data_a, data_b], ["Group A", "Group B"]):
            for source in ['Human', 'AI_Simulated']:
                source_row = profile_data[profile_data['Source'] == source]
                if not source_row.empty:
                    bar_data.append({
                        "Group": group_name,
                        "Source": "🧑‍🤝‍🧑 Human" if source == 'Human' else "🤖 AI Simulated",
                        "Mean": source_row.iloc[0][f"{selected_trait_prefix}_Trait_mean"],
                        "StdDev": source_row.iloc[0][f"{selected_trait_prefix}_Trait_std"],
                        "Sample Size": source_row.iloc[0][f"{selected_trait_prefix}_Trait_count"]
                    })
        
        if bar_data:
            bar_df = pd.DataFrame(bar_data)
            fig_bar = px.bar(bar_df, x='Group', y='Mean', color='Source', barmode='group',
                             error_y='StdDev', text_auto='.2f', color_discrete_sequence=['#636EFA', '#EF553B'])
            fig_bar.update_yaxes(range=[1, 5])
            st.plotly_chart(fig_bar, use_container_width=True)

        with st.expander("View Underlying Data"):
            st.dataframe(pd.concat([data_a, data_b]).reset_index(drop=True))

# ==========================================
# PAGE 2: LITERATURE
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