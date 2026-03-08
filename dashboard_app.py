import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Big Five: Human vs AI", layout="wide", page_icon="📊")
st.title("🤖 LLM vs 🧑‍🤝‍🧑 Human: Personality Persona Dashboard")
st.markdown("Compare the empirical human personality distributions against synthetic LLM personas.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Load our unified pre-calculated stats
        df = pd.read_csv("dashboard_precalc_stats.csv")
        # Ensure demographics are treated as strings for exact matching, including 'All'
        for col in ['Country', 'Race', 'Age', 'Gender']:
            df[col] = df[col].astype(str)
        return df
    except FileNotFoundError:
        st.error("Missing Data File: Ensure 'dashboard_precalc_stats.csv' is in the directory.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    
    # --- HELPER: GET UNIQUE DEMOGRAPHICS ---
    def get_options(column):
        # Sort values but keep 'All' at the top
        opts = sorted([x for x in df[column].unique() if x != 'All'])
        return ['All'] + opts

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Filter Settings")
    st.sidebar.markdown("Build your demographic profiles below.")
    
    # Profile A Setup
    st.sidebar.subheader("Profile A Definition")
    a_country = st.sidebar.selectbox("Country (A)", get_options('Country'), key='a_c')
    a_race = st.sidebar.selectbox("Race (A)", get_options('Race'), key='a_r')
    a_age = st.sidebar.selectbox("Age (A)", get_options('Age'), key='a_a')
    a_gender = st.sidebar.selectbox("Gender (A)", get_options('Gender'), key='a_g')
    name_a = f"Group A" # You can make this dynamic if desired

    # Profile B Setup
    st.sidebar.subheader("Profile B Definition")
    b_country = st.sidebar.selectbox("Country (B)", get_options('Country'), key='b_c', index=0)
    b_race = st.sidebar.selectbox("Race (B)", get_options('Race'), key='b_r', index=0)
    b_age = st.sidebar.selectbox("Age (B)", get_options('Age'), key='b_a', index=0)
    b_gender = st.sidebar.selectbox("Gender (B)", get_options('Gender'), key='b_g', index=0)
    name_b = f"Group B"

    # Deep Dive Selection
    st.sidebar.markdown("---")
    trait_map = {
        "Extroversion": "E", 
        "Agreeableness": "A", 
        "Conscientiousness": "C", 
        "Neuroticism": "N", 
        "Openness": "O"
    }
    selected_trait_name = st.sidebar.selectbox("Select Trait for Deep Dive", list(trait_map.keys()))
    selected_trait_prefix = trait_map[selected_trait_name]

    # --- EXTRACT PROFILES ---
    def get_profile_data(country, race, age, gender):
        # Filter the pre-calculated dataframe for the exact combination
        mask = (df['Country'] == country) & (df['Race'] == race) & \
               (df['Age'] == age) & (df['Gender'] == gender)
        return df[mask]

    data_a = get_profile_data(a_country, a_race, a_age, a_gender)
    data_b = get_profile_data(b_country, b_race, b_age, b_gender)

    if data_a.empty or data_b.empty:
        st.warning("One or both of these specific demographic combinations do not exist in the dataset. Try selecting 'All' for more specific filters.")
        st.stop()

    # --- SECTION 2: TRAIT DEEP DIVE (BAR CHARTS WITH ERROR BARS) ---
    st.header(f"Trait Deep Dive: {selected_trait_name}")
    st.write(f"Comparing absolute scores for **{selected_trait_name}** on a scale of 1 to 5.")
    
    # Prepare data for the bar chart
    bar_data = []
    for profile_data, group_name in zip([data_a, data_b], [name_a, name_b]):
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
                
    bar_df = pd.DataFrame(bar_data)

    if not bar_df.empty:
        fig_bar = px.bar(
            bar_df, 
            x='Group', 
            y='Mean', 
            color='Source', 
            barmode='group',
            error_y='StdDev', 
            text_auto='.2f',
            color_discrete_sequence=['#636EFA', '#EF553B'],
            hover_data=['Sample Size']
        )
        fig_bar.update_yaxes(range=[1, 5])
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- SECTION 3: DATA TABLES ---
    st.markdown("---")
    with st.expander("View Underlying Data"):
        st.write("**Pre-Calculated Statistics for Selected Groups**")
        display_df = pd.concat([data_a, data_b]).reset_index(drop=True)
        st.dataframe(display_df)