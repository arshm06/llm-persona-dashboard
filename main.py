import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Persona Dashboard", layout="wide", page_icon="📊")
st.title("🤖 LLM vs 🧑‍🤝‍🧑 Human: Personality Persona Dashboard")
st.markdown("Compare the empirical human personality distributions against LLM-simulated personas.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Load and aggregate LLM Data
        llm_df = pd.read_csv("all_personality_data.csv")
        llm_agg = llm_df.groupby(['Category', 'Persona', 'Trait'])['Score'].sum().reset_index()
        llm_agg['Mean_Score'] = llm_agg['Score'] / 30  # Assuming 30 iterations
        
        # Load Human Data (from the new processing script)
        human_df = pd.read_csv("subgroup_personality_stats.csv")
        # Calculate Standard Deviation for error bars
        human_df['StdDev'] = np.sqrt(human_df['Variance'])
        
        return llm_agg, human_df
    except FileNotFoundError as e:
        st.error(f"Missing Data File: Ensure 'all_personality_data.csv' and 'subgroup_personality_stats.csv' are in the same folder.")
        return pd.DataFrame(), pd.DataFrame()

llm_data, human_data = load_data()

if not llm_data.empty and not human_data.empty:
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Filter Settings")
    
    # Find overlapping categories
    categories = list(set(llm_data['Category'].unique()).intersection(set(human_data['Category'].unique())))
    category = st.sidebar.selectbox("Select Demographic Category", sorted(categories))
    
    # Filter datasets by category
    cat_llm = llm_data[llm_data['Category'] == category]
    cat_human = human_data[human_data['Category'] == category]
    
    # Find overlapping groups
    available_groups = list(set(cat_llm['Persona'].unique()).intersection(set(cat_human['Subgroup'].unique())))
    available_groups = sorted(available_groups)
    
    if len(available_groups) < 2:
        st.warning(f"Not enough matching subgroups in both datasets for {category}.")
        st.stop()

    group_a = st.sidebar.selectbox("Select Group A", available_groups, index=0)
    group_b = st.sidebar.selectbox("Select Group B", available_groups, index=1)
    
    traits = ["Extroversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Emotional Stability", "Openness"]
    # Ensure we only pick traits that exist in the data
    valid_traits = [t for t in traits if t in cat_llm['Trait'].unique()]
    selected_trait = st.sidebar.selectbox("Select Trait for Deep Dive", valid_traits)

    st.markdown("---")

    # --- SECTION 1: OVERALL PROFILE (RADAR CHARTS) ---
    st.header(f"Overall Personality Profiles: {group_a} vs {group_b}")
    
    col_radar1, col_radar2 = st.columns(2)
    
    # Helper to draw radar chart
    def draw_radar(df, group_col, score_col, title):
        fig = go.Figure()
        for group in [group_a, group_b]:
            group_df = df[df[group_col] == group]
            # Match traits order
            traits_order = ["Extroversion", "Agreeableness", "Conscientiousness", "Emotional Stability", "Openness"]
            # Fallback if Emotional Stability is named Neuroticism in human data
            if "Emotional Stability" not in group_df['Trait'].values and "Neuroticism" in group_df['Trait'].values:
                # We won't map it here to keep the radar simple, we'll just use available traits
                traits_order = group_df['Trait'].unique().tolist()
                
            # Sort data to match traits_order
            group_df = group_df.set_index('Trait').reindex(traits_order).reset_index()
            
            fig.add_trace(go.Scatterpolar(
                r=group_df[score_col].tolist() + [group_df[score_col].tolist()[0]],
                theta=group_df['Trait'].tolist() + [group_df['Trait'].tolist()[0]],
                fill='toself',
                name=group
            ))
        fig.update_layout(
            title=title,
            polar=dict(radialaxis=dict(visible=True, range=[20, 45])), # Zoomed in slightly for better contrast
            showlegend=True
        )
        return fig

    with col_radar1:
        st.plotly_chart(draw_radar(cat_human, 'Subgroup', 'Mean', "🧑‍🤝‍🧑 Actual Human Profiles"), use_container_width=True)
        
    with col_radar2:
        st.plotly_chart(draw_radar(cat_llm, 'Persona', 'Mean_Score', "🤖 LLM Simulated Profiles"), use_container_width=True)


    st.markdown("---")

    # --- SECTION 2: TRAIT DEEP DIVE (BAR CHARTS WITH ERROR BARS) ---
    st.header(f"Trait Deep Dive: {selected_trait}")
    st.write(f"Comparing absolute scores for **{selected_trait}** on a scale of 10 to 50.")
    
    col_bar1, col_bar2 = st.columns(2)
    
    with col_bar1:
        st.subheader("🧑‍🤝‍🧑 Human Data (Mean ± StdDev)")
        human_filtered = cat_human[(cat_human['Trait'] == selected_trait) & 
                                   (cat_human['Subgroup'].isin([group_a, group_b]))]
        
        if not human_filtered.empty:
            fig_human = px.bar(
                human_filtered, x='Subgroup', y='Mean', color='Subgroup', 
                error_y='StdDev', text_auto='.1f',
                color_discrete_sequence=['#636EFA', '#EF553B']
            )
            fig_human.update_yaxes(range=[10, 50])
            st.plotly_chart(fig_human, use_container_width=True)
        else:
            st.info("Trait not found in Human Data (Check if it's labeled 'Neuroticism' vs 'Emotional Stability').")
            
    with col_bar2:
        st.subheader("🤖 LLM Data (Mean Score)")
        llm_filtered = cat_llm[(cat_llm['Trait'] == selected_trait) & 
                               (cat_llm['Persona'].isin([group_a, group_b]))]
        
        if not llm_filtered.empty:
            fig_llm = px.bar(
                llm_filtered, x='Persona', y='Mean_Score', color='Persona', 
                text_auto='.1f',
                color_discrete_sequence=['#636EFA', '#EF553B']
            )
            fig_llm.update_yaxes(range=[10, 50])
            st.plotly_chart(fig_llm, use_container_width=True)

    # --- SECTION 3: DATA TABLES ---
    st.markdown("---")
    with st.expander("View Underlying Data"):
        col_tab1, col_tab2 = st.columns(2)
        with col_tab1:
            st.write("**Human Stats**")
            st.dataframe(human_filtered[['Category', 'Subgroup', 'Trait', 'Sample_Size', 'Mean', 'StdDev']])
        with col_tab2:
            st.write("**LLM Stats**")
            st.dataframe(llm_filtered[['Category', 'Persona', 'Trait', 'Mean_Score']])