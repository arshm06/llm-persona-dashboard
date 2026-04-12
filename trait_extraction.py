import pandas as pd
import numpy as np
from itertools import combinations

# ==========================================
# 1. SCORING LOGIC
# ==========================================
SCORING_KEYS = {
    'E': {'E1':1, 'E2':-1, 'E3':1, 'E4':-1, 'E5':1, 'E6':-1, 'E7':1, 'E8':-1, 'E9':1, 'E10':-1},
    'A': {'A1':-1, 'A2':1, 'A3':-1, 'A4':1, 'A5':-1, 'A6':1, 'A7':-1, 'A8':1, 'A9':1, 'A10':1},
    'C': {'C1':1, 'C2':-1, 'C3':1, 'C4':-1, 'C5':1, 'C6':-1, 'C7':1, 'C8':-1, 'C9':1, 'C10':1},
    'N': {'N1':-1, 'N2':1, 'N3':-1, 'N4':1, 'N5':-1, 'N6':-1, 'N7':-1, 'N8':-1, 'N9':-1, 'N10':-1},
    'O': {'O1':1, 'O2':-1, 'O3':1, 'O4':-1, 'O5':1, 'O6':-1, 'O7':1, 'O8':1, 'O9':1, 'O10':1}
}

def calculate_traits(df, is_simulated=False):
    df = df.copy()
    traits = ['E', 'A', 'C', 'N', 'O']
    
    for trait in traits:
        trait_cols = []
        for q_id, sign in SCORING_KEYS[trait].items():
            col_name = f"{q_id}_score" if is_simulated else q_id
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            
            scored_col = f"{q_id}_final"
            if sign == 1:
                df[scored_col] = df[col_name]
            else:
                df[scored_col] = 6 - df[col_name]
            trait_cols.append(scored_col)
            
        df[f"{trait}_Trait"] = df[trait_cols].mean(axis=1)
    return df

# ==========================================
# 2. LOAD & PREP DATA
# ==========================================
print("Loading datasets...")
try: human_df = pd.read_csv('human_data.csv', sep='\t', low_memory=False)
except: human_df = pd.DataFrame()

try: sim_df = pd.read_csv('simulated_human_data_isolated.csv')
except: sim_df = pd.DataFrame()

try: nlp_df = pd.read_csv('simulated_human_data_nlp.csv')
except: nlp_df = pd.DataFrame()

try: qwen_df = pd.read_csv('simulated_human_data_isolated_qwen.csv')
except: qwen_df = pd.DataFrame()

try: qwen_nlp_df = pd.read_csv('simulated_human_data_nlp_qwen.csv')
except: qwen_nlp_df = pd.DataFrame()

try: order_df = pd.read_csv('simulated_ordering_experiment.csv')
except: order_df = pd.DataFrame()

try: qwen_order_df = pd.read_csv('simulated_ordering_experiment_qwen2.5-3b.csv')
except: qwen_order_df = pd.DataFrame()

# --- FILTER FOR FIRST RUN ONLY ---
if not sim_df.empty: sim_df = sim_df[sim_df['Sim_ID'].str.contains('_R0')].copy()
if not nlp_df.empty: nlp_df = nlp_df[nlp_df['Sim_ID'].str.contains('_R0')].copy()
if not qwen_df.empty: qwen_df = qwen_df[qwen_df['Sim_ID'].str.contains('_R0')].copy()
if not qwen_nlp_df.empty: qwen_nlp_df = qwen_nlp_df[qwen_nlp_df['Sim_ID'].str.contains('_R0')].copy()
if not order_df.empty: order_df = order_df[order_df['Sim_ID'].str.contains('_R0_')].copy()
# Note: qwen_order_df only has 1 iteration built-in, so no filtering needed.

# Clean human data
if not human_df.empty:
    iso_df = pd.read_csv("iso.csv")
    iso_map = dict(zip(iso_df['alpha-2'], iso_df['name']))
    human_df['Country'] = human_df['country'].map(iso_map)

    RACE_MAP = {1: 'Mixed Race', 2: 'Arctic', 3: 'Caucasian (European)', 4: 'Caucasian (Indian)', 5: 'Caucasian (Middle East)', 6: 'Caucasian (North African)', 7: 'Indigenous Australian', 8: 'Native American', 9: 'North East Asian', 10: 'Pacific Islander', 11: 'South East Asian', 12: 'West African', 13: 'Other'}
    GENDER_MAP = {1: 'Male', 2: 'Female', 3: 'Other'}
    human_df['Race'] = pd.to_numeric(human_df['race'], errors='coerce').map(RACE_MAP)
    human_df['Gender'] = pd.to_numeric(human_df['gender'], errors='coerce').map(GENDER_MAP)
    human_df['Age'] = pd.to_numeric(human_df['age'], errors='coerce')

    human_df = human_df.dropna(subset=['Country', 'Race', 'Gender', 'Age'])
    human_df = human_df[(human_df['Age'] >= 13) & (human_df['Age'] <= 100)]

# Calculate Trait Scores
print("Calculating Big Five Traits...")
if not human_df.empty: human_df = calculate_traits(human_df, is_simulated=False)
if not sim_df.empty: sim_df = calculate_traits(sim_df, is_simulated=True)
if not nlp_df.empty: nlp_df = calculate_traits(nlp_df, is_simulated=True)
if not qwen_df.empty: qwen_df = calculate_traits(qwen_df, is_simulated=True)
if not qwen_nlp_df.empty: qwen_nlp_df = calculate_traits(qwen_nlp_df, is_simulated=True)
if not order_df.empty: order_df = calculate_traits(order_df, is_simulated=True)
if not qwen_order_df.empty: qwen_order_df = calculate_traits(qwen_order_df, is_simulated=True)

# Tag data sources
if not human_df.empty: human_df['Source'] = 'Human'
if not sim_df.empty: sim_df['Source'] = 'AI_GPT4o_Explicit'
if not nlp_df.empty: nlp_df['Source'] = 'AI_GPT4o_NLP'
if not qwen_df.empty: qwen_df['Source'] = 'AI_Qwen_Explicit'
if not qwen_nlp_df.empty: qwen_nlp_df['Source'] = 'AI_Qwen_NLP'

# Map Orderings
gpt4o_order_map = {
    "Age -> Gender -> Nationality": "AI_GPT4o_Order_AGN",
    "Gender -> Age -> Nationality": "AI_GPT4o_Order_GAN",
    "Nationality -> Age -> Gender": "AI_GPT4o_Order_NAG"
}
if not order_df.empty: order_df['Source'] = order_df['Ordering'].map(gpt4o_order_map)

qwen_order_map = {
    "Age -> Gender -> Nationality": "AI_Qwen_Order_AGN",
    "Gender -> Age -> Nationality": "AI_Qwen_Order_GAN",
    "Nationality -> Age -> Gender": "AI_Qwen_Order_NAG"
}
if not qwen_order_df.empty: qwen_order_df['Source'] = qwen_order_df['Ordering'].map(qwen_order_map)

def get_age_group(age):
    if age < 18: return "<18"
    if 18 <= age <= 25: return "18-25"
    if 26 <= age <= 34: return "26-34"
    if 35 <= age <= 50: return "35-50"
    return "50+"

dfs_to_concat = []
for d in [human_df, sim_df, nlp_df, qwen_df, qwen_nlp_df, order_df, qwen_order_df]:
    if not d.empty:
        d['Age_Group'] = d['Age'].apply(get_age_group)
        dfs_to_concat.append(d)

demo_cols = ['Country', 'Race', 'Age', 'Age_Group', 'Gender']
trait_cols = ['E_Trait', 'A_Trait', 'C_Trait', 'N_Trait', 'O_Trait']

# Combine all valid datasets
combined_df = pd.concat([d[demo_cols + trait_cols + ['Source']] for d in dfs_to_concat])

# ==========================================
# 3. GENERATE ALL SUBGROUP COMBINATIONS
# ==========================================
print("Aggregating all possible demographic combinations...")
all_aggregations = []

for r in range(len(demo_cols) + 1):
    for combo in combinations(demo_cols, r):
        group_by_cols = list(combo) + ['Source']
        
        if len(combo) == 0:
            agg_df = combined_df.groupby('Source')[trait_cols].agg(['mean', 'std', 'count']).reset_index()
        else:
            agg_df = combined_df.groupby(group_by_cols)[trait_cols].agg(['mean', 'std', 'count']).reset_index()
        
        agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
        
        for col in demo_cols:
            if col not in combo:
                agg_df[col] = 'All'
                
        all_aggregations.append(agg_df)

# ==========================================
# 4. FINALIZE AND EXPORT
# ==========================================
final_dashboard_data = pd.concat(all_aggregations, ignore_index=True)

col_order = demo_cols + ['Source'] + [f"{t}_Trait_{stat}" for t in ['E', 'A', 'C', 'N', 'O'] for stat in ['mean', 'std', 'count']]
final_dashboard_data = final_dashboard_data[col_order]

float_cols = final_dashboard_data.select_dtypes(include=['float64']).columns
final_dashboard_data[float_cols] = final_dashboard_data[float_cols].round(3)

OUTPUT_FILE = "dashboard_precalc_stats_singular.csv"
final_dashboard_data.to_csv(OUTPUT_FILE, index=False)
print(f"Success! Dashboard data saved to {OUTPUT_FILE} with {len(final_dashboard_data)} rows.")