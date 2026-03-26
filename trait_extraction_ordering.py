import pandas as pd
import numpy as np
from itertools import combinations

# ==========================================
# 1. SCORING LOGIC
# ==========================================
# 1 means standard scoring, -1 means reverse scored (6 - value)
SCORING_KEYS = {
    'E': {'E1':1, 'E2':-1, 'E3':1, 'E4':-1, 'E5':1, 'E6':-1, 'E7':1, 'E8':-1, 'E9':1, 'E10':-1},
    'A': {'A1':-1, 'A2':1, 'A3':-1, 'A4':1, 'A5':-1, 'A6':1, 'A7':-1, 'A8':1, 'A9':1, 'A10':1},
    'C': {'C1':1, 'C2':-1, 'C3':1, 'C4':-1, 'C5':1, 'C6':-1, 'C7':1, 'C8':-1, 'C9':1, 'C10':1},
    'N': {'N1':-1, 'N2':1, 'N3':-1, 'N4':1, 'N5':-1, 'N6':-1, 'N7':-1, 'N8':-1, 'N9':-1, 'N10':-1},
    'O': {'O1':1, 'O2':-1, 'O3':1, 'O4':-1, 'O5':1, 'O6':-1, 'O7':1, 'O8':1, 'O9':1, 'O10':1}
}

def calculate_traits(df):
    df = df.copy()
    traits = ['E', 'A', 'C', 'N', 'O']
    
    for trait in traits:
        trait_cols = []
        for q_id, sign in SCORING_KEYS[trait].items():
            col_name = f"{q_id}_score" # Always simulated for this experiment
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
print("Loading ordering experiment data...")
try:
    sim_df = pd.read_csv('simulated_ordering_experiment.csv')
except FileNotFoundError:
    print("simulated_ordering_experiment.csv not found. Run generate_ordering_experiment.py first.")
    exit(1)

print("Calculating Big Five Traits...")
sim_df = calculate_traits(sim_df)

def get_age_group(age):
    if age < 18: return "<18"
    if 18 <= age <= 25: return "18-25"
    if 26 <= age <= 34: return "26-34"
    if 35 <= age <= 50: return "35-50"
    return "50+"

sim_df['Age_Group'] = sim_df['Age'].apply(get_age_group)

demo_cols = ['Country', 'Race', 'Age', 'Age_Group', 'Gender']
trait_cols = ['E_Trait', 'A_Trait', 'C_Trait', 'N_Trait', 'O_Trait']

# Keep only needed columns
combined_df = sim_df[demo_cols + trait_cols + ['Ordering']]

# ==========================================
# 3. GENERATE ALL SUBGROUP COMBINATIONS
# ==========================================
print("Aggregating all possible demographic combinations by Ordering...")
all_aggregations = []

# Support fast dynamic filtering on dashboard by pre-calculating every combo
for r in range(len(demo_cols) + 1):
    for combo in combinations(demo_cols, r):
        group_by_cols = list(combo) + ['Ordering']
        
        if len(combo) == 0:
            agg_df = combined_df.groupby('Ordering')[trait_cols].agg(['mean', 'std', 'count']).reset_index()
        else:
            agg_df = combined_df.groupby(group_by_cols)[trait_cols].agg(['mean', 'std', 'count']).reset_index()
        
        agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
        
        # Fill missing demo cols
        for col in demo_cols:
            if col not in combo:
                agg_df[col] = 'All'
                
        all_aggregations.append(agg_df)

# ==========================================
# 4. FINALIZE AND EXPORT
# ==========================================
final_dashboard_data = pd.concat(all_aggregations, ignore_index=True)

col_order = demo_cols + ['Ordering'] + [f"{t}_Trait_{stat}" for t in ['E', 'A', 'C', 'N', 'O'] for stat in ['mean', 'std', 'count']]
final_dashboard_data = final_dashboard_data[col_order]

float_cols = final_dashboard_data.select_dtypes(include=['float64']).columns
final_dashboard_data[float_cols] = final_dashboard_data[float_cols].round(3)

OUTPUT_FILE = "dashboard_ordering_stats.csv"
final_dashboard_data.to_csv(OUTPUT_FILE, index=False)
print(f"Success! Dashboard ordering stats saved to {OUTPUT_FILE} with {len(final_dashboard_data)} rows.")
