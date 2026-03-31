import pandas as pd
import numpy as np

# 1. Load the pre-calculated stats
print("Loading data...")
try:
    df = pd.read_csv("dashboard_precalc_stats_all.csv")
except FileNotFoundError:
    print("Error: Could not find 'dashboard_precalc_stats_singular.csv'.")
    exit()

# Filter to ONLY look at the actual Human data
human_df = df[df['Source'] == 'Human'].copy()

# 2. Extract the Global Baseline (The row where every demographic is 'All')
demo_cols = ['Country', 'Race', 'Age', 'Age_Group', 'Gender']
baseline_mask = (human_df[demo_cols] == 'All').all(axis=1)

if not baseline_mask.any():
    print("Error: Could not find the Global Baseline row in the data.")
    exit()

baseline_row = human_df[baseline_mask].iloc[0]

# Remove the baseline from the pool so we don't compare it to itself
subgroups_df = human_df[~baseline_mask].copy()

traits = ['E', 'A', 'C', 'N', 'O']
trait_names = {'E': 'Extroversion', 'A': 'Agreeableness', 'C': 'Conscientiousness', 'N': 'Neuroticism', 'O': 'Openness'}

results = []
print("Calculating divergences from the Global Human Average...")

for index, row in subgroups_df.iterrows():
    for t in traits:
        # Subgroup Stats
        m_sub = row[f"{t}_Trait_mean"]
        s_sub = row[f"{t}_Trait_std"]
        n_sub = row[f"{t}_Trait_count"]
        
        # Baseline Stats
        m_base = baseline_row[f"{t}_Trait_mean"]
        s_base = baseline_row[f"{t}_Trait_std"]
        n_base = baseline_row[f"{t}_Trait_count"]
        
        # STRICT FILTER: Only look at subgroups with at least 30 people for statistical validity
        if pd.isna(m_sub) or n_sub < 30:
            continue
            
        # Calculate Pooled Standard Deviation and Cohen's d against the Baseline
        pooled_var = (((n_sub - 1) * s_sub**2) + ((n_base - 1) * s_base**2)) / (n_sub + n_base - 2)
        if pooled_var > 0:
            pooled_std = np.sqrt(pooled_var)
            cohens_d = (m_sub - m_base) / pooled_std
        else:
            cohens_d = 0.0
            
        # Format the profile description nicely
        profile_parts = [f"{col}: {row[col]}" for col in demo_cols if row[col] != 'All']
        profile_desc = " | ".join(profile_parts)
        
        results.append({
            "Profile": profile_desc,
            "Trait": trait_names[t],
            "Global_Mean": round(m_base, 2),
            "Subgroup_Mean": round(m_sub, 2),
            "Difference": round(m_sub - m_base, 2),
            "Cohens_D": round(cohens_d, 3),
            "Abs_Effect_Size": abs(cohens_d),
            "Sample_Size": int(n_sub)
        })

# 3. Create DataFrame and sort by the most extreme divergences
results_df = pd.DataFrame(results)

# Sort by Absolute Effect Size (largest deviations from the norm first)
top_20 = results_df.sort_values(by="Abs_Effect_Size", ascending=False).head(20)

# Drop the absolute column for clean viewing
top_20 = top_20.drop(columns=["Abs_Effect_Size"])

print("\n" + "="*90)
print("🏆 TOP 20 MOST EXTREME HUMAN DEMOGRAPHIC TRAITS (vs Global Baseline)")
print("="*90)

print(top_20.to_string(index=False))

# Export to a CSV
top_20.to_csv("top_20_human_extremes.csv", index=False)
print("\nSuccess! Saved full details to 'top_20_human_extremes.csv'")