import pandas as pd
import numpy as np

INPUT_FILE = "data.csv"
OUTPUT_FILE = "subgroup_personality_stats.csv"

GENDER_MAP = {1: 'Male', 2: 'Female', 3: 'Other'}
RACE_MAP = {
    1: 'Mixed', 2: 'Arctic', 3: 'Caucasian (Euro)', 4: 'Caucasian (Ind)',
    5: 'Caucasian (MidEast)', 6: 'Caucasian (NorthAf)', 7: 'Indigenous Aus',
    8: 'Native Amer', 9: 'NE Asian', 10: 'Pacific', 11: 'SE Asian',
    12: 'W African', 13: 'Other'
}


SCORING_KEYS = {
    "Extroversion": {
        "pos": ['E1', 'E3', 'E5', 'E7', 'E9'],
        "neg": ['E2', 'E4', 'E6', 'E8', 'E10']
    },
    "Neuroticism": {
        "pos": ['N1', 'N3', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10'],
        "neg": ['N2', 'N4']
    },
    "Agreeableness": {
        "pos": ['A2', 'A4', 'A6', 'A8', 'A9', 'A10'],
        "neg": ['A1', 'A3', 'A5', 'A7']
    },
    "Conscientiousness": {
        "pos": ['C1', 'C3', 'C5', 'C7', 'C9', 'C10'],
        "neg": ['C2', 'C4', 'C6', 'C8']
    },
    "Openness": {
        "pos": ['O1', 'O3', 'O5', 'O7', 'O8', 'O9', 'O10'],
        "neg": ['O2', 'O4', 'O6']
    }
}

def analyze_personality_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, sep='\t')
    except FileNotFoundError:
        print("Error: Input file not found.")
        return
    question_cols = [f"{trait}{i}" for trait in ['E', 'N', 'A', 'C', 'O'] for i in range(1, 11)]
    
    available_cols = [col for col in question_cols if col in df.columns]
    df[available_cols] = df[available_cols].replace(0, 3) 

    print("Calculating reverse scores and total trait scores...")
    for trait, keys in SCORING_KEYS.items():
        pos_sum = df[keys['pos']].sum(axis=1) if keys['pos'] else 0
        
        neg_sum = (6 - df[keys['neg']]).sum(axis=1) if keys['neg'] else 0
        
        # Total Trait Score (Range: 10 - 50)
        df[trait] = pos_sum + neg_sum

    print("Mapping demographics (Age, Gender, Race)...")
    
    df['Gender_Label'] = df['gender'].map(GENDER_MAP).fillna('Unknown')
    df['Race_Label'] = df['race'].map(RACE_MAP).fillna('Unknown')
    

    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df[(df['age'] >= 13) & (df['age'] <= 100)].copy()
    
    bins = [12, 17, 25, 35, 50, 100]
    labels = ['<18', '18-25', '25-35', '35-50', '50+']
    df['Age_Group'] = pd.cut(df['age'], bins=bins, labels=labels)

    print("Aggregating statistics...")
    results = []
    traits = list(SCORING_KEYS.keys())
    
    categories = {
        'Gender': 'Gender_Label',
        'Age': 'Age_Group',
        'Race': 'Race_Label'
    }

    for cat_name, cat_col in categories.items():
        grouped = df.groupby(cat_col)
        
        for group_name, group_data in grouped:
            if pd.isna(group_name) or group_name == 'Unknown':
                continue
                
            sample_size = len(group_data)
            
            for trait in traits:
                mean_val = group_data[trait].mean()
                var_val = group_data[trait].var()
                
                results.append({
                    "Category": cat_name,
                    "Subgroup": group_name,
                    "Trait": trait,
                    "Sample_Size": sample_size,
                    "Mean": round(mean_val, 2),
                    "Variance": round(var_val, 2)
                })

    final_df = pd.DataFrame(results)
    
    final_df = final_df.sort_values(by=['Category', 'Subgroup', 'Trait'])
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAnalysis complete! Data saved to {OUTPUT_FILE}")
    print("\nPreview of Results:")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    analyze_personality_data(INPUT_FILE)