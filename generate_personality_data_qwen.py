"""
Arsh's generate_personality_data.py adapted for local Qwen2.5-3B-Instruct.
Explicit persona pipeline — one question at a time, isolated.

Usage:
    CUDA_VISIBLE_DEVICES=3 python generate_personality_data_qwen.py
"""

import math
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "simulated_human_data_isolated_qwen.csv"
SAMPLE_SIZE = 1720
ITERATIONS_PER_ROW = 1  # Arsh does 3 but 1 is enough for tomorrow
BATCH_SIZE = 8
RANDOM_SEED = 42

VALID_WORDS = {"never", "rarely", "sometimes", "often", "always"}

RACE_MAP = {
    1: 'Mixed Race', 2: 'Arctic', 3: 'Caucasian (European)', 4: 'Caucasian (Indian)',
    5: 'Caucasian (Middle East)', 6: 'Caucasian (North African)', 7: 'Indigenous Australian',
    8: 'Native American', 9: 'North East Asian', 10: 'Pacific Islander',
    11: 'South East Asian', 12: 'West African', 13: 'Other'
}
GENDER_MAP = {1: 'Male', 2: 'Female', 3: 'Other'}
SCALE_MAPPING = {"never": 1, "rarely": 2, "sometimes": 3, "often": 4, "always": 5}

IPIP_ITEMS = {
    "E1": "I am [blank] the life of the party.", "E2": "I [blank] don't talk a lot.",
    "E3": "I [blank] feel comfortable around people.", "E4": "I [blank] keep in the background.",
    "E5": "I [blank] start conversations.", "E6": "I [blank] have little to say.",
    "E7": "I [blank] talk to a lot of different people at parties.", "E8": "I [blank] don't like to draw attention to myself.",
    "E9": "I [blank] don't mind being the center of attention.", "E10": "I am [blank] quiet around strangers.",
    "N1": "I [blank] get stressed out easily.", "N2": "I am [blank] relaxed most of the time.",
    "N3": "I [blank] worry about things.", "N4": "I [blank] seldom feel blue.",
    "N5": "I am [blank] easily disturbed.", "N6": "I [blank] get upset easily.",
    "N7": "I [blank] change my mood a lot.", "N8": "I [blank] have frequent mood swings.",
    "N9": "I [blank] get irritated easily.", "N10": "I [blank] often feel blue.",
    "A1": "I [blank] feel little concern for others.", "A2": "I am [blank] interested in people.",
    "A3": "I [blank] insult people.", "A4": "I [blank] sympathize with others' feelings.",
    "A5": "I am [blank] not interested in other people's problems.", "A6": "I [blank] have a soft heart.",
    "A7": "I am [blank] not really interested in others.", "A8": "I [blank] take time out for others.",
    "A9": "I [blank] feel others' emotions.", "A10": "I [blank] make people feel at ease.",
    "C1": "I am [blank] always prepared.", "C2": "I [blank] leave my belongings around.",
    "C3": "I [blank] pay attention to details.", "C4": "I [blank] make a mess of things.",
    "C5": "I [blank] get chores done right away.", "C6": "I [blank] often forget to put things back in their proper place.",
    "C7": "I [blank] like order.", "C8": "I [blank] shirk my duties.",
    "C9": "I [blank] follow a schedule.", "C10": "I am [blank] exacting in my work.",
    "O1": "I [blank] have a rich vocabulary.", "O2": "I [blank] have difficulty understanding abstract ideas.",
    "O3": "I [blank] have a vivid imagination.", "O4": "I am [blank] not interested in abstract ideas.",
    "O5": "I [blank] have excellent ideas.", "O6": "I [blank] do not have a good imagination.",
    "O7": "I am [blank] quick to understand things.", "O8": "I [blank] use difficult words.",
    "O9": "I [blank] spend time reflecting on things.", "O10": "I am [blank] full of ideas."
}

# ==============================
# DATA LOADING
# ==============================
def load_and_clean_data():
    print("Loading and cleaning datasets...")
    df = pd.read_csv("human_data.csv", low_memory=False, sep='\t')
    iso_df = pd.read_csv("iso.csv")
    iso_map = dict(zip(iso_df['alpha-2'], iso_df['name']))
    df['country_name'] = df['country'].map(iso_map)

    questions = list(IPIP_ITEMS.keys())
    for col in questions + ['race', 'gender']:
        df = df[pd.to_numeric(df[col], errors='coerce') > 0]

    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df[(df['age'] >= 13) & (df['age'] <= 100)]
    df = df.dropna(subset=['country_name'])
    print(f"  {len(df)} rows available after cleaning.")
    return df

# ==============================
# BUILD ALL PROMPTS
# ==============================
def build_all_prompts(sampled_df):
    all_items = []
    for index, row in sampled_df.iterrows():
        age = int(row['age'])
        gender = GENDER_MAP[int(row['gender'])]
        race = RACE_MAP[int(row['race'])]
        country = row['country_name']
        persona_desc = f"{age}-year-old {gender} of {race} descent from {country}"

        for i in range(ITERATIONS_PER_ROW):
            sim_id = f"Qwen_P{index}_R{i}"
            for q_id, q_text in IPIP_ITEMS.items():
                prompt = (
                    f"You are roleplaying as a {persona_desc} taking a personality questionnaire.\n"
                    f"Fill in the [blank] in the following statement with exactly one word "
                    f"chosen from this list: never, rarely, sometimes, often, always.\n"
                    f"Output only the single word. Do not explain or add punctuation.\n\n"
                    f"Statement: {q_text}\n"
                    f"Answer:"
                )
                all_items.append({
                    "sim_id": sim_id,
                    "age": age,
                    "gender": gender,
                    "race": race,
                    "country": country,
                    "q_id": q_id,
                    "prompt": prompt,
                })
    return all_items

# ==============================
# BATCH INFERENCE
# ==============================
def run_batch(batch_items, model, tokenizer, device):
    prompts = [item["prompt"] for item in batch_items]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=8, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    results = []
    input_len = inputs["input_ids"].shape[1]
    for output in outputs:
        decoded = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip().lower()
        found = "sometimes"
        for token in decoded.replace(",", "").replace(".", "").replace("!", "").split():
            if token in VALID_WORDS:
                found = token
                break
        results.append(found)
    return results

# ==============================
# RECONSTRUCT ROWS
# ==============================
def reconstruct_rows(all_items, all_words):
    from collections import defaultdict
    groups = defaultdict(list)
    for item, word in zip(all_items, all_words):
        groups[item["sim_id"]].append((item, word))

    rows = []
    for sim_id, item_word_pairs in groups.items():
        first = item_word_pairs[0][0]
        entry = {
            "Sim_ID": sim_id,
            "Age": first["age"], "Gender": first["gender"],
            "Race": first["race"], "Country": first["country"],
        }
        for item, word in item_word_pairs:
            entry[f"{item['q_id']}_response"] = word
            entry[f"{item['q_id']}_score"] = SCALE_MAPPING.get(word, 3)
        rows.append(entry)
    return rows

# ==============================
# MAIN
# ==============================
def main():
    print(f"\n{'='*60}")
    print(f"Model: {MODEL_NAME}  |  Output: {OUTPUT_FILE}")
    print(f"Sample: {SAMPLE_SIZE} x {ITERATIONS_PER_ROW} iterations x 50 questions")
    print(f"Total calls: {SAMPLE_SIZE * ITERATIONS_PER_ROW * 50:,}")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device} ({torch.cuda.get_device_name(0)})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map={"": device}
    )
    model.eval()
    print(f"  VRAM used: {torch.cuda.memory_allocated(device)/1e9:.1f} GB\n")

    human_df = load_and_clean_data()
    sampled_df = human_df.sample(n=SAMPLE_SIZE, replace=True, random_state=RANDOM_SEED).reset_index(drop=True)

    print("Building prompts...")
    all_items = build_all_prompts(sampled_df)
    print(f"  {len(all_items):,} total prompts built\n")

    all_words = []
    num_batches = math.ceil(len(all_items) / BATCH_SIZE)
    print(f"Running inference in batches of {BATCH_SIZE}...")
    for i in tqdm(range(num_batches), desc="Batches"):
        batch = all_items[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        all_words.extend(run_batch(batch, model, tokenizer, device))

        if (i + 1) % 500 == 0:
            rows = reconstruct_rows(all_items[:len(all_words)], all_words)
            pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
            print(f"\n  Checkpoint at batch {i+1}")

    rows = reconstruct_rows(all_items, all_words)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone! {len(df_out)} rows saved to {OUTPUT_FILE}")

    score_cols = [c for c in df_out.columns if c.endswith('_score')]
    fallback_rate = (df_out[score_cols].values == 3).sum() / df_out[score_cols].size * 100
    print(f"Fallback rate: {fallback_rate:.1f}%")

if __name__ == "__main__":
    main()