"""
Ordering experiment using local HuggingFace transformers.
No sudo, no Ollama needed. Runs Qwen2.5-3B-Instruct on a single GPU.

Usage:
    CUDA_VISIBLE_DEVICES=3 python generate_ordering_qwen.py
"""

import os
import math
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "simulated_ordering_experiment_qwen2.5-3b.csv"
SAMPLE_SIZE = 1000
BATCH_SIZE = 8
RANDOM_SEED = 42  # MUST match GPT-4o-mini run for fair comparison

VALID_WORDS = {"never", "rarely", "sometimes", "often", "always"}

# ==============================
# MAPPINGS
# ==============================
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

PROMPT_TEMPLATES = {
    "Age -> Gender -> Nationality": "{age}-year-old {gender} of {race} descent from {country}",
    "Gender -> Age -> Nationality": "{gender}, {age}-year-old from {country}, of {race} descent",
    "Nationality -> Age -> Gender": "From {country}, a {age}-year-old {gender} of {race} descent"
}

# ==============================
# DATA LOADING
# ==============================
def load_and_clean_data():
    print("Loading human data...")
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
# BUILD ALL PROMPTS UPFRONT
# ==============================
def build_all_prompts(sampled_df):
    all_items = []
    for index, row in sampled_df.iterrows():
        age = int(row['age'])
        gender = GENDER_MAP[int(row['gender'])]
        race = RACE_MAP[int(row['race'])]
        country = row['country_name']

        for ordering_name, template in PROMPT_TEMPLATES.items():
            persona_desc = template.format(age=age, gender=gender, race=race, country=country)
            sim_id = f"Qwen3B_P{index}_{ordering_name.replace(' ', '').replace('->', '_')}"

            for q_id, q_text in IPIP_ITEMS.items():
                # KEY FIX: single combined prompt instead of system/user chat split.
                # The chat template caused Qwen to output EOS immediately after the
                # system turn for many questions, producing '!!!!!' in the decoded output.
                # A plain instruction-style prompt is more reliable for single-word outputs.
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
                    "ordering": ordering_name,
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
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,          # greedy — deterministic
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,   # discourages repeating EOS tokens
        )

    results = []
    input_len = inputs["input_ids"].shape[1]
    for output in outputs:
        new_tokens = output[input_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

        # Scan for first valid word, strip any punctuation
        found = "sometimes"  # fallback
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
        first_item = item_word_pairs[0][0]
        entry = {
            "Sim_ID": sim_id,
            "Model": "Qwen2.5-3B-Instruct",
            "Ordering": first_item["ordering"],
            "Age": first_item["age"],
            "Gender": first_item["gender"],
            "Race": first_item["race"],
            "Country": first_item["country"],
        }
        for item, word in item_word_pairs:
            score = SCALE_MAPPING.get(word, 3)
            entry[f"{item['q_id']}_response"] = word
            entry[f"{item['q_id']}_score"] = score
        rows.append(entry)
    return rows

# ==============================
# SANITY CHECK — run before full run to catch issues early
# ==============================
def sanity_check(model, tokenizer, device):
    print("Running sanity check (3 examples)...")
    test_items = [
        {"prompt": "You are roleplaying as a 25-year-old Male of Caucasian (European) descent from Germany taking a personality questionnaire.\nFill in the [blank] in the following statement with exactly one word chosen from this list: never, rarely, sometimes, often, always.\nOutput only the single word. Do not explain or add punctuation.\n\nStatement: I am [blank] the life of the party.\nAnswer:", "q_id": "E1"},
        {"prompt": "You are roleplaying as a 40-year-old Female of North East Asian descent from Japan taking a personality questionnaire.\nFill in the [blank] in the following statement with exactly one word chosen from this list: never, rarely, sometimes, often, always.\nOutput only the single word. Do not explain or add punctuation.\n\nStatement: I [blank] worry about things.\nAnswer:", "q_id": "N3"},
        {"prompt": "You are roleplaying as a 30-year-old Male of West African descent from Nigeria taking a personality questionnaire.\nFill in the [blank] in the following statement with exactly one word chosen from this list: never, rarely, sometimes, often, always.\nOutput only the single word. Do not explain or add punctuation.\n\nStatement: I am [blank] always prepared.\nAnswer:", "q_id": "C1"},
    ]
    words = run_batch(test_items, model, tokenizer, device)
    all_ok = True
    for item, word in zip(test_items, words):
        status = "✓" if word in VALID_WORDS else "✗ FALLBACK"
        print(f"  {item['q_id']}: '{word}' {status}")
        if word not in VALID_WORDS:
            all_ok = False

    if not all_ok:
        print("\nWARNING: Some responses falling back to 'sometimes'. Check prompt format.\n")
    else:
        print("  All valid. Proceeding with full run.\n")
    return all_ok

# ==============================
# MAIN
# ==============================
def main():
    print(f"\n{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Sample size: {SAMPLE_SIZE} | Seed: {RANDOM_SEED}")
    print(f"{'='*60}\n")

    print("Loading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device} ({torch.cuda.get_device_name(0)})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    model.eval()
    print(f"  Model loaded. VRAM used: {torch.cuda.memory_allocated(device)/1e9:.1f} GB\n")

    # Run sanity check before committing to full run
    sanity_check(model, tokenizer, device)

    human_df = load_and_clean_data()
    sampled_df = human_df.sample(n=SAMPLE_SIZE, replace=True, random_state=RANDOM_SEED).reset_index(drop=True)

    print("Building prompts...")
    all_items = build_all_prompts(sampled_df)
    total = len(all_items)
    print(f"  {total} total calls ({SAMPLE_SIZE} personas x 3 orderings x 50 questions)\n")

    all_words = []
    num_batches = math.ceil(total / BATCH_SIZE)

    print(f"Running inference in batches of {BATCH_SIZE}...")
    for i in tqdm(range(num_batches), desc="Batches"):
        batch = all_items[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        words = run_batch(batch, model, tokenizer, device)
        all_words.extend(words)

        if (i + 1) % 500 == 0:
            rows = reconstruct_rows(all_items[:len(all_words)], all_words)
            pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
            print(f"\n  Checkpoint saved at batch {i+1}")

    print("\nReconstructing output rows...")
    rows = reconstruct_rows(all_items, all_words)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone! {len(df_out)} rows saved to {OUTPUT_FILE}")

    # Final diagnostics
    print("\nResponse distribution sample:")
    for q in ["E1", "A1", "C1", "N1", "O1"]:
        col = f"{q}_response"
        if col in df_out.columns:
            print(f"  {q}: {df_out[col].value_counts().to_dict()}")

    score_cols = [c for c in df_out.columns if c.endswith('_score')]
    all_scores = df_out[score_cols].values.flatten()
    fallback_rate = (all_scores == 3).sum() / len(all_scores) * 100
    print(f"\nOverall fallback rate (score=3): {fallback_rate:.1f}%")
    if fallback_rate > 30:
        print("WARNING: High fallback rate — model may not be following instructions well.")
    else:
        print("Fallback rate looks healthy.")

if __name__ == "__main__":
    main()