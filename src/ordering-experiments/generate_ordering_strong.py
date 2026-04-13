"""
Ordering experiment — strong models on a single A6000 (49GB VRAM).
Supports Mistral-Small-24B (float16, ~48GB) and DeepSeek-R1-32B (4-bit, ~20GB).

Usage:
    CUDA_VISIBLE_DEVICES=3 python generate_ordering_strong.py --model mistral
    CUDA_VISIBLE_DEVICES=3 python generate_ordering_strong.py --model deepseek32b
    CUDA_VISIBLE_DEVICES=3 python generate_ordering_strong.py --model deepseek14b
"""

import os
import math
import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from collections import defaultdict

# ==============================
# MODEL CONFIGS
# ==============================
MODEL_CONFIGS = {
    "mistral": {
        "path": "/playpen-ssd/shared/pretrained_models/Mistral-Small-24B-Instruct-2501",
        "output": "/playpen-ssd/yuvraj/llm-persona-dashboard/output/mistral_ordering.csv",
        "quantize_4bit": False,
        "max_new_tokens": 8,        # follows instructions cleanly
        "use_chat_template": True,  # Mistral handles chat templates well
    },
    "deepseek32b": {
        "path": "/playpen-ssd/shared/pretrained_models/DeepSeek-R1-Distill-Qwen-32B",
        "output": "/playpen-ssd/yuvraj/llm-persona-dashboard/output/deepseek32b_ordering.csv",
        "quantize_4bit": True,      # needs 4-bit to fit in 49GB
        "max_new_tokens": 64,       # R1 may think before answering
        "use_chat_template": False, # plain prompt more reliable for single-word output
    },
    "deepseek14b": {
        "path": "/playpen-ssd/shared/pretrained_models/DeepSeek-R1-Distill-Qwen-14B",
        "output": "/playpen-ssd/yuvraj/llm-persona-dashboard/output/deepseek14b_ordering.csv",
        "quantize_4bit": False,
        "max_new_tokens": 64,
        "use_chat_template": False,
    },
}

# ==============================
# CONFIG
# ==============================
SAMPLE_SIZE = 1000
BATCH_SIZE = 4          # conservative; increase to 8 if VRAM allows
RANDOM_SEED = 42

VALID_WORDS = {"never", "rarely", "sometimes", "often", "always"}
SCALE_MAPPING = {"never": 1, "rarely": 2, "sometimes": 3, "often": 4, "always": 5}

# ==============================
# MAPPINGS (unchanged from original)
# ==============================
RACE_MAP = {
    1: 'Mixed Race', 2: 'Arctic', 3: 'Caucasian (European)', 4: 'Caucasian (Indian)',
    5: 'Caucasian (Middle East)', 6: 'Caucasian (North African)', 7: 'Indigenous Australian',
    8: 'Native American', 9: 'North East Asian', 10: 'Pacific Islander',
    11: 'South East Asian', 12: 'West African', 13: 'Other'
}
GENDER_MAP = {1: 'Male', 2: 'Female', 3: 'Other'}

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
    df = pd.read_csv("/playpen-ssd/yuvraj/llm-persona-dashboard/data/human_data.csv", low_memory=False, sep='\t')
    iso_df = pd.read_csv("/playpen-ssd/yuvraj/llm-persona-dashboard/data/iso.csv")
    iso_map = dict(zip(iso_df['alpha-2'], iso_df['name']))
    df['country_name'] = df['country'].map(iso_map)

    questions = list(IPIP_ITEMS.keys())
    for col in questions + ['race', 'gender']:
        df = df[pd.to_numeric(df[col], errors='coerce') > 0]

    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df[(df['age'] >= 13) & (df['age'] <= 100)]
    df = df.dropna(subset=['country_name'])
    print(f"  {len(df)} rows after cleaning.")
    return df

# ==============================
# PROMPT BUILDING
# ==============================
def build_prompt(persona_desc, q_text, use_chat_template, tokenizer):
    """
    Mistral: use chat template (it's instruction-tuned to follow it).
    DeepSeek-R1: plain prompt is more reliable for constrained single-word output.
    """
    if use_chat_template:
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are roleplaying as a {persona_desc} taking a personality questionnaire. "
                    "Fill in the [blank] with exactly one word from: never, rarely, sometimes, often, always. "
                    "Output only that single word. No explanation, no punctuation."
                )
            },
            {"role": "user", "content": f"Statement: {q_text}\nAnswer:"}
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        return (
            f"You are roleplaying as a {persona_desc} taking a personality questionnaire.\n"
            f"Fill in the [blank] with exactly one word from: never, rarely, sometimes, often, always.\n"
            f"Output only that single word. No explanation, no punctuation.\n\n"
            f"Statement: {q_text}\n"
            f"Answer:"
        )

def build_all_prompts(sampled_df, use_chat_template, tokenizer, model_key):
    all_items = []
    for index, row in sampled_df.iterrows():
        age = int(row['age'])
        gender = GENDER_MAP[int(row['gender'])]
        race = RACE_MAP[int(row['race'])]
        country = row['country_name']

        for ordering_name, template in PROMPT_TEMPLATES.items():
            persona_desc = template.format(age=age, gender=gender, race=race, country=country)
            sim_id = f"{model_key}_P{index}_{ordering_name.replace(' ', '').replace('->', '_')}"

            for q_id, q_text in IPIP_ITEMS.items():
                prompt = build_prompt(persona_desc, q_text, use_chat_template, tokenizer)
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
def extract_valid_word(decoded_text):
    """
    Scan decoded output for the first valid frequency word.
    Handles DeepSeek chain-of-thought rambling before the answer.
    """
    cleaned = decoded_text.lower().replace(",", " ").replace(".", " ").replace("!", " ").replace("\n", " ")
    for token in cleaned.split():
        if token in VALID_WORDS:
            return token
    return "sometimes"  # fallback

def run_batch(batch_items, model, tokenizer, device, max_new_tokens):
    prompts = [item["prompt"] for item in batch_items]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    input_len = inputs["input_ids"].shape[1]
    results = []
    for output in outputs:
        new_tokens = output[input_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(extract_valid_word(decoded))
    return results

# ==============================
# RECONSTRUCT ROWS
# ==============================
def reconstruct_rows(all_items, all_words, model_name):
    groups = defaultdict(list)
    for item, word in zip(all_items, all_words):
        groups[item["sim_id"]].append((item, word))

    rows = []
    for sim_id, pairs in groups.items():
        first = pairs[0][0]
        entry = {
            "Sim_ID": sim_id,
            "Model": model_name,
            "Ordering": first["ordering"],
            "Age": first["age"],
            "Gender": first["gender"],
            "Race": first["race"],
            "Country": first["country"],
        }
        for item, word in pairs:
            entry[f"{item['q_id']}_response"] = word
            entry[f"{item['q_id']}_score"] = SCALE_MAPPING.get(word, 3)
        rows.append(entry)
    return rows

# ==============================
# SANITY CHECK
# ==============================
def sanity_check(model, tokenizer, device, use_chat_template, max_new_tokens):
    print("Running sanity check...")
    test_cases = [
        ("25-year-old Male of Caucasian (European) descent from Germany", "I am [blank] the life of the party.", "E1"),
        ("40-year-old Female of North East Asian descent from Japan",      "I [blank] worry about things.",          "N3"),
        ("30-year-old Male of West African descent from Nigeria",          "I am [blank] always prepared.",          "C1"),
    ]
    items = [
        {"prompt": build_prompt(p, q, use_chat_template, tokenizer), "q_id": qid}
        for p, q, qid in test_cases
    ]
    words = run_batch(items, model, tokenizer, device, max_new_tokens)
    all_ok = True
    for item, word in zip(items, words):
        ok = word in VALID_WORDS
        print(f"  {item['q_id']}: '{word}' {'✓' if ok else '✗ FALLBACK'}")
        if not ok:
            all_ok = False
    print("  All valid.\n" if all_ok else "\nWARNING: Fallbacks detected — check prompt format.\n")

# ==============================
# MAIN
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), required=True,
                        help="Which model to run")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    print(f"\n{'='*60}")
    print(f"Model key:  {args.model}")
    print(f"Model path: {cfg['path']}")
    print(f"Output:     {cfg['output']}")
    print(f"4-bit:      {cfg['quantize_4bit']} | Chat template: {cfg['use_chat_template']}")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # ----- Tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- Model -----
    if cfg["quantize_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,   # saves ~0.4 bits extra
            bnb_4bit_quant_type="nf4",         # best quality for LLMs
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            quantization_config=bnb_config,
            device_map={"": device},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.float16,
            device_map={"": device},
        )

    model.eval()
    print(f"VRAM used after load: {torch.cuda.memory_allocated(device)/1e9:.1f} GB\n")

    sanity_check(model, tokenizer, device, cfg["use_chat_template"], cfg["max_new_tokens"])

    # ----- Data -----
    human_df = load_and_clean_data()
    sampled_df = human_df.sample(n=SAMPLE_SIZE, replace=True, random_state=RANDOM_SEED).reset_index(drop=True)

    print("Building prompts...")
    all_items = build_all_prompts(sampled_df, cfg["use_chat_template"], tokenizer, args.model)
    print(f"  {len(all_items)} total inference calls\n")

    # ----- Inference -----
    all_words = []
    num_batches = math.ceil(len(all_items) / args.batch_size)

    for i in tqdm(range(num_batches), desc="Batches"):
        batch = all_items[i * args.batch_size: (i + 1) * args.batch_size]
        all_words.extend(run_batch(batch, model, tokenizer, device, cfg["max_new_tokens"]))

        if (i + 1) % 500 == 0:
            rows = reconstruct_rows(all_items[:len(all_words)], all_words, args.model)
            pd.DataFrame(rows).to_csv(cfg["output"], index=False)
            print(f"\n  Checkpoint saved at batch {i+1}")

    # ----- Save -----
    rows = reconstruct_rows(all_items, all_words, args.model)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(cfg["output"], index=False)
    print(f"\nSaved {len(df_out)} rows → {cfg['output']}")

    # ----- Diagnostics -----
    print("\nResponse distribution (sample items):")
    for q in ["E1", "A1", "C1", "N1", "O1"]:
        col = f"{q}_response"
        if col in df_out.columns:
            print(f"  {q}: {df_out[col].value_counts().to_dict()}")

    score_cols = [c for c in df_out.columns if c.endswith("_score")]
    fallback_rate = (df_out[score_cols].values.flatten() == 3).mean() * 100
    print(f"\nFallback rate (score=3): {fallback_rate:.1f}%")
    if fallback_rate > 30:
        print("WARNING: High fallback rate.")
    else:
        print("Fallback rate looks healthy.")

if __name__ == "__main__":
    main()