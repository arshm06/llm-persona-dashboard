import asyncio
import pandas as pd
import math
import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Load the API key from your .env file
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CHANGED: New output file for NLP data
OUTPUT_FILE = "simulated_human_data_nlp.csv" 
ITERATIONS_PER_ROW = 1
CONCURRENT_REQUESTS = 100

# ==============================
# MAPPINGS & ITEMS
# ==============================


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
# DATA CLEANING
# ==============================
def load_and_clean_data():
    print("Loading and cleaning datasets...")
    df = pd.read_csv("simulated_human_data_isolated.csv", low_memory=False, sep=',')    
    return df

# ==============================
# PHASE 1: NLP BACKSTORY GENERATOR
# ==============================
async def generate_single_backstory(age, gender, race, country, semaphore):
    system_prompt = f"""Generate a very concise realistic, 2-to-3 sentence, first-person backstory for a specific demographic profile.

CRITICAL RULES:
Make sure to explicitly include the specific trait of that person. So if they are a male from a certain country, make sure to include that they are a male from that country. 
Use cultural touchstones, generational life stages, geographic references, and socio-linguistic markers to hint at who they are.
Ground the description in mundane reality. Do not use extreme or offensive stereotypes.
Output ONLY the 2-3 sentences of the backstory. No introductory text. Should not exceed 130 tokens.

INPUT: Age: {age}, Gender: {gender}, Race: {race}, Country: {country}
"""
    
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_prompt}],
                    temperature=0.8,
                    max_tokens=150
                )
                return response.choices[0].message.content.strip()
            except Exception:
                await asyncio.sleep(2 * (2 ** attempt))
        return "I am a typical person living a normal life, trying to balance my daily responsibilities."

async def build_backstory_dictionary(unique_profiles_df, semaphore):
    print(f"Generating Natural Language Profiles for {len(unique_profiles_df)} unique demographic combinations...")
    backstory_dict = {}
    tasks = []
    keys = []
    
    for _, row in unique_profiles_df.iterrows():
        age = row['Age']
        gender = row['Gender']
        race = row['Race']
        country = row['Country']
        
        profile_key = f"{age}_{gender}_{race}_{country}"
        keys.append(profile_key)
        tasks.append(generate_single_backstory(age, gender, race, country, semaphore))
        
    results = await tqdm.gather(*tasks)
    
    for key, backstory in zip(keys, results):
        backstory_dict[key] = backstory
        
    return backstory_dict

# ==============================
# PHASE 2: ISOLATED ASYNC LLM CALL
# ==============================
async def ask_questions_batch(backstory, semaphore):
    questions_text = "\n".join(
        [f"{k}: {v}" for k, v in IPIP_ITEMS.items()]
    )

    system_prompt = (
        f"Here is your background context: '{backstory}'\n\n"
        "You are taking a personality test. For EACH statement, respond with EXACTLY one word from:\n"
        "[never, rarely, sometimes, often, always]\n\n"
        f"{questions_text}\n\n"
        "Output format EXACTLY like:\n"
        "E1: word\nE2: word\n...\nO10: word\n"
        "Do not add anything else."
    )

    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_prompt}],
                    temperature=0.7,
                    max_tokens=300
                )

                lines = response.choices[0].message.content.strip().split("\n")

                result = {}
                for line in lines:
                    if ":" in line:
                        q, word = line.split(":", 1)
                        word = word.strip().lower().replace(".", "")
                        result[q.strip()] = (
                            word,
                            SCALE_MAPPING.get(word, 3)
                        )

                return result

            except Exception:
                await asyncio.sleep(2 * (2 ** attempt))

        # fallback
        return {k: ("sometimes", 3) for k in IPIP_ITEMS.keys()}

# ==============================
# MAIN PIPELINE
# ==============================
# ==============================
# MAIN PIPELINE
# ==============================
async def main():
    # 1. Load the ENTIRE dataset (no sampling)
    human_df = load_and_clean_data()    
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    human_df = human_df.iloc[::3].reset_index(drop=True)
    total_rows = len(human_df)
    with open("nlp_backstories_mapping.json", "r") as f:
        backstory_dict = json.load(f)

    
    # 3. Setup batching for the entire dataset
    BATCH_SIZE = 500 
    total_batches = math.ceil(total_rows / BATCH_SIZE)
    print(f"\nExecuting questionnaire requests for all {total_rows} rows in {total_batches} batches...")

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        # Slice from the full dataset instead of a sample
        batch_df = human_df.iloc[batch_start : batch_start + BATCH_SIZE]
        all_tasks = []

        # Build the tasks for this specific batch
        for index, row in batch_df.iterrows():
            # Construct the key to fetch the specific backstory
            profile_key = f"{int(row['Age'])}_{row['Gender']}_{row['Race']}_{row['Country']}"
            backstory = backstory_dict[profile_key]
            
            for i in range(ITERATIONS_PER_ROW):
                # Using the actual index from the human_df to link them directly
                sim_id = f"NLP_P{index}_R{i}"
                all_tasks.append((sim_id, row, ask_questions_batch(backstory, semaphore)))
        print(f"\nProcessing Batch {batch_idx + 1} / {total_batches} ({len(all_tasks)} requests)")
        
        # Execute batch
        raw_results = await tqdm.gather(*(t[2] for t in all_tasks))
        
        # Reconstruct rows
        final_rows = []
        for (sim_id, original_row, _), answers in zip(all_tasks, raw_results):
            
            entry = {
                "Sim_ID": sim_id,
                "Age": int(original_row['Age']),
                "Gender": original_row['Gender'],
                "Race": original_row['Race'],
                "Country": original_row['Country']
            }

            for q_id, (word, score) in answers.items():
                entry[f"{q_id}_response"] = word
                entry[f"{q_id}_score"] = score

            final_rows.append(entry)

        # Append to CSV
        file_exists = os.path.isfile(OUTPUT_FILE)
        pd.DataFrame(final_rows).to_csv(
            OUTPUT_FILE, 
            mode='a', 
            header=not file_exists, 
            index=False
        )
        print(f"Batch {batch_idx + 1} saved successfully. Moving to next...")

    print(f"\nSuccess! All {total_rows} original rows mapped to NLP personas and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())