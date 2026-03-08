import asyncio
import pandas as pd
import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Load the API key from your .env file
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_FILE = "simulated_human_data_isolated.csv"
SAMPLE_SIZE = 1000
ITERATIONS_PER_ROW = 3
CONCURRENT_REQUESTS = 100

# ==============================
# MAPPINGS & ITEMS
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

# ==============================
# DATA CLEANING
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
    
    print(f"Cleaned data rows available: {len(df)}")
    return df

# ==============================
# ISOLATED ASYNC LLM CALL
# ==============================
async def ask_single_question(persona_desc, q_id, q_text, semaphore):
    system_prompt = (
        f"You are a {persona_desc}. You are taking a personality test. "
        "Fill in the [blank] with exactly one word from this list: "
        "[never, rarely, sometimes, often, always]. "
        "Do not explain. Output only the word."
    )
    user_prompt = f"Statement: {q_text}\nAnswer:"
    
    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=5
                )
                word = response.choices[0].message.content.strip().lower().replace(".", "")
                return q_id, word, SCALE_MAPPING.get(word, 3)
            except Exception:
                await asyncio.sleep(2 * (2 ** attempt)) # Exponential backoff
        return q_id, "sometimes", 3

async def main():
    human_df = load_and_clean_data() # Ensure this function is in your script
    sampled_df = human_df.sample(n=SAMPLE_SIZE, replace=True).reset_index(drop=True)
    
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    all_tasks = []

    # Flatten the experiment: Every question for every iteration for every persona
    for index, row in sampled_df.iterrows():
        persona_desc = f"{int(row['age'])}-year-old {GENDER_MAP[row['gender']]} of {RACE_MAP[row['race']]} descent from {row['country_name']}"
        for i in range(ITERATIONS_PER_ROW):
            sim_id = f"P{index}_R{i}"
            for q_id, q_text in IPIP_ITEMS.items():
                all_tasks.append((sim_id, row, ask_single_question(persona_desc, q_id, q_text, semaphore)))

    print(f"Launching {len(all_tasks)} isolated requests...")
    
    # Run with progress bar
    results = []
    # We gather the raw tasks
    raw_results = await tqdm.gather(*(t[2] for t in all_tasks))
    
    # Reconstruct the rows from the flat result list
    # (Since there are 50 questions per iteration, we chunk the results)
    final_rows = []
    for i in range(0, len(all_tasks), 50):
        chunk = raw_results[i : i + 50]
        sim_id, original_row, _ = all_tasks[i]
        
        entry = {
            "Sim_ID": sim_id,
            "Age": int(original_row['age']),
            "Gender": GENDER_MAP[original_row['gender']],
            "Race": RACE_MAP[original_row['race']],
            "Country": original_row['country_name']
        }
        for q_id, word, score in chunk:
            entry[f"{q_id}_response"] = word
            entry[f"{q_id}_score"] = score
        final_rows.append(entry)

    pd.DataFrame(final_rows).to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())