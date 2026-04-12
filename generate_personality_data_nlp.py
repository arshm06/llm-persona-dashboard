import asyncio
import pandas as pd
import math
import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Load API key
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_FILE = "simulated_human_data_nlp.csv" 
ITERATIONS_PER_ROW = 1
CONCURRENT_REQUESTS = 300

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

def load_and_clean_data():
    print("Loading and cleaning datasets...")
    df = pd.read_csv("simulated_human_data_isolated.csv", low_memory=False)
    return df


# ==============================
# SINGLE QUESTION CALL
# ==============================
async def ask_single_question(backstory, q_id, question_text, semaphore):
    system_prompt = (
        f"Here is your background context: '{backstory}'\n\n"
        "You are taking a personality test. Respond with EXACTLY one word from:\n"
        "[never, rarely, sometimes, often, always]\n\n"
        f"{q_id}: {question_text}\n\n"
        "Output format EXACTLY like:\n"
        f"{q_id}: word\n"
        "Do not add anything else."
    )

    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_prompt}],
                    temperature=0.7,
                    max_tokens=20
                )

                line = response.choices[0].message.content.strip()

                if ":" in line:
                    _, word = line.split(":", 1)
                    word = word.strip().lower().replace(".", "")
                    return word, SCALE_MAPPING.get(word, 3)

            except Exception:
                await asyncio.sleep(2 * (2 ** attempt))

        return "sometimes", 3


# ==============================
# MAIN PIPELINE (RESUME SAFE)
# ==============================
async def main():
    human_df = load_and_clean_data()
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    human_df = human_df.iloc[::3].reset_index(drop=True)
    total_rows = len(human_df)

    # 🔥 LOAD COMPLETED IDS (RESUME)
    if os.path.isfile(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        completed_ids = set(existing_df["Sim_ID"])
        print(f"Resuming: {len(completed_ids)} rows already completed.")
    else:
        completed_ids = set()

    with open("nlp_backstories_mapping.json", "r") as f:
        backstory_dict = json.load(f)

    BATCH_SIZE = 500
    total_batches = math.ceil(total_rows / BATCH_SIZE)

    print(f"\nProcessing {total_rows} rows across {total_batches} batches...")

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_df = human_df.iloc[batch_start : batch_start + BATCH_SIZE]

        print(f"\nBatch {batch_idx + 1}/{total_batches}")

        all_tasks = []
        metadata = []

        for index, row in batch_df.iterrows():
            profile_key = f"{int(row['Age'])}_{row['Gender']}_{row['Race']}_{row['Country']}"
            backstory = backstory_dict[profile_key]

            for i in range(ITERATIONS_PER_ROW):
                sim_id = f"NLP_P{index}_R{i}"

                # 🔥 SKIP IF ALREADY DONE
                if sim_id in completed_ids:
                    continue

                for q_id, question_text in IPIP_ITEMS.items():
                    task = ask_single_question(backstory, q_id, question_text, semaphore)
                    all_tasks.append(task)
                    metadata.append((sim_id, row, q_id))

        if not all_tasks:
            print("All rows in this batch already completed.")
            continue

        print(f"API calls this batch: {len(all_tasks)}")

        results = await tqdm.gather(*all_tasks)

        rows_dict = {}

        for (sim_id, row, q_id), (word, score) in zip(metadata, results):
            if sim_id not in rows_dict:
                rows_dict[sim_id] = {
                    "Sim_ID": sim_id,
                    "Age": int(row['Age']),
                    "Gender": row['Gender'],
                    "Race": row['Race'],
                    "Country": row['Country']
                }

            rows_dict[sim_id][f"{q_id}_response"] = word
            rows_dict[sim_id][f"{q_id}_score"] = score

        final_rows = list(rows_dict.values())

        file_exists = os.path.isfile(OUTPUT_FILE)
        pd.DataFrame(final_rows).to_csv(
            OUTPUT_FILE,
            mode='a',
            header=not file_exists,
            index=False
        )

        print(f"Batch {batch_idx + 1} saved.")

    print("\n✅ DONE")


if __name__ == "__main__":
    asyncio.run(main())