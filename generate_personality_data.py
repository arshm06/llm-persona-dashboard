import asyncio
import pandas as pd
from openai import AsyncOpenAI
import os
from tqdm.asyncio import tqdm


client = AsyncOpenAI()

OUTPUT_FILE = "all_personality_data.csv"
ITERATIONS = 15


race_map = {
    1: 'Mixed', 2: 'Arctic', 3: 'Caucasian (Euro)', 4: 'Caucasian (Ind)',
    5: 'Caucasian (MidEast)', 6: 'Caucasian (NorthAf)', 7: 'Indigenous Aus',
    8: 'Native Amer', 9: 'NE Asian', 10: 'Pacific', 11: 'SE Asian',
    12: 'W African', 13: 'Other'
}

age_map = {
    "<18": "a person under 18 years old",
    "18-25": "a person aged 18-25",
    "35-50": "a person aged 35-50",
    "50+": "a person over 50 years old"
}

target_personas = []

for r in race_map.values():
    target_personas.append({"label": r, "prompt": r, "category": "Race"})
# Add Genders
for g in ["Male", "Female"]:
    target_personas.append({"label": g, "prompt": g, "category": "Gender"})
# Add Ages
for label, prompt in age_map.items():
    target_personas.append({"label": label, "prompt": prompt, "category": "Age"})

# ==============================
# IPIP ITEMS & SCORING
# ==============================
SCALE_MAPPING = {"never": 1, "rarely": 2, "sometimes": 3, "often": 4, "always": 5}

IPIP_ITEMS = [
    {"id": 1, "text": "I am [blank] the life of the party.", "trait": "Extroversion", "sign": 1},
    {"id": 6, "text": "I [blank] don't talk a lot.", "trait": "Extroversion", "sign": -1},
    {"id": 11, "text": "I [blank] feel comfortable around people.", "trait": "Extroversion", "sign": 1},
    {"id": 16, "text": "I [blank] keep in the background.", "trait": "Extroversion", "sign": -1},
    {"id": 21, "text": "I [blank] start conversations.", "trait": "Extroversion", "sign": 1},
    {"id": 26, "text": "I [blank] have little to say.", "trait": "Extroversion", "sign": -1},
    {"id": 31, "text": "I [blank] talk to a lot of different people at parties.", "trait": "Extroversion", "sign": 1},
    {"id": 36, "text": "I [blank] don't like to draw attention to myself.", "trait": "Extroversion", "sign": -1},
    {"id": 41, "text": "I [blank] don't mind being the center of attention.", "trait": "Extroversion", "sign": 1},
    {"id": 46, "text": "I am [blank] quiet around strangers.", "trait": "Extroversion", "sign": -1},

    {"id": 2, "text": "I [blank] feel little concern for others.", "trait": "Agreeableness", "sign": -1},
    {"id": 7, "text": "I am [blank] interested in people.", "trait": "Agreeableness", "sign": 1},
    {"id": 12, "text": "I [blank] insult people.", "trait": "Agreeableness", "sign": -1},
    {"id": 17, "text": "I [blank] sympathize with others' feelings.", "trait": "Agreeableness", "sign": 1},
    {"id": 22, "text": "I am [blank] not interested in other people's problems.", "trait": "Agreeableness", "sign": -1},
    {"id": 27, "text": "I [blank] have a soft heart.", "trait": "Agreeableness", "sign": 1},
    {"id": 32, "text": "I am [blank] not really interested in others.", "trait": "Agreeableness", "sign": -1},
    {"id": 37, "text": "I [blank] take time out for others.", "trait": "Agreeableness", "sign": 1},
    {"id": 42, "text": "I [blank] feel others' emotions.", "trait": "Agreeableness", "sign": 1},
    {"id": 47, "text": "I [blank] make people feel at ease.", "trait": "Agreeableness", "sign": 1},

    {"id": 3, "text": "I am [blank] prepared.", "trait": "Conscientiousness", "sign": 1},
    {"id": 8, "text": "I [blank] leave my belongings around.", "trait": "Conscientiousness", "sign": -1},
    {"id": 13, "text": "I [blank] pay attention to details.", "trait": "Conscientiousness", "sign": 1},
    {"id": 18, "text": "I [blank] make a mess of things.", "trait": "Conscientiousness", "sign": -1},
    {"id": 23, "text": "I [blank] get chores done right away.", "trait": "Conscientiousness", "sign": 1},
    {"id": 28, "text": "I [blank] forget to put things back in their proper place.", "trait": "Conscientiousness", "sign": -1},
    {"id": 33, "text": "I [blank] like order.", "trait": "Conscientiousness", "sign": 1},
    {"id": 38, "text": "I [blank] shirk my duties.", "trait": "Conscientiousness", "sign": -1},
    {"id": 43, "text": "I [blank] follow a schedule.", "trait": "Conscientiousness", "sign": 1},
    {"id": 48, "text": "I am [blank] exacting in my work.", "trait": "Conscientiousness", "sign": 1},

    {"id": 4, "text": "I [blank] get stressed out easily.", "trait": "Emotional Stability", "sign": -1},
    {"id": 9, "text": "I am [blank] relaxed most of the time.", "trait": "Emotional Stability", "sign": 1},
    {"id": 14, "text": "I [blank] worry about things.", "trait": "Emotional Stability", "sign": -1},
    {"id": 19, "text": "I [blank] seldom feel blue.", "trait": "Emotional Stability", "sign": 1},
    {"id": 24, "text": "I am [blank] easily disturbed.", "trait": "Emotional Stability", "sign": -1},
    {"id": 29, "text": "I [blank] get upset easily.", "trait": "Emotional Stability", "sign": -1},
    {"id": 34, "text": "I [blank] change my mood a lot.", "trait": "Emotional Stability", "sign": -1},
    {"id": 39, "text": "I [blank] have frequent mood swings.", "trait": "Emotional Stability", "sign": -1},
    {"id": 44, "text": "I [blank] get irritated easily.", "trait": "Emotional Stability", "sign": -1},
    {"id": 49, "text": "I [blank] feel blue.", "trait": "Emotional Stability", "sign": -1},

    {"id": 5, "text": "I [blank] have a rich vocabulary.", "trait": "Openness", "sign": 1},
    {"id": 10, "text": "I [blank] have difficulty understanding abstract ideas.", "trait": "Openness", "sign": -1},
    {"id": 15, "text": "I [blank] have a vivid imagination.", "trait": "Openness", "sign": 1},
    {"id": 20, "text": "I am [blank] not interested in abstract ideas.", "trait": "Openness", "sign": -1},
    {"id": 25, "text": "I [blank] have excellent ideas.", "trait": "Openness", "sign": 1},
    {"id": 30, "text": "I [blank] do not have a good imagination.", "trait": "Openness", "sign": -1},
    {"id": 35, "text": "I am [blank] quick to understand things.", "trait": "Openness", "sign": 1},
    {"id": 40, "text": "I [blank] use difficult words.", "trait": "Openness", "sign": 1},
    {"id": 45, "text": "I [blank] spend time reflecting on things.", "trait": "Openness", "sign": 1},
    {"id": 50, "text": "I am [blank] full of ideas.", "trait": "Openness", "sign": 1},
]

def calculate_score(response_word, sign):
    clean = response_word.replace(".", "").strip().lower()
    score = SCALE_MAPPING.get(clean, 3)
    return score if sign == 1 else (6 - score)

async def get_llm_response(persona_prompt, question_text, semaphore):
    system_prompt = (
        f"You are {persona_prompt}. "
        "You are taking a personality test. "
        "Fill in the [blank] with exactly one word from this list: "
        "[never, rarely, sometimes, often, always]. "
        "Do not explain. Output only the word."
    )
    user_prompt = f"Statement: {question_text}\nAnswer:"
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=0.7, 
                max_tokens=5
            )
            return response.choices[0].message.content.strip().lower()
        except:
            return "sometimes"

async def run_full_test(persona_config, iteration_id, semaphore):
    label = persona_config['label']
    prompt = persona_config['prompt']
    category = persona_config['category']

    tasks = [get_llm_response(prompt, item["text"], semaphore) for item in IPIP_ITEMS]
    responses = await asyncio.gather(*tasks)

    rows = []
    for item, word in zip(IPIP_ITEMS, responses):
        score = calculate_score(word, item["sign"])
        rows.append({
            "Iteration_ID": iteration_id,
            "Category": category,
            "Persona": label,
            "Trait": item["trait"],
            "Question": item["text"],
            "Response": word,
            "Score": score
        })
    return rows

async def main():
    semaphore = asyncio.Semaphore(10)
    all_data = []

    print(f"Generating data for {len(target_personas)} subgroups.")
    print(f"Iterations per subgroup: {ITERATIONS}")

    for persona in target_personas:
        desc = f"{persona['label']} ({persona['category']})"
        print(f"Starting: {desc}")

        iteration_tasks = []
        for i in range(ITERATIONS):
            iteration_tasks.append(run_full_test(persona, i, semaphore))

        results = await asyncio.gather(*iteration_tasks)

        for res in results:
            all_data.extend(res)

    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Saved {len(df)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())