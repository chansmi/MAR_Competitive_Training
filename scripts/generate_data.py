import os
import json
import numpy as np
import re
import yaml

from openai import OpenAI
client = OpenAI()

from dotenv import load_dotenv

from utils.valuation_generation import generate_valuations
from utils.prompt_templates import create_system_prompt

# --- Load configuration from config/config.yaml ---
config_path = os.path.join("config", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

N_OBJECTS = config.get("n_objects", 4)
M_ROUNDS = config.get("m_rounds", 5)
P_FRACTION = config.get("p_fraction", "1/4")
Q_FRACTION = config.get("q_fraction", "1/2")
BASE_RANGE = config.get("valuation_range", [-10, 10])
SCORE_THRESHOLD = config.get("score_threshold", 5)
THETA_VALUES = config.get("theta_values", {"competitive": 0.0, "cooperative": 3.14159, "mixed": 1.5708})
NUM_TRANSCRIPTS = 10  # Number of transcripts to generate per dataset

# --- Set OpenAI API key ---
openai.api_key = os.getenv('OPENAI_API_KEY')

def parse_claims(claim_text):
    """Parse a claim message and return a list of 0-indexed object indices."""
    numbers = re.findall(r'\d+', claim_text)
    return [int(num) - 1 for num in numbers]

def compute_payoffs(v1, v2, claims_agent1, claims_agent2, p=0.25, q=0.5):
    """Compute the payoffs for two agents given their claim lists and valuation vectors."""
    payoff_agent1, payoff_agent2 = 0, 0
    for i in range(len(v1)):
        a1_claim = (i in claims_agent1)
        a2_claim = (i in claims_agent2)
        if a1_claim and not a2_claim:
            payoff_agent1 += v1[i]
        elif a2_claim and not a1_claim:
            payoff_agent2 += v2[i]
        elif a1_claim and a2_claim:
            payoff_agent1 += p * v1[i]
            payoff_agent2 += p * v2[i]
        else:
            payoff_agent1 += q * v1[i]
            payoff_agent2 += q * v2[i]
    return payoff_agent1, payoff_agent2

def generate_negotiation_transcript(system_prompt, valuations, m_rounds=M_ROUNDS, model="gpt-4o-mini-2024-07-18"):
    """
    Generates a negotiation transcript between two agents.
    It calls the Chat Completion API with `store=True` to mimic the updated behavior.
    """
    transcript = []
    agents = ["agent1", "agent2"]
    np.random.shuffle(agents)
    v1, v2 = valuations

    # --- Dialogue Rounds ---
    for round_idx in range(m_rounds):
        for agent in agents:
            conversation = [{"role": "system", "content": system_prompt}]
            conversation.extend(transcript)
            current_valuation = v1 if agent == "agent1" else v2
            turn_prompt = (
                f"Round {round_idx+1}: You are {agent}. Your valuation is: "
                f"{np.array2string(current_valuation, precision=2)}. Please send your message."
            )
            conversation.append({"role": "user", "content": turn_prompt})

            # Correctly call the Chat Completion API with store=True
            response = client.chat.completions.create(
                model="o1-mini-2024-09-12",
                store=True,
                messages=conversation,
                temperature=0.7,
            )
            message_content = response['choices'][0]['message']['content']
            transcript.append({"role": agent, "content": message_content})

    # --- Final Claim Round ---
    final_claims = {}
    for agent in agents:
        conversation = [{"role": "system", "content": system_prompt}]
        conversation.extend(transcript)
        claim_prompt = f"{agent}, please list the object numbers you claim."
        conversation.append({"role": "user", "content": claim_prompt})
        response = client.chat.completions.create(
            model="o1-mini-2024-09-12",
            store=True,
            messages=conversation,
            temperature=0.7,
        )
        claim_text = response['choices'][0]['message']['content']
        final_claims[agent] = claim_text
        transcript.append({"role": agent, "content": claim_text})

    return transcript, final_claims

def filter_transcript(payoff1, payoff2, threshold=SCORE_THRESHOLD):
    """Accept transcripts where both agents achieve at least the threshold payoff."""
    return (payoff1 >= threshold) and (payoff2 >= threshold)

def save_transcript(dataset_name, transcript):
    """Append a transcript (as JSON) to the appropriate dataset file."""
    out_dir = os.path.join("data", "finetuning")
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f"{dataset_name}.jsonl")
    with open(file_path, "a") as f:
        json.dump({"messages": transcript}, f)
        f.write("\n")

def main():
    dataset_settings = {
        "competitive": THETA_VALUES.get("competitive", 0.0),
        "cooperative": THETA_VALUES.get("cooperative", 3.14159),
        "mixed": THETA_VALUES.get("mixed", 1.5708)
    }
    for dataset_name, theta in dataset_settings.items():
        print(f"Generating {NUM_TRANSCRIPTS} transcripts for {dataset_name} dataset...")
        for _ in range(NUM_TRANSCRIPTS):
            v1, v2 = generate_valuations(N_OBJECTS, base_range=tuple(BASE_RANGE), theta=theta)
            system_prompt = create_system_prompt(
                n_objects=N_OBJECTS,
                v_agent=v1,
                m_rounds=M_ROUNDS,
                game_context=f"{dataset_name.capitalize()} Negotiation",
                p_fraction=P_FRACTION,
                q_fraction=Q_FRACTION
            )
            transcript, claims = generate_negotiation_transcript(system_prompt, (v1, v2))
            claims_agent1 = parse_claims(claims["agent1"])
            claims_agent2 = parse_claims(claims["agent2"])
            payoff1, payoff2 = compute_payoffs(v1, v2, claims_agent1, claims_agent2)
            if filter_transcript(payoff1, payoff2):
                save_transcript(dataset_name, transcript)
            else:
                print("Transcript filtered out due to low payoffs.")

if __name__ == '__main__':
    main()
