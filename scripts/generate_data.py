import json
import os
import numpy as np
import openai
import re
from pathlib import Path
from dotenv import load_dotenv

import sys
script_dir = Path(__file__).parent.parent
sys.path.append(str(script_dir))

from utils.valuation_generation import generate_valuations
from utils.prompt_templates import create_system_prompt


# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configuration (could also load from config/config.yaml)
N_OBJECTS = 4
M_ROUNDS = 5
P_FRACTION = "1/4"
Q_FRACTION = "1/2"
BASE_RANGE = (-10, 10)
SCORE_THRESHOLD = 5
NUM_TRANSCRIPTS = 10  # per dataset

def parse_claims(claim_text):
    numbers = re.findall(r'\d+', claim_text)
    return [int(num)-1 for num in numbers]

def compute_payoffs(v1, v2, claims_agent1, claims_agent2, p=0.25, q=0.5):
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
    transcript = []
    agents = ["agent1", "agent2"]
    np.random.shuffle(agents)
    v1, v2 = valuations
    # Dialogue rounds
    for round_idx in range(m_rounds):
        for agent in agents:
            conversation = [{"role": "system", "content": system_prompt}]
            conversation.extend(transcript)
            current_valuation = v1 if agent == "agent1" else v2
            turn_prompt = f"Round {round_idx+1}: You are {agent}. Your valuation is: {np.array2string(current_valuation, precision=2)}. Please send your message."
            conversation.append({"role": "user", "content": turn_prompt})
            response = openai.ChatCompletion.create(
                model=model,
                messages=conversation,
                temperature=0.7
            )
            message_content = response['choices'][0]['message']['content']
            transcript.append({"role": agent, "content": message_content})
    # Final claims
    final_claims = {}
    for agent in agents:
        conversation = [{"role": "system", "content": system_prompt}]
        conversation.extend(transcript)
        claim_prompt = f"{agent}, please list the object numbers you claim."
        conversation.append({"role": "user", "content": claim_prompt})
        response = openai.ChatCompletion.create(
            model=model,
            messages=conversation,
            temperature=0.7
        )
        claim_text = response['choices'][0]['message']['content']
        final_claims[agent] = claim_text
        transcript.append({"role": agent, "content": claim_text})
    return transcript, final_claims

def filter_transcript(payoff1, payoff2, threshold=SCORE_THRESHOLD):
    return (payoff1 >= threshold) and (payoff2 >= threshold)

def save_transcript(dataset_name, transcript):
    out_dir = os.path.join("data", "finetuning")
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f"{dataset_name}.jsonl")
    with open(file_path, "a") as f:
        json.dump({"messages": transcript}, f)
        f.write("\n")

def main():
    # For each dataset type, generate transcripts
    dataset_settings = {
        "competitive": 0.0,
        "cooperative": 3.14159,
        "mixed": 1.5708
    }
    for dataset_name, theta in dataset_settings.items():
        print(f"Generating {NUM_TRANSCRIPTS} transcripts for {dataset_name} dataset...")
        for _ in range(NUM_TRANSCRIPTS):
            v1, v2 = generate_valuations(N_OBJECTS, base_range=BASE_RANGE, theta=theta)
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
