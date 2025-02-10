from pathlib import Path
import os
import json
import numpy as np
import re
import yaml
import math

from openai import OpenAI
client = OpenAI()

from dotenv import load_dotenv
load_dotenv()

from utils.valuation_generation import generate_valuations
from utils.prompt_templates import create_system_prompt

# Load configuration from config/config.yaml using Path
config_path = Path("config") / "config.yaml"
with config_path.open("r") as f:
    config = yaml.safe_load(f)

N_OBJECTS = config.get("n_objects", 4)
M_ROUNDS = config.get("m_rounds", 5)
P_FRACTION = config.get("p_fraction", "1/4")
Q_FRACTION = config.get("q_fraction", "1/2")
BASE_RANGE = config.get("valuation_range", [-10, 10])
NUM_TRANSCRIPTS = config.get("num_transcripts", 10)  # number of strong transcripts per category

# Set API key for OpenAI client
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def parse_claims(claim_text):
    """Parse a claim message and return a list of 0-indexed object indices."""
    numbers = re.findall(r'\d+', claim_text)
    return [int(num) - 1 for num in numbers]

def compute_payoffs(v1, v2, claims_agent1, claims_agent2, p=0.25, q=0.5):
    """
    Compute raw payoffs over all objects, and then discretize the outcome:
      - If agent1's raw payoff > agent2's, return (1, -1)
      - If agent1's raw payoff < agent2's, return (-1, 1)
      - Otherwise, return (0, 0)
    """
    raw_payoff_agent1 = 0
    raw_payoff_agent2 = 0
    for i in range(len(v1)):
        a1_claim = (i in claims_agent1)
        a2_claim = (i in claims_agent2)
        if a1_claim and not a2_claim:
            raw_payoff_agent1 += v1[i]
        elif a2_claim and not a1_claim:
            raw_payoff_agent2 += v2[i]
        elif a1_claim and a2_claim:
            raw_payoff_agent1 += p * v1[i]
            raw_payoff_agent2 += p * v2[i]
        else:
            raw_payoff_agent1 += q * v1[i]
            raw_payoff_agent2 += q * v2[i]
    if raw_payoff_agent1 > raw_payoff_agent2:
        return 1, -1
    elif raw_payoff_agent1 < raw_payoff_agent2:
        return -1, 1
    else:
        return 0, 0


def save_transcript(category, transcript, cos_sim):
    """Append a transcript (as JSON) to the appropriate dataset file."""
    out_dir = Path("data") / "finetuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{category}.jsonl"
    with file_path.open("a") as f:
        json.dump({"cosine_similarity": cos_sim, "messages": transcript}, f)
        f.write("\n")

def generate_negotiation_transcript(system_prompt, valuations, m_rounds=M_ROUNDS, model="gpt-4o-mini-2024-07-18"):
    """
    Generates a negotiation transcript between two agents.
    It calls the Chat Completion API with store=True to mimic the updated behavior.
    """
    transcript = []
    agents = ["agent1", "agent2"]
    np.random.shuffle(agents)
    v1, v2 = valuations

    # --- Dialogue Rounds ---
    for round_idx in range(m_rounds):
        for agent in agents:
            # Use the system message as intended since gpt-4o-mini-2024-07-18 supports it.
            conversation = [{"role": "system", "content": system_prompt}]
            # Include previous transcript messages as assistant responses.
            for msg in transcript:
                conversation.append({"role": "assistant", "content": msg["content"]})
            current_valuation = v1 if agent == "agent1" else v2
            turn_prompt = (
                f"Round {round_idx+1}: You are {agent}. Your valuation is: "
                f"{np.array2string(current_valuation, precision=2)}. Please send your message."
            )
            conversation.append({"role": "user", "content": turn_prompt})
            
            response = client.chat.completions.create(
                model=model,
                store=True,
                messages=conversation,
                temperature=0.7,
            )
            message_content = response.choices[0].message.content
            # Save with an accepted role and include the agent identifier in the content.
            transcript.append({"role": "assistant", "content": f"[{agent}] {message_content}"})

    # --- Final Claim Round ---
    final_claims = {}
    for agent in agents:
        conversation = [{"role": "system", "content": system_prompt}]
        for msg in transcript:
            conversation.append({"role": "assistant", "content": msg["content"]})
        claim_prompt = f"{agent}, please list the object numbers you claim."
        conversation.append({"role": "user", "content": claim_prompt})
        
        response = client.chat.completions.create(
            model=model,
            store=True,
            messages=conversation,
            temperature=0.7,
        )
        claim_text = response.choices[0].message.content
        final_claims[agent] = claim_text
        transcript.append({"role": "assistant", "content": f"[{agent}] {claim_text}"})
    
    return transcript, final_claims

def filter_transcript(payoff1, payoff2):
    """
    With our new discrete payoffs (only 1, 0, or -1), every transcript is acceptable.
    """
    return True

def main():
    """
    Main function that generates negotiation transcripts.
    It dynamically generates valuation vectors with a random theta,
    computes the cosine similarity between v1 and -v2 (which equals -cos(theta)),
    and labels the transcript as 'cooperative', 'conflict', or 'mixed' based on strong thresholds:
      - Cooperative: cosine similarity >= 0.90 (theta near pi)
      - Conflict: cosine similarity <= -0.90 (theta near 0)
      - Mixed: absolute cosine similarity <= 0.1 (theta near pi/2)
    Transcripts not meeting these criteria are skipped.
    The accepted transcripts are saved as JSONL files in the "data/finetuning" directory
    and later used for supervised fine-tuning (SFT).
    """
    target = NUM_TRANSCRIPTS
    counts = {"cooperative": 0, "conflict": 0, "mixed": 0}
    
    # Clear output files
    out_dir = Path("data") / "finetuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    for category in counts:
        file_path = out_dir / f"{category}.jsonl"
        file_path.write_text("")
    
    while any(counts[cat] < target for cat in counts):
        # Generate a random theta in [0, pi]
        theta = np.random.uniform(0, np.pi)
        v1, v2 = generate_valuations(N_OBJECTS, tuple(BASE_RANGE), theta)
        
        # Cosine similarity between v1 and -v2 (given our generation method) is -cos(theta)
        cos_sim = -np.cos(theta)
        
        label = None
        if cos_sim >= 0.90:
            label = "cooperative"
        elif cos_sim <= -0.90:
            label = "conflict"
        elif abs(cos_sim) <= 0.1:
            label = "mixed"
        else:
            continue  # Skip non-strong examples
        
        # Only generate if we still need examples for this label
        if counts[label] >= target:
            continue
        
        system_prompt = create_system_prompt(
            n_objects=N_OBJECTS,
            v_agent=v1,
            m_rounds=M_ROUNDS,
            game_context=f"{label.capitalize()} Negotiation",
            p_fraction=P_FRACTION,
            q_fraction=Q_FRACTION
        )
        
        transcript, claims = generate_negotiation_transcript(system_prompt, (v1, v2))
        agent1_claims = parse_claims(claims["agent1"])
        agent2_claims = parse_claims(claims["agent2"])
        payoff1, payoff2 = compute_payoffs(v1, v2, agent1_claims, agent2_claims)
        
        if filter_transcript(payoff1, payoff2):
            save_transcript(label, transcript, cos_sim)
            counts[label] += 1
            print(f"Saved {label} transcript {counts[label]} / {target} with discrete payoffs: ({payoff1}, {payoff2}), cosine similarity: {cos_sim:.2f}")
        else:
            print("Transcript filtered out due to low payoffs.")

if __name__ == '__main__':
    main()