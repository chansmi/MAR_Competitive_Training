from pathlib import Path
import os
import json
import numpy as np
import re
import yaml
import math
import uuid
import datetime
import time
import argparse

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

# We override the number of objects with randomness: choose between 3 and 5.
# The number of rounds will also vary between 3 and 6.
P_FRACTION = config.get("p_fraction", "1/4")
Q_FRACTION = config.get("q_fraction", "1/2")
BASE_RANGE = config.get("valuation_range", [-10, 10])
NUM_TRANSCRIPTS = config.get("num_transcripts", 25) #Change

# Set API key for OpenAI client
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def parse_claims(claim_text):
    """Parse a claim message and return a list of 0-indexed object indices."""
    numbers = re.findall(r'\d+', claim_text)
    return [int(num) - 1 for num in numbers]

def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (norm1 * norm2) if norm1 and norm2 else 0

def compute_payoffs(v1, v2, claims_agent1, claims_agent2, p=0.25, q=0.5):
    """
    Compute raw payoffs over all objects, then discretize the outcome:
      - If agent1's raw payoff > agent2's, the discrete outcome is (1, -1)
      - If agent1's raw payoff < agent2's, the discrete outcome is (-1, 1)
      - Otherwise, (0, 0)
    Returns:
      (discrete_payoff_agent1, discrete_payoff_agent2, raw_payoff_agent1, raw_payoff_agent2)
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
        discrete = (1, -1)
    elif raw_payoff_agent1 < raw_payoff_agent2:
        discrete = (-1, 1)
    else:
        discrete = (0, 0)
    return discrete[0], discrete[1], raw_payoff_agent1, raw_payoff_agent2

def save_transcript(category, transcript, cos_sim, metadata):
    """Append a transcript (with enriched metadata as JSON) to the appropriate dataset file."""
    out_dir = Path("data") / "finetuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{category}.jsonl"
    data = {
        "cosine_similarity": cos_sim,
        "messages": transcript,
    }
    data.update(metadata)
    with file_path.open("a") as f:
        json.dump(data, f)
        f.write("\n")

def generate_negotiation_transcript(system_prompt, valuations, m_rounds, model="gpt-4o-mini-2024-07-18"):
    """
    Generates a negotiation transcript between two agents.
    This version alternates between user and assistant messages while preserving agent tracking.
    Each turn includes a user prompt (with a tag for the agent) and the assistant response.
    """
    transcript = []  # Each message is a dict with keys: role, content, speaker
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
            user_message = {"role": "user", "content": turn_prompt, "speaker": agent}
            conversation.append(user_message)
            transcript.append(user_message)
            
            response = client.chat.completions.create(
                model=model,
                store=True,
                messages=conversation,
                temperature=0.7,
            )
            assistant_message_text = response.choices[0].message.content
            assistant_message = {"role": "assistant", "content": f"[{agent}] {assistant_message_text}", "speaker": agent}
            transcript.append(assistant_message)

    # --- Final Claim Round ---
    final_claims = {}
    for agent in agents:
        conversation = [{"role": "system", "content": system_prompt}]
        conversation.extend(transcript)
        claim_prompt = f"{agent}, please list the object numbers you claim."
        user_claim = {"role": "user", "content": claim_prompt, "speaker": agent}
        conversation.append(user_claim)
        transcript.append(user_claim)
        response = client.chat.completions.create(
            model=model,
            store=True,
            messages=conversation,
            temperature=0.7,
        )
        claim_text = response.choices[0].message.content
        assistant_claim = {"role": "assistant", "content": f"[{agent}] {claim_text}", "speaker": agent}
        transcript.append(assistant_claim)
        final_claims[agent] = claim_text
    
    return transcript, final_claims

def filter_transcript(raw_payoff_agent1, raw_payoff_agent2, v1):
    """
    Filter transcripts based on agent1's performance.
    We compute a dynamic threshold for agent1 as:
        agent1_threshold = 0.5 * sum(max(v1[i], 0) for all i)
    """
    threshold = 0.5 * sum(max(val, 0) for val in v1)
    return raw_payoff_agent1 >= threshold

def main():
    """
    Main function that generates negotiation transcripts.
    Accepts an optional command-line argument to filter by scenario category.
    """
    parser = argparse.ArgumentParser(
        description="Generate negotiation transcripts."
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["cooperative", "conflict", "mixed", "all"],
        default="all",
        help="Generate only transcripts of the given category. (default: all)"
    )
    args = parser.parse_args()

    # If generating a specific category, track only that one.
    if args.category != "all":
        target = NUM_TRANSCRIPTS
        count = 0
    else:
        target = NUM_TRANSCRIPTS
        counts = {"cooperative": 0, "conflict": 0, "mixed": 0}

    # Prepare output directory and file(s).
    out_dir = Path("data") / "finetuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.category != "all":
        file_path = out_dir / f"{args.category}.jsonl"
        file_path.write_text("")  # Clear file
    else:
        for category in ["cooperative", "conflict", "mixed"]:
            file_path = out_dir / f"{category}.jsonl"
            file_path.write_text("")

    # Transcripts generation loop.
    while True:
        if args.category != "all" and count >= target:
            break
        if args.category == "all" and all(counts[cat] >= target for cat in counts):
            break

        # Choose a random number of objects between 3 and 5.
        num_objects = np.random.randint(3, 6)
        # Choose a random number of rounds between 3 and 6.
        m_rounds = np.random.randint(3, 7)
        
        # Generate a random theta in [0, π]
        theta = np.random.uniform(0, np.pi)
        # Generate valuations for the chosen number of objects.
        v1, v2 = generate_valuations(num_objects, tuple(BASE_RANGE), theta)
        
        # Compute cosine similarity between v1 and -v2.
        cos_sim = compute_cosine_similarity(v1, -v2)
        
        # Label the transcript.
        if cos_sim >= 0.90:
            label = "cooperative"
        elif cos_sim <= -0.90:
            label = "conflict"
        else:
            label = "mixed"
        
        # If a category is specified and this transcript isn't of that type, skip it.
        if args.category != "all" and label != args.category:
            continue
        
        system_prompt = create_system_prompt(
            n_objects=num_objects,
            v_agent=v1,
            m_rounds=m_rounds,
            game_context=f"{label.capitalize()} Negotiation",
            p_fraction=P_FRACTION,
            q_fraction=Q_FRACTION
        )
        
        transcript, claims = generate_negotiation_transcript(system_prompt, (v1, v2), m_rounds)
        agent1_claims = parse_claims(claims["agent1"])
        agent2_claims = parse_claims(claims["agent2"])
        discrete1, discrete2, raw1, raw2 = compute_payoffs(v1, v2, agent1_claims, agent2_claims)
        
        if filter_transcript(raw1, raw2, v1):
            transcript_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now().isoformat()
            metadata = {
                "transcript_id": transcript_id,
                "timestamp": timestamp,
                "v1": v1.tolist() if isinstance(v1, np.ndarray) else v1,
                "v2": v2.tolist() if isinstance(v2, np.ndarray) else v2,
                "theta": theta,
                "raw_payoffs": [raw1, raw2],
                "discrete_payoffs": [discrete1, discrete2],
                "num_objects": num_objects,
                "num_rounds": m_rounds
            }
            save_transcript(label, transcript, cos_sim, metadata)
            if args.category != "all":
                count += 1
                print(f"Saved {label} transcript {count} / {target} "
                      f"with raw payoffs: ({raw1}, {raw2}), cosine similarity: {cos_sim:.2f}")
            else:
                counts[label] += 1
                print(f"Saved {label} transcript {counts[label]} / {target} "
                      f"with raw payoffs: ({raw1}, {raw2}), cosine similarity: {cos_sim:.2f}")
        else:
            print("Transcript filtered out because agent1 did not do sufficiently well.")

if __name__ == '__main__':
    main()
