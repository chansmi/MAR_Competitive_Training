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
SCORE_THRESHOLD = config.get("score_threshold", 5)
NUM_TRANSCRIPTS = config.get("num_transcripts", 10)  # number of strong transcripts per category

# Set API key for OpenAI client
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

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
            
            response = client.chat.completions.create(
                model=model,
                store=True,
                messages=conversation,
                temperature=0.7,
            )
            message_content = response.choices[0].message.content
            transcript.append({"role": "assistant", "content": f"[{agent}] {message_content}"})

    # --- Final Claim Round ---
    final_claims = {}
    for agent in agents:
        conversation = [{"role": "system", "content": system_prompt}]
        conversation.extend(transcript)
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

def filter_transcript(payoff1, payoff2, threshold=SCORE_THRESHOLD):
    """Accept transcripts where both agents achieve at least the threshold payoff."""
    return (payoff1 >= threshold) and (payoff2 >= threshold)

def save_transcript(category, transcript, cos_sim):
    """Append a transcript (as JSON) to the appropriate dataset file."""
    out_dir = Path("data") / "finetuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{category}.jsonl"
    with file_path.open("a") as f:
        json.dump({"cosine_similarity": cos_sim, "messages": transcript}, f)
        f.write("\n")

def main():
    """
    Main function that generates negotiation transcripts.
    It dynamically generates valuation vectors with a random theta,
    computes the cosine similarity between v1 and -v2 (which equals -cos(theta)),
    and labels the transcript as 'cooperative', 'conflict', or 'mixed' based on strong thresholds:
      - Cooperative: cosine similarity >= 0.90 (theta near pi)
      - Conflict: cosine similarity <= -0.90 (theta near 0)
      - Mixed: absolute cosine similarity <= 0.1 (theta near pi/2)
    Transcripts not meeting these criteria are discarded.
    The accepted transcripts are saved as JSONL files in the "data/finetuning" directory,
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
        # Generate a random theta in [0, pi] for each transcript
        theta = np.random.uniform(0, np.pi)
        v1, v2 = generate_valuations(N_OBJECTS, tuple(BASE_RANGE), theta)
        
        # Compute cosine similarity between v1 and -v2.
        # Given our generation method, cos_sim = -cos(theta)
        cos_sim = -np.cos(theta)
        
        label = None
        if cos_sim >= 0.90:
            label = "cooperative"
        elif cos_sim <= -0.90:
            label = "conflict"
        elif abs(cos_sim) <= 0.1:
            label = "mixed"
        else:
            # Not a strong example – skip this transcript.
            continue
        
        # Check if we already have enough samples for this category
        if counts[label] >= target:
            continue
        
        # Create the system prompt using the label for the game context
        system_prompt = create_system_prompt(
            n_objects=N_OBJECTS,
            v_agent=v1,
            m_rounds=M_ROUNDS,
            game_context=f"{label.capitalize()} Negotiation",
            p_fraction=P_FRACTION,
            q_fraction=Q_FRACTION
        )
        
        transcript, claims = generate_negotiation_transcript(system_prompt, (v1, v2))
        claims_agent1 = parse_claims(claims["agent1"])
        claims_agent2 = parse_claims(claims["agent2"])
        payoff1, payoff2 = compute_payoffs(v1, v2, claims_agent1, claims_agent2)
        
        if filter_transcript(payoff1, payoff2):
            save_transcript(label, transcript, cos_sim)
            counts[label] += 1
            print(f"Saved {label} transcript {counts[label]} / {target}, cosine similarity: {cos_sim:.2f}")
        else:
            print("Transcript filtered out due to low payoffs.")

if __name__ == '__main__':
    main()