from pathlib import Path
import json
import numpy as np
import re
import math
import uuid
import datetime
import argparse
import random

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

# Import utilities for valuation generation and prompt creation.
from utils.valuation_generation import generate_valuations
from utils.prompt_templates import create_system_prompt

# --- Constants ---
P_FRACTION = 0.25
Q_FRACTION = 0.5
BASE_RANGE = [-10, 10]

# --- Helper Functions ---
def parse_claims(claim_text: str):
    """
    Parse a claim message and return a list of 0-indexed object indices.
    """
    numbers = re.findall(r'\d+', claim_text)
    return [int(num) - 1 for num in numbers]

def compute_cosine_similarity(v1, v2):
    """
    Compute cosine similarity between two vectors.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (norm1 * norm2) if norm1 and norm2 else 0

def compute_payoffs(v1, v2, claims_agent1, claims_agent2, p=P_FRACTION, q=Q_FRACTION):
    """
    Compute the raw and discretized payoffs for a negotiation outcome.
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

def simulate_negotiation(model_agent1, model_agent2, system_prompt, v1, v2, m_rounds):
    """
    Simulate a negotiation between two agents using the specified models.
    Returns the full transcript and the final claim texts.
    """
    transcript = []
    # Agents are fixed as ("agent1", model_agent1) and ("agent2", model_agent2)
    agents = [("agent1", model_agent1), ("agent2", model_agent2)]
    
    # --- Dialogue Rounds ---
    for round_idx in range(m_rounds):
        for agent_name, model in agents:
            conversation = [{"role": "system", "content": system_prompt}] + transcript
            current_valuation = v1 if agent_name == "agent1" else v2
            turn_prompt = (
                f"Round {round_idx+1}: You are {agent_name}. Your valuation is: "
                f"{np.array2string(current_valuation, precision=2)}. Please send your message."
            )
            user_message = {"role": "user", "content": turn_prompt, "speaker": agent_name}
            conversation.append(user_message)
            transcript.append(user_message)
            
            response = client.chat.completions.create(
                model=model,
                store=True,
                messages=conversation,
                temperature=0.7,
            )
            assistant_message_text = response.choices[0].message.content
            assistant_message = {"role": "assistant", "content": f"[{agent_name}] {assistant_message_text}", "speaker": agent_name}
            transcript.append(assistant_message)
    
    # --- Final Claim Round ---
    final_claims = {}
    for agent_name, model in agents:
        conversation = [{"role": "system", "content": system_prompt}] + transcript
        claim_prompt = f"{agent_name}, please list the object numbers you claim."
        user_claim = {"role": "user", "content": claim_prompt, "speaker": agent_name}
        conversation.append(user_claim)
        transcript.append(user_claim)
        
        response = client.chat.completions.create(
            model=model,
            store=True,
            messages=conversation,
            temperature=0.7,
        )
        claim_text = response.choices[0].message.content
        assistant_claim = {"role": "assistant", "content": f"[{agent_name}] {claim_text}", "speaker": agent_name}
        transcript.append(assistant_claim)
        final_claims[agent_name] = claim_text
        
    return transcript, final_claims

def evaluate_model_pair(model_name1, model_name2, num_scenarios_per_label=5, seed=None):
    """
    Evaluate a given pair of models (agent1 and agent2) over held-out scenarios.
    Returns a list of evaluation results.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    scenario_labels = ["cooperative", "conflict", "mixed"]
    results = []
    scenario_counts = {label: 0 for label in scenario_labels}
    target = num_scenarios_per_label
    
    while any(scenario_counts[label] < target for label in scenario_labels):
        num_objects = np.random.randint(3, 6)
        m_rounds = np.random.randint(3, 7)
        theta = np.random.uniform(0, np.pi)
        v1, v2 = generate_valuations(num_objects, tuple(BASE_RANGE), theta)
        
        cos_sim = compute_cosine_similarity(v1, -v2)
        if cos_sim >= 0.90:
            label = "cooperative"
        elif cos_sim <= -0.90:
            label = "conflict"
        else:
            label = "mixed"
        
        if scenario_counts[label] >= target:
            continue
        
        system_prompt = create_system_prompt(
            n_objects=num_objects,
            v_agent=v1,
            m_rounds=m_rounds,
            game_context=f"{label.capitalize()} Negotiation",
            p_fraction=P_FRACTION,
            q_fraction=Q_FRACTION
        )
        
        transcript, claims = simulate_negotiation(model_name1, model_name2, system_prompt, v1, v2, m_rounds)
        agent1_claims = parse_claims(claims["agent1"])
        agent2_claims = parse_claims(claims["agent2"])
        discrete1, discrete2, raw1, raw2 = compute_payoffs(v1, v2, agent1_claims, agent2_claims, P_FRACTION, Q_FRACTION)
        
        scenario_id = str(uuid.uuid4())
        result = {
            "scenario_id": scenario_id,
            "label": label,
            "num_objects": num_objects,
            "num_rounds": m_rounds,
            "theta": theta,
            "cosine_similarity": cos_sim,
            "v1": v1.tolist() if isinstance(v1, np.ndarray) else v1,
            "v2": v2.tolist() if isinstance(v2, np.ndarray) else v2,
            "model_pair": (model_name1, model_name2),
            "raw_payoffs": [raw1, raw2],
            "discrete_payoffs": [discrete1, discrete2],
            "transcript_length": len(transcript),
            "transcript": transcript,
            "claims": claims
        }
        results.append(result)
        scenario_counts[label] += 1
        print(f"Evaluated scenario {scenario_id} with label '{label}' for model pair ({model_name1}, {model_name2}).")
        
    return results

# --- Main Evaluation Routine ---
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned models in negotiation tasks via self-play and cross-play."
    )
    parser.add_argument("--num_scenarios", type=int, default=5, 
                        help="Number of scenarios per label (cooperative, conflict, mixed) to evaluate for each model pair.")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    models = {
        "baseline": "gpt-4o-mini-2024-07-18",
        "competitive": "ft:gpt-4o-mini-2024-07-18:neural-interactive-proofs:conflict-ft-20250213-1518:B0a8c774",
        "cooperative": "ft:gpt-4o-mini-2024-07-18:neural-interactive-proofs:cooperative-ft-20250213-1518:B0a7nLB7",
        "mixed": "ft:gpt-4o-mini-2024-07-18:neural-interactive-proofs:mixed-ft-20250213-1518:B0aBWOlT",
    }

        #    "baseline": "gpt-4o-mini-2024-07-18",
        # "competitive": "ft:gpt-4o-mini-2024-07-18:cooperative-ai-foundation:mixed-ft-20250213-1343:B0Yg8mBH",
        # "cooperative": "ft:gpt-4o-mini-2024-07-18:cooperative-ai-foundation:cooperative-ft-20250213-1343:B0YdG6xe",
        # "mixed": "ft:gpt-4o-mini-2024-07-18:cooperative-ai-foundation:mixed-ft-20250213-1343:B0Yg8mBH",
    
    evaluation_results = {}
    
    # --- Self-play Evaluations ---
    print("Starting self-play evaluations...")
    for model_label, model_name in models.items():
        print(f"\nEvaluating self-play for model: {model_label}")
        results = evaluate_model_pair(model_name, model_name, num_scenarios_per_label=args.num_scenarios, seed=args.seed)
        evaluation_results[f"{model_label}_selfplay"] = results
    
    # --- Cross-play Evaluations ---
    # For each pair of distinct models, evaluate both orderings.
    cross_pairs = []
    model_labels = list(models.keys())
    for i in range(len(model_labels)):
        for j in range(i+1, len(model_labels)):
            cross_pairs.append((model_labels[i], model_labels[j]))
    
    print("\nStarting cross-play evaluations...")
    for label1, label2 in cross_pairs:
        model_name1 = models[label1]
        model_name2 = models[label2]
        print(f"\nEvaluating cross-play for pair: {label1} (agent1) vs {label2} (agent2)")
        results1 = evaluate_model_pair(model_name1, model_name2, num_scenarios_per_label=args.num_scenarios, seed=args.seed)
        evaluation_results[f"{label1}_vs_{label2}"] = results1
        
        print(f"\nEvaluating cross-play for pair: {label2} (agent1) vs {label1} (agent2)")
        results2 = evaluate_model_pair(model_name2, model_name1, num_scenarios_per_label=args.num_scenarios, seed=args.seed)
        evaluation_results[f"{label2}_vs_{label1}"] = results2
    
    # Create a run-labeled folder (using timestamp) to store results.
    run_label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = Path("evaluation") / run_label
    run_folder.mkdir(parents=True, exist_ok=True)
    
    # Save aggregated evaluation results.
    aggregated_file = run_folder / "aggregated_results.json"
    with aggregated_file.open("w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\nEvaluation complete. Aggregated results saved to {aggregated_file}")
    
    # Save the full exchanges (transcripts and claims) as a JSONL file.
    transcript_file = run_folder / "exchanges.jsonl"
    with transcript_file.open("w") as f:
        for eval_type, scenarios in evaluation_results.items():
            for scenario in scenarios:
                scenario_record = scenario.copy()
                scenario_record["evaluation_type"] = eval_type
                f.write(json.dumps(scenario_record) + "\n")
    print(f"Exchanges saved to {transcript_file}")

if __name__ == '__main__':
    main() 