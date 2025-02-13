#!/usr/bin/env python3
import json
import argparse
from collections import defaultdict
import statistics

def analyze_scenarios(file_path):
    overall_payoffs = defaultdict(list)
    label_payoffs = defaultdict(lambda: defaultdict(list))
    scenario_count = 0

    # Try to load the file as an aggregated JSON object.
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file: {e}")
        return overall_payoffs, label_payoffs, scenario_count

    # Determine if the file is an aggregated JSON (dict with lists) or a single list.
    scenarios = []
    if isinstance(data, list):
        scenarios = data
    elif isinstance(data, dict):
        # Iterate over keys; if the value is a list, add its elements.
        for key, value in data.items():
            if isinstance(value, list):
                scenarios.extend(value)
            else:
                print(f"Skipping key '{key}' since its value is not a list.")
    else:
        print("Unknown data format in JSON file.")
        return overall_payoffs, label_payoffs, scenario_count

    # Process each scenario.
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            print(f"Skipping non-dictionary scenario: {scenario}")
            continue

        label = scenario.get("label", "unknown")
        models = scenario.get("model_pair", [])
        payoffs = scenario.get("raw_payoffs", [])

        # Ensure that we have the expected types.
        if not isinstance(models, list) or not isinstance(payoffs, list):
            print(f"Skipping scenario {scenario.get('scenario_id', 'unknown')} because 'model_pair' or 'raw_payoffs' is not a list.")
            continue

        # Zip models and payoffs and record the data.
        for model, payoff in zip(models, payoffs):
            if not isinstance(model, str):
                print(f"Skipping invalid model entry: {model}")
                continue
            try:
                payoff = float(payoff)
            except (ValueError, TypeError):
                print(f"Skipping invalid payoff: {payoff}")
                continue
            overall_payoffs[model].append(payoff)
            label_payoffs[label][model].append(payoff)
        scenario_count += 1

    return overall_payoffs, label_payoffs, scenario_count

def print_narrative(overall_payoffs, label_payoffs, scenario_count):
    print("Cross-Play Analysis Report")
    print("===========================")
    print(f"Total scenarios analyzed: {scenario_count}\n")
    
    # Compute average payoffs for each model overall.
    avg_payoffs = {}
    print("Overall Average Payoffs per Model:")
    for model, payoffs in overall_payoffs.items():
        avg = sum(payoffs) / len(payoffs)
        avg_payoffs[model] = avg
        stdev = statistics.stdev(payoffs) if len(payoffs) > 1 else 0.0
        print(f"  Model: {model}")
        print(f"    Scenarios participated: {len(payoffs)}")
        print(f"    Average raw payoff: {avg:.3f}")
        print(f"    Standard deviation: {stdev:.3f}\n")
    
    # Determine the best performing model overall.
    if avg_payoffs:
        best_model = max(avg_payoffs, key=avg_payoffs.get)
        print("Overall Best Performing Model:")
        print(f"  Model: {best_model} with an average payoff of {avg_payoffs[best_model]:.3f}\n")
    else:
        print("No valid payoff data found.")
        return
    
    # Define the competitive model id (from your provided metadata).
    COMPETITIVE_MODEL_ID = "ft:gpt-4o-mini-2024-07-18:cooperative-ai-foundation:mixed-ft-20250213-1343:B0Yg8mBH"
    if best_model == COMPETITIVE_MODEL_ID:
        print("The competitive model performed the best overall.")
    else:
        print("The competitive model did not perform the best overall.")
    
    # Breakdown performance by scenario label.
    print("\nPerformance by Scenario Type:")
    for label, model_dict in label_payoffs.items():
        print(f"\nScenario Label: {label}")
        for model, payoffs in model_dict.items():
            mean = sum(payoffs) / len(payoffs)
            stdev = statistics.stdev(payoffs) if len(payoffs) > 1 else 0.0
            print(f"  Model: {model}")
            print(f"    Scenarios: {len(payoffs)}")
            print(f"    Average payoff: {mean:.3f}")
            print(f"    Std Dev: {stdev:.3f}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze cross-play negotiation scenarios from an aggregated JSON file."
    )
    parser.add_argument("json_file", help="Path to the aggregated JSON file containing scenario data")
    args = parser.parse_args()
    
    overall_payoffs, label_payoffs, scenario_count = analyze_scenarios(args.json_file)
    print_narrative(overall_payoffs, label_payoffs, scenario_count)

if __name__ == "__main__":
    main()
