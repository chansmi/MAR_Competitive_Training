from pathlib import Path
import json
import argparse
import pandas as pd

def load_evaluation_data(file_path: Path) -> pd.DataFrame:
    """
    Load the evaluation JSON file and convert the nested structure
    into a flat DataFrame that includes the raw and discrete payoffs.
    """
    with file_path.open("r") as f:
        data = json.load(f)

    records = []
    for eval_type, scenarios in data.items():
        for sc in scenarios:
            # Unpack raw and discrete payoffs into separate fields.
            if "raw_payoffs" in sc:
                sc["raw_agent1"] = sc["raw_payoffs"][0]
                sc["raw_agent2"] = sc["raw_payoffs"][1]
            if "discrete_payoffs" in sc:
                sc["discrete_agent1"] = sc["discrete_payoffs"][0]
                sc["discrete_agent2"] = sc["discrete_payoffs"][1]
            # Tag the record with the evaluation type (e.g. baseline_selfplay, a_vs_b, etc.)
            sc["evaluation_type"] = eval_type
            records.append(sc)
    df = pd.DataFrame(records)
    return df

def analyze_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform grouping and aggregation to summarize evaluation results.
    It computes average raw payoffs, the difference between them, win counts,
    and other useful statistics per evaluation type and scenario label.
    """
    # Create a column for the payoff difference (agent1 - agent2).
    df["raw_diff"] = df["raw_agent1"] - df["raw_agent2"]

    # Group by the evaluation type and scenario label.
    summary = df.groupby(["evaluation_type", "label"]).agg(
        count=("scenario_id", "count"),
        avg_raw_agent1=("raw_agent1", "mean"),
        avg_raw_agent2=("raw_agent2", "mean"),
        avg_raw_diff=("raw_diff", "mean"),
        win_agent1=("discrete_agent1", lambda x: (x == 1).sum()),
        win_agent2=("discrete_agent1", lambda x: (x == -1).sum()),
        avg_cosine_sim=("cosine_similarity", "mean")
    ).reset_index()

    return summary

def main():
    parser = argparse.ArgumentParser(
        description="Analyze the evaluation results JSON file for model performance."
    )
    parser.add_argument("file", type=str, help="Path to the evaluation JSON file.")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    df = load_evaluation_data(file_path)
    summary = analyze_data(df)

    print("Summary of Evaluation Results:")
    print(summary.to_string(index=False))

    # Optionally, save the summary to a CSV file for further review.
    output_file = file_path.parent / "evaluation_summary.csv"
    summary.to_csv(output_file, index=False)
    print(f"Summary saved to {output_file}")

if __name__ == "__main__":
    main() 