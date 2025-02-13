from pathlib import Path
import json
import argparse
import pandas as pd
import numpy as np

def load_evaluation_data(file_path: Path) -> pd.DataFrame:
    """
    Load a single aggregated evaluation JSON file and flatten the records.
    """
    with file_path.open("r") as f:
        data = json.load(f)
    records = []
    for eval_type, scenarios in data.items():
        for sc in scenarios:
            # Unpack raw and discrete payoffs into separate columns.
            if "raw_payoffs" in sc:
                sc["raw_agent1"] = sc["raw_payoffs"][0]
                sc["raw_agent2"] = sc["raw_payoffs"][1]
            if "discrete_payoffs" in sc:
                sc["discrete_agent1"] = sc["discrete_payoffs"][0]
                sc["discrete_agent2"] = sc["discrete_payoffs"][1]
            sc["evaluation_type"] = eval_type
            records.append(sc)
    return pd.DataFrame(records)

def load_evaluation_data_from_folder(folder: Path) -> pd.DataFrame:
    """
    Search all subfolders of the given folder for aggregated_results.json files
    and return a combined DataFrame with a 'run_id' field.
    """
    records = []
    for aggregated_file in folder.glob("*/aggregated_results.json"):
        run_id = aggregated_file.parent.name
        with aggregated_file.open("r") as f:
            data = json.load(f)
        for eval_type, scenarios in data.items():
            for sc in scenarios:
                if "raw_payoffs" in sc:
                    sc["raw_agent1"] = sc["raw_payoffs"][0]
                    sc["raw_agent2"] = sc["raw_payoffs"][1]
                if "discrete_payoffs" in sc:
                    sc["discrete_agent1"] = sc["discrete_payoffs"][0]
                    sc["discrete_agent2"] = sc["discrete_payoffs"][1]
                sc["evaluation_type"] = eval_type
                sc["run_id"] = run_id
                records.append(sc)
    return pd.DataFrame(records)

def analyze_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive summary statistics to investigate negotiation efficiency.

    Additional computed columns:
    - raw_diff: difference in raw outcomes between agent1 and agent2.
    - social_welfare: sum of both agents' raw payoffs.
    - win_diff: difference between the discrete outcomes (win: 1, loss: -1, tie: 0).
    
    Aggregated metrics include:
    - Count of scenarios.
    - Mean and standard deviation for raw payoffs, payoff differences, and social welfare.
    - Total wins for each agent.
    - Average cosine similarity.
    """
    df["raw_diff"] = df["raw_agent1"] - df["raw_agent2"]
    df["social_welfare"] = df["raw_agent1"] + df["raw_agent2"]
    # Compute win difference per scenario: if agent1 wins then (1 - (-1)) = 2, if agent2 wins then (-1 - 1) = -2, 0 for tie.
    df["win_diff"] = df["discrete_agent1"] - df["discrete_agent2"]

    # Group by run_id, evaluation_type, and label.
    summary = df.groupby(["run_id", "evaluation_type", "label"]).agg(
        count=("scenario_id", "count"),
        mean_raw_agent1=("raw_agent1", "mean"),
        std_raw_agent1=("raw_agent1", "std"),
        mean_raw_agent2=("raw_agent2", "mean"),
        std_raw_agent2=("raw_agent2", "std"),
        mean_raw_diff=("raw_diff", "mean"),
        std_raw_diff=("raw_diff", "std"),
        mean_social_welfare=("social_welfare", "mean"),
        std_social_welfare=("social_welfare", "std"),
        win_agent1_count=("discrete_agent1", lambda x: (x == 1).sum()),
        win_agent2_count=("discrete_agent1", lambda x: (x == -1).sum()),
        mean_win_diff=("win_diff", "mean"),
        std_win_diff=("win_diff", "std"),
        avg_cosine_sim=("cosine_similarity", "mean")
    ).reset_index()
    
    return summary

def additional_analysis(df: pd.DataFrame):
    """
    Perform additional analysis to compare negotiation efficiency metrics.
    This includes evaluating average social welfare and win differences
    across evaluation types and play styles (self-play vs cross-play), and
    computing correlations between efficiency measures.
    """
    # Pivot table: average social welfare for each evaluation type.
    pivot_sw = pd.pivot_table(df, values="social_welfare", index="evaluation_type", aggfunc=[np.mean, np.std, len])
    print("Social Welfare by Evaluation Type:")
    print(pivot_sw)
    
    # Pivot table: average win difference (agent1 win - agent2 win) by evaluation type.
    pivot_win = pd.pivot_table(df, values="win_diff", index="evaluation_type", aggfunc=[np.mean, np.std, len])
    print("\nWin Difference (agent1 win minus agent2 win) by Evaluation Type:")
    print(pivot_win)
    
    # Correlation between raw difference and social welfare
    corr = df[["raw_diff", "social_welfare"]].corr()
    print("\nCorrelation between Payoff Imbalance (raw_diff) and Social Welfare:")
    print(corr)
    
    # Determine play type based on evaluation_type string
    df["play_type"] = df["evaluation_type"].apply(lambda et: "self-play" if "selfplay" in et else "cross-play")
    pivot_play = pd.pivot_table(df, values=["raw_agent1", "raw_agent2", "social_welfare"],
                                  index="play_type", aggfunc=[np.mean, np.std, len])
    print("\nPerformance by Play Type (Self-play vs Cross-play):")
    print(pivot_play)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze aggregated evaluation results to investigate negotiation efficiency and potential inefficiencies."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str,
                       help="Path to a single aggregated evaluation JSON file.")
    group.add_argument("--folder", type=str,
                       help="Path to the folder containing run subfolders with aggregated evaluation JSON files.")
    args = parser.parse_args()

    # Load data from file or folder.
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return
        df = load_evaluation_data(file_path)
    else:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"Folder not found: {folder}")
            return
        df = load_evaluation_data_from_folder(folder)
    
    # Compute comprehensive summary.
    summary = analyze_data(df)
    print("Comprehensive Summary of Evaluation Results:")
    print(summary.to_string(index=False))
    
    # Save summary analysis to CSV.
    output_file = Path("evaluation_summary.csv")
    summary.to_csv(output_file, index=False)
    print(f"\nSummary saved to {output_file}")
    
    # Run additional analyses.
    print("\nAdditional Analysis:")
    additional_analysis(df)

if __name__ == "__main__":
    main() 