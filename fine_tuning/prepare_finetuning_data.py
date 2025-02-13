import json
import argparse
from pathlib import Path

def clean_messages(messages: list) -> list:
    """
    Processes the list of messages and adds a weight field as follows:
      - For each assistant message from agent1, assign weight 0.5,
        except for the final such message (the final claim) which gets weight 1.
      - For each assistant message from agent2, assign weight 0.
    Other messages are left unchanged.
    """
    # Identify indices for agent1 assistant messages
    agent1_indices = [
        i for i, msg in enumerate(messages)
        if msg.get("role") == "assistant" and msg.get("speaker") == "agent1"
    ]
    last_agent1_index = agent1_indices[-1] if agent1_indices else None

    cleaned = []
    for i, msg in enumerate(messages):
        new_msg = dict(msg)  # Create a shallow copy
        if new_msg.get("role") == "assistant":
            if new_msg.get("speaker") == "agent1":
                new_msg["weight"] = 1 if i == last_agent1_index else 0.5
            elif new_msg.get("speaker") == "agent2":
                new_msg["weight"] = 0
        cleaned.append(new_msg)
    return cleaned

def process_file(input_path: Path, output_path: Path) -> None:
    """
    Reads an input JSONL file containing transcripts with a 'messages' key.
    For each transcript, cleans the messages and writes a new JSON object 
    (with only the "messages" field) to the output file.
    """
    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line:\n{line}\nError: {e}")
                continue
            if "messages" not in data:
                print(f"Skipping line with no 'messages' key:\n{line}")
                continue

            cleaned_messages = clean_messages(data["messages"])
            cleaned_data = {"messages": cleaned_messages}
            outfile.write(json.dumps(cleaned_data))
            outfile.write("\n")

def main():
    parser = argparse.ArgumentParser(
        description="Clean existing finetuning JSONL data for training compatibility."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input JSONL file to clean (e.g., data/finetuning/mixed.jsonl)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to the output cleaned JSONL file. If not provided, it will be saved in the 'clean' subfolder with a '_clean' suffix."
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Input file '{input_path}' does not exist.")
        return

    # Determine the output path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        # Create a 'clean' subfolder next to the input file if it doesn't exist.
        output_dir = input_path.parent / "clean"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = input_path.stem + "_clean.jsonl"
        output_path = output_dir / output_filename

    process_file(input_path, output_path)
    print(f"Cleaned file written to: {output_path}")

if __name__ == "__main__":
    main()