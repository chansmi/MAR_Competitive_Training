import json
from pathlib import Path
from fine_tuning.utils import validate_format

def prepare_finetuning_data(input_file: Path, output_file: Path) -> None:
    """
    Reads a raw fine-tuning data file in JSONL format, validates each example,
    and writes only valid examples (in the proper chat format) to the output file.
    """
    with input_file.open("r") as fin, output_file.open("w") as fout:
        for line in fin:
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                # Skip lines that are not valid JSON.
                continue

            if not validate_format(example):
                # Skip invalid training examples.
                continue

            json.dump(example, fout)
            fout.write("\n")

if __name__ == '__main__':
    input_path = Path("data/finetuning/conflict.jsonl")  # or mixed.jsonl, as appropriate
    output_path = Path("data/finetuning/prepared_conflict.jsonl")
    prepare_finetuning_data(input_path, output_path)
    print("Prepared fine-tuning data.")
