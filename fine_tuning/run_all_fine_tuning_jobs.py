import os
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from fine_tuning_job import create_finetuning_job

client = OpenAI()

def upload_clean_data_file(dataset: str) -> str:
    """
    Uploads the cleaned fine-tuning data file for the given dataset.
    Returns the training file ID from OpenAI.
    """
    clean_file = Path("data/finetuning/clean") / f"{dataset}_clean.jsonl"
    if not clean_file.exists():
        raise FileNotFoundError(f"Clean file for dataset '{dataset}' not found at {clean_file}")
    
    with clean_file.open("rb") as fp:
        response = client.files.create(file=fp, purpose="fine-tune")

    print(response)
    
    file_id = response.id
    print(f"Uploaded '{dataset}' file: {clean_file} with ID: {file_id}")
    return file_id

def generate_suffix(dataset: str) -> str:
    """
    Generate a unique suffix for the fine-tuned model name.
    This suffix includes the dataset name and a timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{dataset}_ft_{timestamp}"

def main():
    # Define the datasets to process.
    datasets = ["conflict", "cooperative", "mixed"]
    dataset_file_ids = {}

    print("Starting the upload of cleaned training files...")
    # Upload each cleaned file and collect the corresponding file IDs.
    for ds in datasets:
        print(f"Processing dataset: '{ds}'")
        training_file_id = upload_clean_data_file(ds)
        dataset_file_ids[ds] = training_file_id

    print("\nCreating fine-tuning jobs for each dataset...")
    # For each dataset, generate a unique suffix for naming the fine-tuned model,
    # then create the fine-tuning job using that training file ID.
    for ds, file_id in dataset_file_ids.items():
        suffix = generate_suffix(ds)
        print(f"Launching fine-tuning job for '{ds}' with suffix: '{suffix}'")
        create_finetuning_job(training_file_id=file_id, suffix=suffix)

if __name__ == '__main__':
    main()
