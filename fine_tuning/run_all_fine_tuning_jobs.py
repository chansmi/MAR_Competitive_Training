import os
import openai
from fine_tuning.fine_tuning_job import create_finetuning_job

# Set your OpenAI API key from an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Mapping from dataset names to their corresponding training file IDs.
# Replace the file IDs with the actual IDs returned after uploading your prepared datasets.
dataset_file_ids = {
    "conflict": "file-conflict-abc123",
    "cooperative": "file-cooperative-def456",
    "mixed": "file-mixed-ghi789"
}

def main():
    for dataset, file_id in dataset_file_ids.items():
        print(f"Creating fine-tuning job for {dataset} dataset...")
        create_finetuning_job(file_id, dataset)

if __name__ == '__main__':
    main()
