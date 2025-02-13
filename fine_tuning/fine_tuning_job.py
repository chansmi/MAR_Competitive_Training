import argparse
from openai import OpenAI

client = OpenAI()  # Using the new SDK format

def create_finetuning_job(training_file_id: str, suffix: str) -> None:
    """
    Creates a supervised fine-tuning job for the model using the training file identified by 
    training_file_id. The suffix is attached to the fine-tuned model's name for identification.
    """
    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model="gpt-4o-mini-2024-07-18",
        suffix=suffix,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {"n_epochs": 5},
                "batch_size": 8,
            }
        }
    )
    print(f"Fine-tuning job for '{suffix}' dataset created:")
    print(job)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a fine-tuning job for a provided training file.")
    parser.add_argument('--training_file_id', required=True, help='The file ID of the training data')
    parser.add_argument('--suffix', required=True, help='Suffix for the fine-tuned model (e.g., conflict, cooperative, or mixed)')
    
    args = parser.parse_args()
    create_finetuning_job(args.training_file_id, args.suffix)
