import openai

# Set your OpenAI API key (ideally load this from an environment variable)
openai.api_key = "YOUR_OPENAI_API_KEY"

def create_finetuning_job(training_file_id: str, suffix: str) -> None:
    """
    Creates a supervised fine-tuning job for the specified model using
    the training file identified by training_file_id. A suffix is added
    to the fine-tuned model's name for identification.
    """
    response = openai.FineTuning.jobs.create(
        training_file=training_file_id,
        model="gpt-4o-mini-2024-07-18",
        suffix=suffix,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {"n_epochs": 2}
            }
        }
    )
    print(f"Fine-tuning job for '{suffix}' dataset created:")
    print(response)

if __name__ == '__main__':
    training_file_id = "file-abc123"
    suffix = "competitive"  # could be "conflict", "cooperative", or "mixed"
    create_finetuning_job(training_file_id, suffix)
