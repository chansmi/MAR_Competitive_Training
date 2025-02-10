import openai

# Set your OpenAI API key (ideally load this from an environment variable)
openai.api_key = "YOUR_OPENAI_API_KEY"

def create_finetuning_job(training_file_id: str) -> None:
    """
    Creates a supervised fine-tuning job for the specified model using
    the training file identified by training_file_id.
    """
    response = openai.FineTuning.jobs.create(
        training_file=training_file_id,
        model="gpt-4o-mini-2024-07-18",
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {"n_epochs": 2}
            }
        }
    )
    print("Fine-tuning job created:")
    print(response)

if __name__ == '__main__':
    # Replace with your actual training file ID (from your upload step)
    training_file_id = "file-abc123"
    create_finetuning_job(training_file_id)
