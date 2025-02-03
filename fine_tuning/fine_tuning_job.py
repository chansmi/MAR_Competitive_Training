import openai

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

def create_finetuning_job(training_file_id):
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
    # Example: assume you've already uploaded your training file and have its file ID.
    training_file_id = "file-abc123"  # Replace with your actual file ID
    create_finetuning_job(training_file_id)
