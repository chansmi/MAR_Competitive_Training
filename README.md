# Multi-Agent Negotiation Experiment

This repository implements an experiment to illustrate that training in competitive settings can lead to inefficiencies in multi‑agent negotiations. It includes:

- **Data Generation:** Synthetic negotiation transcripts across competitive, cooperative, and mixed settings.
- **Fine‑Tuning Datasets:** Filtered transcripts used to fine‑tune three models.
- **Evaluation:** Cross‑play and self‑play evaluations measuring individual payoffs and social welfare.
- **Fine‑Tuning on GPT Playground:** Instructions for fine‑tuning with model `gpt-4o-mini-2024-07-18`.

## Repository Structure

- `config/`: Contains configuration files.
- `data/finetuning/`: Fine‑tuning data in JSONL format.
- `scripts/`: Contains scripts for data generation, simulation, scoring, training, and evaluation.
- `fine_tuning/`: Contains fine‑tuning job creation scripts and utilities.
- `utils/`: Contains helper modules for valuation generation and prompt templating.

## Setup

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Generate data and transcripts:
    ```bash
    python scripts/generate_data.py
    ```

3. Run training and evaluation as needed:
    ```bash
    python scripts/train_models.py
    python scripts/evaluate_models.py
    ```

4. For fine‑tuning on GPT Playground, refer to `fine_tuning/fine_tuning_job.py`.
