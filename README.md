# Fine-tuning the Nucleotide Transformer

This project provides a modular and configurable pipeline to fine-tune the [Nucleotide Transformer model](https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-human-ref) from InstaDeep on various downstream genomics tasks.

## Project Overview

This framework is a refactored and structured version of the original fine-tuning notebook. It is designed for robustness, reusability, and ease of experimentation. Instead of a monolithic notebook, the code is organized into a modular structure with separate components for data loading, model definition, and training, all controlled by a central configuration file.

The primary tasks demonstrated are:
-   **Promoter Prediction**: A binary classification task to predict whether a DNA sequence is a promoter region.
-   **Splice Site Prediction**: A multi-class classification task to identify splice sites.

## Project Structure

`​`​`
.
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml         # Main configuration file for tasks and hyperparameters
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Handles dataset loading, splitting, and tokenization
│   ├── model.py          # Defines function to get model and tokenizer
│   └── train.py          # Contains training logic and metrics computation
└── run_training.py         # Main script to run the training pipeline
`​`​`

## Setup and Installation

1.  **Clone the repository:**
    `​`​`bash
    git clone https://github.com/adityasengar/Fine_tuning_Nucleotide_Transformer.git
    cd Fine_tuning_Nucleotide_Transformer
    `​`​`

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    `​`​`bash
    pip install -r requirements.txt
    `​`​`

## How to Run

The training process is controlled by the `config/config.yaml` file and executed via the `run_training.py` script.

### 1. Configure the Task

Open `config/config.yaml` to set up your experiment. You can define multiple tasks in the `tasks` section.

Example for `promoter_prediction`:
`​`​`yaml
tasks:
  promoter_prediction:
    dataset_name: "promoter_all"
    num_labels: 2
    training_args:
      output_dir: "./results/promoter_prediction"
      learning_rate: 1e-5
      per_device_train_batch_size: 8
      num_train_epochs: 2
      max_steps: 1000
      # ... other training arguments
`​`​`

You can easily add new tasks or modify existing ones by changing the parameters in this file.

### 2. Run Training

To run the fine-tuning process for a specific task, execute the `run_training.py` script and specify the task name using the `--task` argument.

**Example for Promoter Prediction:**
`​`​`bash
python run_training.py --task promoter_prediction
`​`​`

**Example for Splice Site Prediction:**
`​`​`bash
python run_training.py --task splice_site_prediction
`​`​`

The script will handle everything: loading the correct dataset and model, setting up the trainer with the specified hyperparameters, running the training and evaluation, and printing the final test set metrics. Results and checkpoints will be saved to the `output_dir` specified in the configuration.
