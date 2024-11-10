import argparse
import yaml
import torch
from transformers import TrainingArguments
from src.model import get_model_and_tokenizer
from src.data_loader import get_datasets
from src.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Nucleotide Transformer")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["promoter_prediction", "splice_site_prediction"],
        help="The task to run.",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Get task-specific config
    task_config = config["tasks"][args.task]
    model_name = config["model_name"]
    device = config["device"]

    print(f"Starting task: {args.task}")

    # 1. Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(
        model_name, task_config["num_labels"]
    )
    model.to(device)

    # 2. Load and prepare datasets
    print("Loading and preparing datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(
        task_config["dataset_name"], tokenizer
    )

    # 3. Set up Training Arguments
    print("Setting up training arguments...")
    training_args_dict = task_config["training_args"]
    training_args = TrainingArguments(**training_args_dict)

    # 4. Train the model
    print("Starting training...")
    trainer, train_results = train_model(
        model, tokenizer, train_dataset, val_dataset, training_args
    )

    print("Training finished.")
    print(train_results)

    # 5. Evaluate on the test set
    print("Evaluating on the test set...")
    test_results = trainer.predict(test_dataset)
    print("Test results:")
    print(test_results.metrics)

if __name__ == "__main__":
    main()
