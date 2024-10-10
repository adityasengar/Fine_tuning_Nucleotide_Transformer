from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

def tokenize_function(examples, tokenizer):
    """
    Tokenizes the data.
    """
    return tokenizer(examples["data"])

def get_datasets(dataset_name, tokenizer):
    """
    Loads, splits, and tokenizes the dataset.
    """
    # Load dataset from Hugging Face Hub
    train_dataset = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        dataset_name,
        split="train",
        streaming=False,
    )
    test_dataset = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        dataset_name,
        split="test",
        streaming=False,
    )

    # Get training data and create a validation split
    train_sequences = train_dataset['sequence']
    train_labels = train_dataset['label']
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        train_sequences, train_labels, test_size=0.05, random_state=42
    )

    # Get test data
    test_sequences = test_dataset['sequence']
    test_labels = test_dataset['label']

    # Create Hugging Face Dataset objects
    ds_train = Dataset.from_dict({"data": train_sequences, 'labels': train_labels})
    ds_val = Dataset.from_dict({"data": val_sequences, 'labels': val_labels})
    ds_test = Dataset.from_dict({"data": test_sequences, 'labels': test_labels})

    # Tokenize datasets
    tokenized_train = ds_train.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["data"],
    )
    tokenized_val = ds_val.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["data"],
    )
    tokenized_test = ds_test.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["data"],
    )

    return tokenized_train, tokenized_val, tokenized_test
