from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_model_and_tokenizer(model_name, num_labels):
    """
    Loads the Nucleotide Transformer model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return model, tokenizer
