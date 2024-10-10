import numpy as np
from sklearn.metrics import f1_score
from transformers import Trainer, TrainingArguments

def compute_metrics_f1_score(eval_pred):
    """
    Computes F1 score for binary classification.
    """
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    r = {'f1_score': f1_score(references, predictions, average='macro')}
    return r

def train_model(model, tokenizer, train_dataset, val_dataset, training_args):
    """
    Initializes and runs the Hugging Face Trainer.
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_f1_score,
    )

    train_results = trainer.train()

    return trainer, train_results
