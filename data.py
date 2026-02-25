from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"


def load_and_prepare_data():
    # Load IMDB dataset
    dataset = load_dataset("imdb")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenization function
    def tokenize_function(example):
        return tokenizer(
            example["text"], padding="max_length", truncation=True, max_length=256
        )

    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Remove unnecessary columns
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # Set PyTorch format
    tokenized_dataset.set_format("torch")

    return tokenized_dataset, tokenizer
