from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from datasets import load_dataset
from utils import compute_metrics

MODEL_PATH = "./saved_model"


def main():
    print("Loading saved model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    dataset = load_dataset("imdb")

    def tokenize_function(example):
        return tokenizer(
            example["text"], padding="max_length", truncation=True, max_length=256
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    trainer = Trainer(model=model, compute_metrics=compute_metrics)

    print("Evaluating model...")
    metrics = trainer.evaluate(tokenized_dataset["test"])
    print("Evaluation Results:", metrics)


if __name__ == "__main__":
    main()
