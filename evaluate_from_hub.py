from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from data import load_and_prepare_data
from utils import compute_metrics

MODEL_NAME = "tanvi130506/distilbert-imdb-mlops"


def main():
    print("Loading dataset...")
    dataset, tokenizer = load_and_prepare_data()

    print("Loading model from Hugging Face Hub...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir="./eval_results",
        per_device_eval_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    print("Running evaluation...")
    metrics = trainer.evaluate()
    print("Evaluation Results:", metrics)


if __name__ == "__main__":
    main()
