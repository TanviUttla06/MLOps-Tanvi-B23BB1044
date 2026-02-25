import os
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from data import load_and_prepare_data
from utils import compute_metrics

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./saved_model"


def main():
    print("Loading dataset...")
    dataset, tokenizer = load_and_prepare_data()

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # ALL of the following must be indented by 4 spaces to stay inside main()
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        push_to_hub=True,  # ADD THIS
        hub_model_id="tanvi130506/distilbert-imdb-mlops",  # CHANGE USERNAME
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    trainer.push_to_hub()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Running evaluation...")
    metrics = trainer.evaluate()
    print("Evaluation Results:", metrics)


if __name__ == "__main__":
    main()
