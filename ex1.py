###########################################################
# Exercise 1 - Advanced Natural Language Processing 67664 #
###########################################################
### Imports ###
import argparse
import evaluate
import numpy as np
import sklearn
import os
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
### Imports End ###
### Global Vars ###
SAVE_DIR = "./outputs"
### Global Vars End ###


def parse_args():
    parser = argparse.ArgumentParser()
    # Dataset size options
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    # Modes
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    # Model path (used for prediction)
    parser.add_argument("--model_path", type=str, default=None)
    # Added for accuracy comparison
    parser.add_argument("--predict_split", type=str, default="test",
        choices=["test", "validation", "train"]
    )
    return parser.parse_args()

def select_data_subset(split_dataset, max_samples):
    if max_samples == -1:
        return split_dataset
    n = min(max_samples, len(split_dataset))  # handle subsets
    return split_dataset.select(range(n))


def apply_sample_limits(raw_dataset, args):
    train_dataset = select_data_subset(
        raw_dataset["train"], args.max_train_samples)
    eval_dataset = select_data_subset(
        raw_dataset["validation"], args.max_eval_samples)
    predict_dataset = select_data_subset(
        raw_dataset["test"], args.max_predict_samples)
    return train_dataset, eval_dataset, predict_dataset


def tokenize_function(samples, tokenizer):
    return tokenizer(
        samples["sentence1"],
        samples["sentence2"],
        truncation=True,
    )


def tokenize_and_remove_columns(sub_dataset, tokenizer):
    tokenized_dataset = sub_dataset.map(
        lambda samples: tokenize_function(samples, tokenizer),
        batched=True,
    ).remove_columns(["sentence1", "sentence2", "idx"])
    return tokenized_dataset


def compute_metrics(eval_pred, metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def initialize_wandb(args):
    wandb.init(
        project="anlp-ex1-mrpc",
        name=f"lr_{args.lr}_bs_{args.batch_size}_ep_{args.num_train_epochs}",
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.num_train_epochs,
        }
    )


def create_train_args(args):
    training_args = TrainingArguments(
        output_dir=f"{SAVE_DIR}",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_strategy="steps",
        logging_steps=1,
        report_to="wandb",
        load_best_model_at_end=False,
    )
    return training_args


def create_prediction_args(args):
    prediction_args = TrainingArguments(
        output_dir=f"{SAVE_DIR}/prediction_run",
        report_to="none",
        per_device_eval_batch_size=args.batch_size,
    )
    return prediction_args

def main():
    args = parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    # Components for both train and predict:
    # Load tokenizer, dynamic padding and evaluation components
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy_metric = evaluate.load("accuracy")
    # Load dataset & apply sample limits if given
    raw_dataset = load_dataset("glue", "mrpc")
    train_dataset, eval_dataset, test_dataset = apply_sample_limits(
        raw_dataset, args)
    print("\nFinished loading the raw dataset")
    print(f"Train size: {len(train_dataset)}, eval size: {len(eval_dataset)}, "
          f"predict size: {len(test_dataset)}")
    ### Train block ###
    if args.do_train:
        print("Training mode:")
        # Training arguments
        training_args = create_train_args(args)
        initialize_wandb(args)
        # Tokenize data & remove unnecessary columns
        tokenized_train = tokenize_and_remove_columns(train_dataset, tokenizer)
        tokenized_eval = tokenize_and_remove_columns(eval_dataset, tokenizer)
        # change label to labels
        tokenized_train = tokenized_train.rename_column("label", "labels")
        tokenized_eval = tokenized_eval.rename_column("label", "labels")
        print("Finished tokenizing, removing unnecessary columns, "
              "and renaming 'label' to 'labels'.")
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
        )
        # Creating the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred,
                                                              accuracy_metric),
        )
        trainer.train()
        eval_results = trainer.evaluate()
        print(f"Validation accuracy: {eval_results['eval_accuracy']:.4f}")
        with open(os.path.join(training_args.output_dir, "res.txt"), "a") as f:
            f.write(
                f"epoch_num: {args.num_train_epochs}, "
                f"lr: {args.lr}, "
                f"batch_size: {args.batch_size}, "
                f"eval_acc: {eval_results['eval_accuracy']:.4f}\n"
            )
        output_dir = os.path.join(SAVE_DIR,
            f"model_ep{args.num_train_epochs}_lr{args.lr}_bs{args.batch_size}"
            )
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        wandb.finish()
        print(f"Trained model saved to: {output_dir}")
    ### Predict block ###
    if args.do_predict:
        if args.model_path is None:
            raise ValueError(
                "When using --do_predict, you must provide --model_path")
        print("Prediction mode:")
        # tokenization & column removal
        split_to_dataset = {
            "train": train_dataset,
            "validation": eval_dataset,
            "test": test_dataset,
        }
        selected_predict_dataset = split_to_dataset[args.predict_split]
        tokenized_predict = tokenize_and_remove_columns(selected_predict_dataset,
                                                        tokenizer)
        # could cause issues so rename
        if "label" in tokenized_predict.column_names:
            tokenized_predict = tokenized_predict.rename_column(
                "label", "labels")
        # Training arguments
        prediction_args = create_prediction_args(args)
        prediction_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path)
        prediction_model.eval()
        prediction_trainer = Trainer(
            model=prediction_model,
            args=prediction_args,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_pred: compute_metrics(
                eval_pred,
                accuracy_metric
            ),
        )
        predictions_output = prediction_trainer.predict(tokenized_predict)
        predicted_labels = np.argmax(predictions_output.predictions, axis=-1)
        with open(os.path.join(prediction_args.output_dir,
                               "predictions.txt"), "w") as f:
            for i in range(len(predicted_labels)):
                line = (f"{selected_predict_dataset['sentence1'][i]}###"
                        f"{selected_predict_dataset['sentence2'][i]}###"
                        f"{predicted_labels[i]}")
                f.write(f"{line}\n")
        print("Predictions saved to predictions.txt")


if __name__ == "__main__":
    main()