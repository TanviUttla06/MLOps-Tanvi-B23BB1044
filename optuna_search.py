"""
Q1: Optuna hyperparameter search for LoRA
Usage:
    python Q1/optuna_search.py --n_trials 20 --epochs 5
"""

import argparse
import torch
import optuna
import wandb
from Q1.train import get_dataloaders, build_model, train_one_epoch, evaluate
import torch.nn as nn
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--epochs",   type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--wandb_project", type=str, default="Assignment5-Q1-Optuna")
    return parser.parse_args()


def objective(trial, args, train_loader, val_loader, device):
    rank    = trial.suggest_categorical("rank",    [2, 4, 8])
    alpha   = trial.suggest_categorical("alpha",   [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.1)
    lr      = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    run_name = f"trial_{trial.number}_r{rank}_a{alpha}_d{dropout}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={"rank": rank, "alpha": alpha, "dropout": dropout, "lr": lr},
        reinit=True,
    )

    model, _ = build_model(use_lora=True, rank=rank, alpha=alpha, dropout=dropout)
    model     = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-2
    )
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler)
        _, val_acc = evaluate(model, val_loader, criterion, device)
        wandb.log({"val/acc": val_acc, "epoch": epoch})
        best_val_acc = max(best_val_acc, val_acc)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

    wandb.log({"best_val_acc": best_val_acc})
    wandb.finish()
    return best_val_acc


def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(args.batch_size)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name="LoRA-CIFAR100",
    )
    study.optimize(
        lambda trial: objective(trial, args, train_loader, val_loader, device),
        n_trials=args.n_trials,
    )

    print("\n" + "="*50)
    print("Best trial:")
    t = study.best_trial
    print(f"  Val Acc: {t.value:.4f}")
    print(f"  Params: {t.params}")

    os.makedirs("Q1/weights", exist_ok=True)
    with open("Q1/best_hparams.txt", "w") as f:
        f.write(f"Best Val Accuracy: {t.value:.4f}\n")
        for k, v in t.params.items():
            f.write(f"{k}: {v}\n")
    print("Best hyperparameters saved to Q1/best_hparams.txt")


if __name__ == "__main__":
    main()
