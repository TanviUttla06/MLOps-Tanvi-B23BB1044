"""
Q1: ViT-S Fine-tuning on CIFAR-100 with and without LoRA
Usage:
    # Without LoRA (baseline):
    python Q1/train.py --use_lora False --epochs 10

    # With LoRA:
    python Q1/train.py --use_lora True --rank 4 --alpha 4 --dropout 0.1 --epochs 10
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import wandb
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import numpy as np


# ─────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description="ViT-S CIFAR-100 Fine-tuning")
    parser.add_argument("--use_lora", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--rank", type=int, default=4, choices=[2, 4, 8])
    parser.add_argument("--alpha", type=int, default=4, choices=[2, 4, 8])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wandb_project", type=str, default="Assignment5-Q1")
    parser.add_argument("--save_dir", type=str, default="Q1/weights")
    return parser.parse_args()


# ─────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────
def get_dataloaders(batch_size):
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────
def build_model(use_lora, rank, alpha, dropout):
    # Load pretrained ViT-S/16 from timm
    model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=100)

    if not use_lora:
        # Freeze everything except the classification head
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        print("[Baseline] Only classification head is trainable.")
    else:
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False

        # Manually inject LoRA into Q, K, V projection weights of each attention block
        for block in model.blocks:
            attn = block.attn
            embed_dim = attn.qkv.in_features
            # We patch qkv with a LoRA-enabled version
            attn.qkv = LoRAQKV(attn.qkv, embed_dim, rank, alpha, dropout)

        # Keep classification head trainable
        for param in model.head.parameters():
            param.requires_grad = True

        print(f"[LoRA] rank={rank}, alpha={alpha}, dropout={dropout}")

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model, trainable


# ─────────────────────────────────────────
# LoRA Layer for QKV
# ─────────────────────────────────────────
class LoRAQKV(nn.Module):
    """Wraps a linear layer and adds LoRA to Q, K, V separately."""

    def __init__(
        self,
        original_qkv: nn.Linear,
        embed_dim: int,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        self.original_qkv = original_qkv
        self.original_qkv.weight.requires_grad = False
        if self.original_qkv.bias is not None:
            self.original_qkv.bias.requires_grad = False

        self.rank = rank
        self.scale = alpha / rank
        out_features = original_qkv.out_features  # 3 * embed_dim

        self.lora_A = nn.Linear(embed_dim, rank * 3, bias=False)
        self.lora_B = nn.Linear(rank * 3, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.original_qkv(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return base_out + lora_out


# ─────────────────────────────────────────
# Train / Eval Functions
# ─────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="[Val]"):
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # WandB run name
    if args.use_lora:
        run_name = f"LoRA_r{args.rank}_a{args.alpha}_d{args.dropout}"
    else:
        run_name = "Baseline_NoLoRA"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )

    train_loader, val_loader = get_dataloaders(args.batch_size)
    model, trainable_params = build_model(
        args.use_lora, args.rank, args.alpha, args.dropout
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0

    print("\n" + "=" * 60)
    print(
        f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Train Acc':>10} {'Val Acc':>10}"
    )
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} {train_acc:>10.4f} {val_acc:>10.4f}"
        )

        # Log LoRA gradient norms
        lora_grad_norms = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and "lora" in name:
                lora_grad_norms[f"grad_norm/{name}"] = param.grad.norm().item()

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/acc": train_acc,
                "val/acc": val_acc,
                **lora_grad_norms,
            }
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, f"{run_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  → Best model saved: {save_path} (val_acc={val_acc:.4f})")

    wandb.log({"best_val_acc": best_val_acc, "trainable_params": trainable_params})
    wandb.finish()
    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
