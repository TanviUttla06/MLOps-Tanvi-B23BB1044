"""
Q1: Testing ViT-S on CIFAR-100
Usage:
    python Q1/test.py --weights Q1/weights/LoRA_r4_a4_d0.1_best.pth --use_lora True --rank 4 --alpha 4
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from Q1.train import LoRAQKV, build_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--use_lora", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--wandb_project", type=str, default="Assignment5-Q1")
    return parser.parse_args()


def get_test_loader(batch_size):
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def test(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_classwise_accuracy(preds, labels, save_path="Q1/classwise_accuracy.png"):
    num_classes = 100
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    for p, l in zip(preds, labels):
        class_total[l] += 1
        class_correct[l] += int(p == l)
    class_acc = class_correct / class_total

    plt.figure(figsize=(24, 6))
    plt.bar(range(num_classes), class_acc, color="steelblue", alpha=0.8)
    plt.xlabel("Class Index")
    plt.ylabel("Accuracy")
    plt.title("CIFAR-100 Class-wise Test Accuracy")
    plt.axhline(
        y=class_acc.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {class_acc.mean():.3f}",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved class-wise accuracy plot: {save_path}")
    return class_acc


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f"TEST_{'LoRA' if args.use_lora else 'Baseline'}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    loader = get_test_loader(args.batch_size)
    model, _ = build_model(args.use_lora, args.rank, args.alpha, args.dropout)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)

    preds, labels = test(model, loader, device)
    overall_acc = (preds == labels).mean()
    print(f"\nOverall Test Accuracy: {overall_acc:.4f}")

    class_acc = plot_classwise_accuracy(preds, labels)

    # Log to WandB
    wandb.log({"test/overall_accuracy": overall_acc})
    hist_img = wandb.Image(
        "Q1/classwise_accuracy.png", caption="Class-wise Test Accuracy"
    )
    wandb.log({"test/classwise_accuracy_histogram": hist_img})

    wandb.finish()


if __name__ == "__main__":
    main()
