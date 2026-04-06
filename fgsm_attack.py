"""
Q2(i): FGSM Attack - From Scratch vs IBM ART
Usage:
    python Q2/fgsm_attack.py --weights Q2/weights/resnet18_clean_best.pth
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import wandb
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# IBM ART
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from Q2.train_resnet18 import build_resnet18


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default="Q2/weights/resnet18_clean_best.pth"
    )
    parser.add_argument(
        "--eps_list", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--n_samples", type=int, default=1000, help="Samples for attack evaluation"
    )
    parser.add_argument("--wandb_project", type=str, default="Assignment5-Q2")
    return parser.parse_args()


def get_test_loader(batch_size, n_samples=None):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    if n_samples:
        indices = torch.randperm(len(dataset))[:n_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# ─────────────────────────────────────────
# FGSM from Scratch
# ─────────────────────────────────────────
def fgsm_scratch(model, images, labels, eps, device):
    """Manual FGSM implementation."""
    images = images.to(device).clone().requires_grad_(True)
    labels = labels.to(device)
    criterion = nn.CrossEntropyLoss()

    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()

    perturbation = eps * images.grad.sign()
    adv_images = (images + perturbation).detach()
    return adv_images


def evaluate_accuracy(model, loader, device, attack_fn=None, eps=None):
    model.eval()
    correct, total = 0, 0
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        if attack_fn is not None:
            images = attack_fn(model, images, labels, eps, device)
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return correct / total


# ─────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────
def visualize_fgsm(model, loader, device, eps, save_dir, wandb_run):
    """Show original vs adversarial images (scratch vs ART)."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])

    images, labels = next(iter(loader))
    images, labels = images[:10].to(device), labels[:10].to(device)

    # Scratch FGSM
    adv_scratch = fgsm_scratch(model, images, labels, eps, device)

    # ART FGSM
    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(((0 - mean) / std).min().item(), ((1 - mean) / std).max().item()),
    )
    fgsm_art = FastGradientMethod(estimator=classifier, eps=eps)
    adv_art = torch.tensor(fgsm_art.generate(images.cpu().numpy()), device=device)

    def denorm(t):
        t = t.cpu() * std[:, None, None] + mean[:, None, None]
        return t.clamp(0, 1).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    cifar10_classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    for i in range(10):
        axes[0, i].imshow(denorm(images[i]))
        axes[0, i].set_title(cifar10_classes[labels[i].item()], fontsize=7)
        axes[1, i].imshow(denorm(adv_scratch[i]))
        axes[2, i].imshow(denorm(adv_art[i]))
        for ax in axes[:, i]:
            ax.axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("FGSM Scratch", fontsize=9)
    axes[2, 0].set_ylabel("FGSM ART", fontsize=9)
    plt.suptitle(f"FGSM Comparison (eps={eps})", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"fgsm_comparison_eps{eps}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    wandb_run.log({f"fgsm_comparison/eps{eps}": wandb.Image(save_path)})
    print(f"Saved: {save_path}")


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("Q2/results", exist_ok=True)

    run = wandb.init(
        project=args.wandb_project, name="FGSM_Scratch_vs_ART", config=vars(args)
    )

    loader = get_test_loader(args.batch_size, args.n_samples)
    model = build_resnet18().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # ART classifier
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(((0 - mean) / std).min().item(), ((1 - mean) / std).max().item()),
    )

    # Clean accuracy
    clean_acc = evaluate_accuracy(model, loader, device)
    print(f"\nClean Accuracy: {clean_acc:.4f}")
    run.log({"clean_accuracy": clean_acc})

    results = {"eps": [], "scratch_acc": [], "art_acc": []}
    for eps in args.eps_list:
        print(f"\n--- eps = {eps} ---")

        # Scratch FGSM
        scratch_acc = evaluate_accuracy(
            model, loader, device, attack_fn=fgsm_scratch, eps=eps
        )

        # ART FGSM (collect all batches)
        fgsm_art = FastGradientMethod(estimator=classifier, eps=eps)
        all_preds, all_labels = [], []
        for images, labels in tqdm(loader, desc="ART FGSM", leave=False):
            adv = fgsm_art.generate(images.numpy())
            with torch.no_grad():
                preds = model(torch.tensor(adv, device=device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        art_acc = (np.array(all_preds) == np.array(all_labels)).mean()

        print(f"  Scratch FGSM Acc: {scratch_acc:.4f} | ART FGSM Acc: {art_acc:.4f}")
        run.log({"eps": eps, "scratch_fgsm_acc": scratch_acc, "art_fgsm_acc": art_acc})
        results["eps"].append(eps)
        results["scratch_acc"].append(scratch_acc)
        results["art_acc"].append(art_acc)

    # Plot perturbation vs performance
    plt.figure(figsize=(8, 5))
    plt.plot(results["eps"], results["scratch_acc"], marker="o", label="FGSM Scratch")
    plt.plot(results["eps"], results["art_acc"], marker="s", label="FGSM ART")
    plt.axhline(
        y=clean_acc, color="green", linestyle="--", label=f"Clean ({clean_acc:.3f})"
    )
    plt.xlabel("Epsilon (Perturbation Strength)")
    plt.ylabel("Accuracy")
    plt.title("FGSM: Perturbation Strength vs Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("Q2/results/fgsm_eps_vs_acc.png", dpi=150)
    run.log({"fgsm_eps_vs_acc": wandb.Image("Q2/results/fgsm_eps_vs_acc.png")})

    # Visualize 10 samples
    vis_loader = get_test_loader(10, 10)
    visualize_fgsm(
        model, vis_loader, device, eps=0.1, save_dir="Q2/results", wandb_run=run
    )

    run.finish()
    print("\n✅ FGSM analysis complete!")


if __name__ == "__main__":
    main()
