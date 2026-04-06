"""
Q2(ii): Adversarial Detector using ResNet-34 (Binary Classification: clean vs adversarial)
Supports both PGD and BIM attacks via IBM ART.

Usage:
    # Train detector for PGD
    python Q2/detector.py --attack pgd --weights Q2/weights/resnet18_clean_best.pth

    # Train detector for BIM
    python Q2/detector.py --attack bim --weights Q2/weights/resnet18_clean_best.pth
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
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import classification_report

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod

from Q2.train_resnet18 import build_resnet18


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, default="pgd", choices=["pgd", "bim"])
    parser.add_argument(
        "--weights", type=str, default="Q2/weights/resnet18_clean_best.pth"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--save_dir", type=str, default="Q2/weights")
    parser.add_argument("--wandb_project", type=str, default="Assignment5-Q2")
    return parser.parse_args()


# ─────────────────────────────────────────
# Generate Adversarial Dataset
# ─────────────────────────────────────────
def generate_adversarial_data(victim_model, loader, attack_type, eps, device):
    """Generate adversarial examples using IBM ART."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])

    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=victim_model,
        loss=criterion,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(((0 - mean) / std).min().item(), ((1 - mean) / std).max().item()),
    )

    if attack_type == "pgd":
        attack = ProjectedGradientDescent(
            estimator=classifier,
            eps=eps,
            eps_step=eps / 10,
            max_iter=40,
            targeted=False,
        )
    else:  # bim
        attack = BasicIterativeMethod(
            estimator=classifier,
            eps=eps,
            eps_step=eps / 10,
            max_iter=40,
        )

    clean_images, adv_images = [], []
    print(f"Generating {attack_type.upper()} adversarial examples...")
    for images, _ in tqdm(loader):
        adv = attack.generate(images.numpy())
        clean_images.append(images.numpy())
        adv_images.append(adv)

    clean_images = np.concatenate(clean_images, axis=0)
    adv_images = np.concatenate(adv_images, axis=0)

    # Labels: 0 = clean, 1 = adversarial
    all_images = np.concatenate([clean_images, adv_images], axis=0)
    all_labels = np.array([0] * len(clean_images) + [1] * len(adv_images))

    # Shuffle
    idx = np.random.permutation(len(all_labels))
    return all_images[idx], all_labels[idx]


# ─────────────────────────────────────────
# Detector Model (ResNet-34 binary)
# ─────────────────────────────────────────
def build_detector():
    model = models.resnet34(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 2)  # Binary: clean vs adversarial
    return model


# ─────────────────────────────────────────
# Train Detector
# ─────────────────────────────────────────
def train_detector(model, train_loader, val_loader, epochs, device, run):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch} [Train]", leave=False
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
        train_acc = correct / total
        scheduler.step()

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Val", leave=False):
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        run.log(
            {
                "epoch": epoch,
                "detector/train_acc": train_acc,
                "detector/val_acc": val_acc,
                "detector/train_loss": train_loss / total,
                "detector/val_loss": val_loss / val_total,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc


# ─────────────────────────────────────────
# Visualize Samples
# ─────────────────────────────────────────
def visualize_samples(clean_imgs, adv_imgs, attack_type, save_dir, run):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    def denorm(img):
        img = img * std[:, None, None] + mean[:, None, None]
        return np.clip(img.transpose(1, 2, 0), 0, 1)

    n = 10
    fig, axes = plt.subplots(2, n, figsize=(20, 4))
    for i in range(n):
        axes[0, i].imshow(denorm(clean_imgs[i]))
        axes[1, i].imshow(denorm(adv_imgs[i]))
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Clean", fontsize=10)
    axes[1, 0].set_ylabel(f"{attack_type.upper()} Adv", fontsize=10)
    plt.suptitle(f"Clean vs {attack_type.upper()} Adversarial Samples")
    plt.tight_layout()

    path = os.path.join(save_dir, f"{attack_type}_samples.png")
    plt.savefig(path, dpi=150)
    plt.close()
    run.log({f"{attack_type}_samples": wandb.Image(path)})
    print(f"Saved: {path}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("Q2/results", exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        name=f"Detector_{args.attack.upper()}",
        config=vars(args),
    )

    # Load victim ResNet18
    victim_model = build_resnet18().to(device)
    victim_model.load_state_dict(torch.load(args.weights, map_location=device))
    victim_model.eval()

    # Data
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader_raw = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader_raw = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Generate adversarial data
    print(f"\nGenerating {args.attack.upper()} adversarial training data...")
    train_images, train_labels = generate_adversarial_data(
        victim_model, train_loader_raw, args.attack, args.eps, device
    )

    print(f"\nGenerating {args.attack.upper()} adversarial test data...")
    test_images, test_labels = generate_adversarial_data(
        victim_model, test_loader_raw, args.attack, args.eps, device
    )

    # Visualize 10 samples on WandB
    n_clean = len(test_images) // 2
    visualize_samples(
        test_images[:10],
        test_images[n_clean : n_clean + 10],
        args.attack,
        "Q2/results",
        run,
    )

    # Build TensorDatasets
    def make_dataset(images, labels):
        X = torch.tensor(images, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        return TensorDataset(X, y)

    train_ds = make_dataset(train_images, train_labels)
    test_ds = make_dataset(test_images, test_labels)

    split = int(0.8 * len(train_ds))
    train_subset = torch.utils.data.Subset(train_ds, range(split))
    val_subset = torch.utils.data.Subset(train_ds, range(split, len(train_ds)))

    train_dl = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_dl = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Train detector
    detector = build_detector().to(device)
    print(f"\nTraining {args.attack.upper()} Adversarial Detector...")
    best_val_acc = train_detector(detector, train_dl, val_dl, args.epochs, device, run)

    # Save
    save_path = f"{args.save_dir}/detector_{args.attack}_best.pth"
    torch.save(detector.state_dict(), save_path)
    print(f"Saved detector: {save_path}")

    # Final test evaluation
    detector.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_dl:
            images = images.to(device)
            preds = detector(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"\n✅ {args.attack.upper()} Detector Test Accuracy: {test_acc:.4f}")
    if test_acc < 0.70:
        print("⚠️  Warning: Below 70% target. Consider more epochs or tuning.")

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels, all_preds, target_names=["Clean", "Adversarial"]
        )
    )

    run.log(
        {
            f"detector_{args.attack}/test_accuracy": test_acc,
            f"detector_{args.attack}/best_val_acc": best_val_acc,
        }
    )
    run.finish()


if __name__ == "__main__":
    main()
