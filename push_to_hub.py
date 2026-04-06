"""
Q1: Push best model to HuggingFace Hub
Usage:
    python Q1/push_to_hub.py --weights Q1/weights/LoRA_r4_a4_d0.1_best.pth --repo_name your-username/vit-cifar100-lora
"""

import argparse
import torch
import timm
from huggingface_hub import HfApi, create_repo
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   type=str, required=True)
    parser.add_argument("--repo_name", type=str, required=True, help="e.g. username/vit-cifar100-lora")
    return parser.parse_args()


def main():
    args = get_args()
    api  = HfApi()

    # Create repo if not exists
    try:
        create_repo(args.repo_name, exist_ok=True)
        print(f"Repo ready: https://huggingface.co/{args.repo_name}")
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Upload weights file
    api.upload_file(
        path_or_fileobj=args.weights,
        path_in_repo=os.path.basename(args.weights),
        repo_id=args.repo_name,
        repo_type="model",
    )
    print(f"✅ Uploaded {args.weights} to https://huggingface.co/{args.repo_name}")

    # Upload a model card
    model_card = f"""---
tags:
- image-classification
- ViT
- LoRA
- CIFAR-100
---

# ViT-S Fine-tuned on CIFAR-100 with LoRA

This model is a ViT-Small (patch16/224) fine-tuned on CIFAR-100 using LoRA (Low-Rank Adaptation).

## Model Details
- Base model: `vit_small_patch16_224` (pretrained on ImageNet via timm)
- Fine-tuned on: CIFAR-100 (100 classes)
- LoRA applied to: Q, K, V attention weights

## Usage
```python
import timm, torch
model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=100)
model.load_state_dict(torch.load('model.pth'))
```
"""
    with open("README_hf.md", "w") as f:
        f.write(model_card)

    api.upload_file(
        path_or_fileobj="README_hf.md",
        path_in_repo="README.md",
        repo_id=args.repo_name,
        repo_type="model",
    )
    print("✅ Model card uploaded.")


if __name__ == "__main__":
    main()
