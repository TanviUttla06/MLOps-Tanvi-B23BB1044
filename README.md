# MLOps-Tanvi-B23BB1044
# Assignment 5 — ViT LoRA Fine-tuning & Adversarial Attacks

> WandB Project: [link here]  
> HuggingFace Model: [link here]

---

## 📁 Project Structure

```
Assignment-5/
├── Q1/
│   ├── train.py           # ViT-S fine-tuning with/without LoRA
│   ├── test.py            # Class-wise accuracy + WandB logging
│   ├── optuna_search.py   # Optuna hyperparameter tuning
│   ├── run_all.py         # Run all rank/alpha combinations
│   ├── push_to_hub.py     # Upload best model to HuggingFace
│   └── weights/           # Saved model weights
├── Q2/
│   ├── train_resnet18.py  # Train ResNet18 on clean CIFAR-10
│   ├── fgsm_attack.py     # FGSM scratch vs IBM ART
│   ├── detector.py        # Adversarial detector (PGD / BIM)
│   ├── weights/           # Saved model weights
│   └── results/           # Visualizations
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🐳 Docker Setup (Required)

```bash
# Build Docker image
docker build -t assignment5 .

# Run container with GPU
docker run --gpus all -it \
  -v $(pwd):/app \
  -e WANDB_API_KEY=your_wandb_key \
  -e HUGGINGFACE_TOKEN=your_hf_token \
  assignment5 bash
```

---

## 📦 Install Libraries (inside Docker or local)

```bash
pip install -r requirements.txt

# Login to WandB
wandb login

# Login to HuggingFace
huggingface-cli login
```

---

## 🔬 Q1: ViT-S Fine-tuning on CIFAR-100

### Run all experiments (baseline + all LoRA combinations)
```bash
python Q1/run_all.py --epochs 10
```

### Run individual experiment

Baseline (no LoRA):
```bash
python Q1/train.py --use_lora False --epochs 10
```

With LoRA (example: rank=4, alpha=4):
```bash
python Q1/train.py --use_lora True --rank 4 --alpha 4 --dropout 0.1 --epochs 10
```

All combinations of rank ∈ {2,4,8} and alpha ∈ {2,4,8} with dropout=0.1.

### Test a model
```bash
python Q1/test.py \
  --weights Q1/weights/LoRA_r4_a4_d0.1_best.pth \
  --use_lora True --rank 4 --alpha 4
```

### Optuna hyperparameter search
```bash
python Q1/optuna_search.py --n_trials 20 --epochs 5
```

### Push best model to HuggingFace
```bash
python Q1/push_to_hub.py \
  --weights Q1/weights/LoRA_r4_a4_d0.1_best.pth \
  --repo_name your-username/vit-cifar100-lora
```

---

## Q1 Results
###Train - val Accuracy Table
|Experiment No. |  train/acc   |  val/acc |
|--------------|--------------|----------|
|Baseline       | 0.83914      | 0.7968   |


### Test Accuracy Table

| LoRA | Rank | Alpha | Dropout | Test Accuracy | Trainable Params |
|------|------|-------|---------|--------------|-----------------|
| No   | -    | -     | -       | 0.7969       | 38,500 / 21,704,164 (0.18%)              |
| Yes  | 2    | 2     | 0.1     | 0.8958       | 149,092 / 21,814,756 (0.68%)             |
| Yes  | 2    | 4     | 0.1     | 0.8969       | 149,092 / 21,814,756 (0.68%)             |
| Yes  | 2    | 8     | 0.1     | 0.9001       | 149,092 / 21,814,756 (0.68%)             |
| Yes  | 4    | 2     | 0.1     | 0.8969       | 259,684 / 21,925,348 (1.18%)             |
| Yes  | 4    | 4     | 0.1     | 0.8977       | 259,684 / 21,925,348 (1.18%)             |
| Yes  | 4    | 8     | 0.1     | 0.9019       | 259,684 / 21,925,348 (1.18%)             |
| Yes  | 8    | 2     | 0.1     | 0.8966       | 480,868 / 22,146,532 (2.17%)             |
| Yes  | 8    | 4     | 0.1     | 0.8999       | 480,868 / 22,146,532 (2.17%)             |
| Yes  | 8    | 8     | 0.1     | 0.9028       | 480,868 / 22,146,532 (2.17%)             |

Graphs:
<img width="891" height="463" alt="image" src="https://github.com/user-attachments/assets/ddd774d3-8e98-42a6-9968-67bd5f98a438" />

<img width="891" height="479" alt="image" src="https://github.com/user-attachments/assets/86b89d64-f03c-42ba-9b11-8a38dcc9638e" />

---

## 🛡️ Q2: Adversarial Attacks

### Step 1: Train ResNet18 on clean CIFAR-10
```bash
python Q2/train_resnet18.py --epochs 30
```
Target: ≥ 72% test accuracy.

### Step 2: FGSM Attack (Scratch vs IBM ART)
```bash
python Q2/fgsm_attack.py \
  --weights Q2/weights/resnet18_clean_best.pth
```

### Step 3: Train Adversarial Detector (PGD)
```bash
python Q2/detector.py \
  --attack pgd \
  --weights Q2/weights/resnet18_clean_best.pth \
  --epochs 10
```

### Step 4: Train Adversarial Detector (BIM)
```bash
python Q2/detector.py \
  --attack bim \
  --weights Q2/weights/resnet18_clean_best.pth \
  --epochs 10
```

---

## Q2 Results

| Model        | Attack | Clean Acc | Adv Acc | Detection Acc |
|--------------|--------|-----------|---------|--------------|
| ResNet18     | None   | 0.9392    | -       | -            |
| ResNet18     | FGSM   | 0.942     | 0.7080(epsilon=0.01)  | -            |
| ResNet34 Det | PGD    | -         | -       | 0.5          |
| ResNet34 Det | BIM    | -         | -       | TBD          |

Results of Q2:
<img width="3000" height="900" alt="fgsm_comparison_eps0 1" src="https://github.com/user-attachments/assets/355b283a-5e5e-4808-905a-07a818c86391" />
<img width="3000" height="600" alt="pgd_samples" src="https://github.com/user-attachments/assets/f558a669-de7d-4be8-aeaf-258be8842af3" />
<img width="1200" height="750" alt="fgsm_eps_vs_acc" src="https://github.com/user-attachments/assets/524123f6-3be5-4bd5-90cc-917467801cfa" />


