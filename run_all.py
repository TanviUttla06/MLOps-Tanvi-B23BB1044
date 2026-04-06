"""
Q1: Run all LoRA combinations (rank x alpha) + baseline
Usage:
    python Q1/run_all.py --epochs 10
"""

import subprocess
import sys

RANKS   = [2, 4, 8]
ALPHAS  = [2, 4, 8]
DROPOUT = 0.1
EPOCHS  = 10


def run(cmd):
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, check=True)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    # 1. Baseline (no LoRA)
    run([
        sys.executable, "Q1/train.py",
        "--use_lora", "False",
        "--epochs", str(args.epochs),
    ])

    # 2. All LoRA combinations
    exp_no = 1
    for rank in RANKS:
        for alpha in ALPHAS:
            print(f"\n[Experiment {exp_no}] rank={rank}, alpha={alpha}, dropout={DROPOUT}")
            run([
                sys.executable, "Q1/train.py",
                "--use_lora", "True",
                "--rank",    str(rank),
                "--alpha",   str(alpha),
                "--dropout", str(DROPOUT),
                "--epochs",  str(args.epochs),
            ])
            exp_no += 1

    print("\n✅ All Q1 experiments complete!")
