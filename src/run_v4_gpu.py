"""One-command V4 Champion GPU Runner for RTX 5060 Ti 16GB VRAM + 16GB RAM Windows PC.

Priority order (highest AUC leap first, 100% resumable per fold & per model!):
  STEP 1: 10-Fold Target-First Digit-Token Causal LM (rev: high-signal-first order)  ~20-25 min
  STEP 2: 10-Fold Target-First Digit-Token Causal LM (fwd: forward/natural order)    ~20-25 min
  STEP 3: 10-Fold Target-First Causal LM (disc: joint generative + Bayes-LLR BCE)    ~35-40 min
  STEP 4: 10-Fold RealMLP-PBLD (high-freq periodic embeddings sigma=5 + LOO cluster) ~15-20 min
  STEP 5: 10-Fold LOO-Cluster GBDT + GAM (lr_gam_loo, xgb_d2_loo, lgb_d2_loo)       ~12-15 min
  STEP 6: V3 + V4 Grand Ensemble + Zero-Fraud Zones 1-4 -> sub_v4_champion.csv       ~30 sec

Tip: Every step automatically saves checkpoints after every fold!
     You can also run `uv run python src/blend_v4.py` anytime between steps to get an instant upgraded submission!
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_step(cmd: list[str], title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"STEP: {title}")
    print(f"CMD : {' '.join(cmd)}")
    print(f"{'=' * 70}\n", flush=True)
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        raise SystemExit(f"Step failed with exit code {res.returncode}: {title}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--lm-epochs", type=int, default=16)
    parser.add_argument("--skip-disc-lm", action="store_true", help="Skip slower joint disc LM if short on time")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    ff = ["--force"] if args.force else []
    nf = str(args.n_folds)

    # 1. Target-First Digit-Token Causal LM (Reversed / High-Signal-First Order)
    run_step(
        [py, "src/train_tf_lm_gpu.py", "--mode", "rev", "--n-folds", nf, "--epochs", str(args.lm_epochs), *ff],
        "STEP 1/6: Target-First Digit-Token Causal LM (rev order, pure Bayes LLR)",
    )

    # Interim blend right after first LM finishes so you already have a huge jump saved!
    run_step(
        [py, "src/blend_v4.py", "--tags", "v3,v4", "--n-folds", nf, "--out", "sub_v4_after_lm1.csv"],
        "INTERIM BLEND: V3 (10 models) + tf_lm_rev + Zones 1-4",
    )

    # 2. Target-First Digit-Token Causal LM (Forward / Ascending Order)
    run_step(
        [py, "src/train_tf_lm_gpu.py", "--mode", "fwd", "--n-folds", nf, "--epochs", str(args.lm_epochs), *ff],
        "STEP 2/6: Target-First Digit-Token Causal LM (fwd order, pure Bayes LLR)",
    )

    # 3. Target-First Causal LM (Joint Generative + Bayes-LLR Discriminative Calibration)
    if not args.skip_disc_lm:
        run_step(
            [py, "src/train_tf_lm_gpu.py", "--mode", "disc", "--n-folds", nf, "--epochs", str(min(14, args.lm_epochs)), *ff],
            "STEP 3/6: Target-First Causal LM (disc mode: generative + Bayes-BCE)",
        )

    # 4. RealMLP-PBLD (High-Frequency Periodic Embeddings sigma=5.0 + 8-Member Ensemble + LOO Cluster)
    run_step(
        [py, "src/train_realmlp_gpu.py", "--n-folds", nf, *ff],
        "STEP 4/6: RealMLP-PBLD GPU (sigma=5.0 periodic embeddings + LOO cluster)",
    )

    # 5. LOO-Cluster-Enhanced GAM + XGBoost GPU + LightGBM
    run_step(
        [py, "src/train_v4_loo_models.py", "--device", "auto", "--n-folds", nf, *ff],
        "STEP 5/6: LOO-Cluster GAM + XGBoost GPU + LightGBM",
    )

    # 6. Final Grand V3 + V4 Champion Ensemble + Zero-Fraud Zones 1-4
    run_step(
        [py, "src/blend_v4.py", "--tags", "v3,v4", "--n-folds", nf, "--out", "sub_v4_champion.csv"],
        "STEP 6/6: V3 + V4 Grand Ensemble (17 Models) + Zero-Fraud Zones 1-4",
    )


if __name__ == "__main__":
    main()
