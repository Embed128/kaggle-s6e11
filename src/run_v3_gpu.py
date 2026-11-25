"""One-command V3 GPU runner for RTX 5060 Ti Windows PC (or any CUDA/CPU machine).

Runs:
  1. 10-Fold GBDT + CatBoost GPU + GAM suite (src/train_v3_gpu.py)
  2. 10-Fold PyTorch CUDA TabM BatchEnsemble NN (src/train_tabm_gpu.py, auto-skips cleanly if torch not installed)
  3. Strictly Nested-CV multi-family ensemble optimization (src/blend.py)

Usage:
  uv run python src/run_v3_gpu.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_step(cmd: list[str], title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"STEP: {title}")
    print(f"CMD : {' '.join(cmd)}")
    print(f"{'=' * 65}\n", flush=True)
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        raise SystemExit(f"Step failed with exit code {res.returncode}: {title}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--tag", type=str, default="v3")
    parser.add_argument("--skip-nn", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    force_flag = ["--force"] if args.force else []

    # 1. GBDT + CatBoost GPU + GAM suite
    run_step(
        [
            py, "src/train_v3_gpu.py",
            "--device", args.device,
            "--n-folds", str(args.n_folds),
            "--tag", args.tag,
            *force_flag,
        ],
        f"10-Fold GBDT + CatBoost GPU + GAM Suite (tag={args.tag})",
    )

    # 2. PyTorch CUDA TabM NN (gracefully skips if PyTorch not installed)
    if not args.skip_nn:
        run_step(
            [
                py, "src/train_tabm_gpu.py",
                "--n-folds", str(args.n_folds),
                "--tag", args.tag,
                *force_flag,
            ],
            f"10-Fold PyTorch CUDA TabM BatchEnsemble NN (tag={args.tag})",
        )

    # 3. Strictly Nested-CV Ensemble & Submissions
    run_step(
        [
            py, "src/blend.py",
            "--tag", args.tag,
            "--n-folds", str(args.n_folds),
            "--out", f"sub_{args.tag}_gpu_ensemble.csv",
        ],
        f"Nested-CV Ensemble Optimization (tag={args.tag})",
    )


if __name__ == "__main__":
    main()
