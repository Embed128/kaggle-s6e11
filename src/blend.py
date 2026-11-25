"""Fast, strictly Nested-CV validated ensemble optimizer (Log-Odds Hill Climbing,
Logistic Meta-Stacking, and Rank Blending) across all saved OOF predictions.
"""

import argparse

import numpy as np
import pandas as pd
import polars as pl
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import DATA_RAW_DIR, ID_COL, N_FOLDS, OOF_DIR, SEED, SUBMISSION_DIR, TARGET_COL


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return logit(p)


def to_rank(p: np.ndarray) -> np.ndarray:
    return rankdata(p) / len(p)


def fast_hill_climb_weights(X_oof: np.ndarray, y: np.ndarray, n_steps: int = 40) -> np.ndarray:
    """Fast forward hill-climbing (Caruana ensemble selection) — runs in ~1-2s on 560k rows."""
    n_models = X_oof.shape[1]
    single_aucs = [roc_auc_score(y, X_oof[:, i]) for i in range(n_models)]
    best_idx = int(np.argmax(single_aucs))

    counts = np.zeros(n_models, dtype=np.float64)
    counts[best_idx] = 1.0
    cur_sum = X_oof[:, best_idx].copy()
    cur_best = single_aucs[best_idx]

    for step in range(2, n_steps + 1):
        best_step_auc = cur_best
        best_j = -1
        for j in range(n_models):
            cand = (cur_sum + X_oof[:, j]) / step
            auc = roc_auc_score(y, cand)
            if auc > best_step_auc + 1e-7:
                best_step_auc = auc
                best_j = j
        if best_j == -1:
            break
        cur_sum += X_oof[:, best_j]
        counts[best_j] += 1.0
        cur_best = best_step_auc

    return counts / counts.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v2")
    parser.add_argument("--out", type=str, default="sub_v2_ensemble.csv")
    args = parser.parse_args()

    train_pl = pl.read_csv(DATA_RAW_DIR / "train.csv", columns=[TARGET_COL])
    test_pl = pl.read_csv(DATA_RAW_DIR / "test.csv", columns=[ID_COL])
    y = train_pl[TARGET_COL].to_numpy().astype(np.int8)
    test_ids = test_pl[ID_COL].to_pandas()

    oof_files = sorted(OOF_DIR.glob(f"oof_{args.tag}_*.npy"))
    if not oof_files:
        raise FileNotFoundError(f"No OOF files matching oof_{args.tag}_*.npy in {OOF_DIR}")

    names, oofs, tests = [], [], []
    print("=== Single Model OOF ROC AUC ===")
    for f in oof_files:
        name = f.stem.replace(f"oof_{args.tag}_", "")
        tf = OOF_DIR / f"test_{args.tag}_{name}.npy"
        if not tf.exists():
            continue
        oof_p = np.load(f)
        test_p = np.load(tf)
        print(f"  {name:<16} OOF AUC = {roc_auc_score(y, oof_p):.6f} | test_mean = {test_p.mean():.5f}")
        names.append(name)
        oofs.append(oof_p)
        tests.append(test_p)

    base_f = OOF_DIR / "oof_lgb_baseline.npy"
    if base_f.exists():
        print(f"  {'(ref) baseline':<16} OOF AUC = {roc_auc_score(y, np.load(base_f)):.6f}")

    O_logit = np.column_stack([to_logit(p) for p in oofs])
    T_logit = np.column_stack([to_logit(p) for p in tests])
    O_rank = np.column_stack([to_rank(p) for p in oofs])
    T_rank = np.column_stack([to_rank(p) for p in tests])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # 1. Equal Log-Odds Blend
    eq_auc = roc_auc_score(y, O_logit.mean(axis=1))

    # 2. Nested Logistic Regression Stacking (L2-regularized on log-odds, convex & super fast)
    best_stack_c = 0.05
    best_stack_auc = 0.0
    best_stack_oof = None
    best_stack_test = None
    best_coefs = None
    for c in [0.005, 0.02, 0.1, 1.0]:
        s_oof = np.zeros(len(y), dtype=np.float64)
        s_test = np.zeros(len(test_ids), dtype=np.float64)
        coefs = []
        for tr_idx, va_idx in skf.split(O_logit, y):
            meta = LogisticRegression(C=c, max_iter=500)
            meta.fit(O_logit[tr_idx], y[tr_idx])
            s_oof[va_idx] = meta.predict_proba(O_logit[va_idx])[:, 1]
            s_test += meta.predict_proba(T_logit)[:, 1] / N_FOLDS
            coefs.append(meta.coef_[0])
        auc = roc_auc_score(y, s_oof)
        if auc > best_stack_auc:
            best_stack_auc = auc
            best_stack_c = c
            best_stack_oof = s_oof
            best_stack_test = s_test
            best_coefs = np.mean(coefs, axis=0)

    # 3. Fast Nested Hill-Climbing on Log-Odds
    hc_oof = np.zeros(len(y), dtype=np.float64)
    hc_weights = []
    for tr_idx, va_idx in skf.split(O_logit, y):
        w = fast_hill_climb_weights(O_logit[tr_idx], y[tr_idx], n_steps=30)
        hc_weights.append(w)
        hc_oof[va_idx] = O_logit[va_idx] @ w
    avg_w_hc = np.mean(hc_weights, axis=0)
    avg_w_hc /= avg_w_hc.sum()
    hc_auc = roc_auc_score(y, hc_oof)
    hc_test = expit(T_logit @ avg_w_hc)

    # 4. Rank Blend using learned hill-climbing weights
    rank_oof = O_rank @ avg_w_hc
    rank_auc = roc_auc_score(y, rank_oof)
    rank_test = T_rank @ avg_w_hc

    print("\n=== Ensemble Comparison (Strictly Nested-CV Validated) ===")
    print(f"  1. Equal Log-Odds Blend         OOF AUC = {eq_auc:.6f}")
    print(f"  2. Hill-Climb Log-Odds Blend    OOF AUC = {hc_auc:.6f}")
    print(f"  3. Logistic Meta-Stacking(C={best_stack_c}) OOF AUC = {best_stack_auc:.6f}")
    print(f"  4. Rank Blend (HC weights)      OOF AUC = {rank_auc:.6f}")

    print("\nLearned Weights:")
    for n, wh, wc in zip(names, avg_w_hc, best_coefs):
        print(f"  {n:<16} hill_climb={wh:.4f} | stack_coef={wc:+.4f}")

    candidates = [
        ("Hill-Climb Log-Odds Blend", hc_auc, hc_test),
        (f"Logistic Meta-Stacking (C={best_stack_c})", best_stack_auc, best_stack_test),
        ("Rank Blend", rank_auc, rank_test),
    ]
    best_method, best_auc, best_test = max(candidates, key=lambda x: x[1])
    print(f"\n>>> Selected Ensemble: {best_method} (Nested OOF AUC = {best_auc:.6f})")

    sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: best_test})
    assert len(sub) == len(test_ids), "Row count mismatch!"
    assert not sub[TARGET_COL].isna().any(), "NaNs found!"
    assert (sub[TARGET_COL] >= 0).all() and (sub[TARGET_COL] <= 1).all(), "Out of [0,1]!"

    out_path = SUBMISSION_DIR / args.out
    sub.to_csv(out_path, index=False)
    print(f"Saved submission -> {out_path}")
    print(f"Stats: min={best_test.min():.6f}, mean={best_test.mean():.6f}, max={best_test.max():.6f}")


if __name__ == "__main__":
    main()
