"""V5.2 Fast Direct-AUC Pruned Grand Stacker for Kaggle S6E11 (~45s total on cached 209 models).

Fixes V5.1 L2 dilution:
- Pre-selects the elite diverse pool strictly inside each fold (top 25 AUC + top 20 LR-Logit weight = ~35-45 diverse members, always keeping all LMs, TabPFN, RealMLP, TabM, EBM, GLM, GBDT, and own models!)
- Runs ultra-fast Direct-AUC Caruana Hill-Climbing (60 steps) + Pruned NonNeg L2 Logistic + Pruned Free L2 Logistic
- Outputs BOTH:
    - TRACK A: Pure Base Models Only (Private Shield -> sub_v5_2_pure_shield.csv)
    - TRACK B: Full 209-Stream Grand Champion (-> sub_v5_2_grand_champion.csv)
"""

import gc
import time
import warnings
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import DATA_RAW_DIR, ID_COL, OOF_DIR, SEED, SUBMISSION_DIR, TARGET_COL, ensure_dirs
from train_grand_stack_v5 import (
    PUBLIC_OOF_DATASETS,
    apply_zone_push,
    discover_all_in_dir,
    fit_nonneg_l2_logistic,
    get_zero_fraud_zones,
    to_clean_logit,
    to_probit_rank,
    to_unit_rank,
    valid_pair,
    verify_and_save,
)

warnings.filterwarnings("ignore")


def fast_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Fast exact ROC AUC via argsort (~3-4x faster than sklearn overhead)."""
    order = np.argsort(score, kind="mergesort")
    y_sorted = y[order]
    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos
    # Sum of 1-based ranks of positive examples
    ranks_pos = np.nonzero(y_sorted)[0].astype(np.float64) + 1.0
    return float((ranks_pos.sum() - n_pos * (n_pos + 1) * 0.5) / (n_pos * n_neg))


def fast_direct_auc_hill_climb(X: np.ndarray, y: np.ndarray, cand_idx: np.ndarray, n_steps: int = 60) -> np.ndarray:
    """Fast direct-AUC forward ensemble selection over candidate indices."""
    M = X.shape[1]
    single = np.array([fast_auc(y, X[:, j]) for j in cand_idx])
    best_local = int(np.argmax(single))
    best_j = int(cand_idx[best_local])

    counts = np.zeros(M, dtype=np.float64)
    counts[best_j] = 1.0
    cur_sum = X[:, best_j].astype(np.float64, copy=True)
    cur_auc = float(single[best_local])

    for step in range(2, n_steps + 1):
        step_best = cur_auc
        step_j = -1
        for j in cand_idx:
            auc = fast_auc(y, cur_sum + X[:, j])
            if auc > step_best + 1e-8:
                step_best = auc
                step_j = int(j)
        if step_j == -1:
            break
        cur_sum += X[:, step_j]
        counts[step_j] += 1.0
        cur_auc = step_best
    return counts / counts.sum()


def run_pruned_nested_stack(
    O_lg: np.ndarray,
    T_lg: np.ndarray,
    O_pr: np.ndarray,
    T_pr: np.ndarray,
    y: np.ndarray,
    names: list[str],
    aucs: list[float],
    splits: list,
    zones_tr: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    n_train, M = O_lg.shape
    n_test = T_lg.shape[0]
    t0 = time.time()
    print(f"\n================ {label} (M={M} models) ================", flush=True)

    O_lg_std = ((O_lg - O_lg.mean(axis=0)) / (O_lg.std(axis=0) + 1e-9)).astype(np.float32)
    T_lg_std = ((T_lg - O_lg.mean(axis=0)) / (O_lg.std(axis=0) + 1e-9)).astype(np.float32)
    O_pr32 = O_pr.astype(np.float32)
    T_pr32 = T_pr.astype(np.float32)

    hc_lg_oof = np.zeros(n_train, dtype=np.float64)
    hc_pr_oof = np.zeros(n_train, dtype=np.float64)
    pruned_nn_oof = np.zeros(n_train, dtype=np.float64)
    pruned_nn_test = np.zeros(n_test, dtype=np.float64)
    pruned_lr_oof = np.zeros(n_train, dtype=np.float64)
    pruned_lr_test = np.zeros(n_test, dtype=np.float64)

    hc_lg_ws, hc_pr_ws = [], []

    for f_i, (tr_i, va_i) in enumerate(splits, 1):
        ft0 = time.time()
        y_tr = y[tr_i]
        # Inside-fold candidate screening: top 25 single AUC + top 25 LR-Logit positive weight -> ~35-45 diverse elites!
        clf_screen = LogisticRegression(C=0.02, solver="lbfgs", max_iter=500)
        clf_screen.fit(O_lg_std[tr_i], y_tr)
        lr_c = clf_screen.coef_[0]
        top_auc_idx = np.argsort([-fast_auc(y_tr, O_lg_std[tr_i, j]) for j in range(M)])[:25]
        top_lr_idx = np.argsort(-lr_c)[:25]
        cand_idx = np.unique(np.concatenate([top_auc_idx, top_lr_idx]))

        # 1. Direct-AUC HC on Logits
        w_lg = fast_direct_auc_hill_climb(O_lg_std[tr_i], y_tr, cand_idx, n_steps=60)
        hc_lg_ws.append(w_lg)
        hc_lg_oof[va_i] = O_lg_std[va_i] @ w_lg

        # 2. Direct-AUC HC on Probit-Ranks
        w_pr = fast_direct_auc_hill_climb(O_pr32[tr_i], y_tr, cand_idx, n_steps=60)
        hc_pr_ws.append(w_pr)
        hc_pr_oof[va_i] = O_pr32[va_i] @ w_pr

        # 3. Pruned Non-Negative L2 Logistic on strictly selected subset (zero weak-tree dilution!)
        sel = np.where((w_lg > 0) | (w_pr > 0))[0]
        w_nn, b_nn = fit_nonneg_l2_logistic(O_lg_std[tr_i][:, sel].astype(np.float64), y_tr, l2_reg=2.0)
        pruned_nn_oof[va_i] = O_lg_std[va_i][:, sel] @ w_nn + b_nn
        pruned_nn_test += (T_lg_std[:, sel] @ w_nn + b_nn) / len(splits)

        # 4. Pruned Free L2 Logistic on strictly selected subset
        clf_p = LogisticRegression(C=0.05, solver="lbfgs", max_iter=600)
        clf_p.fit(O_lg_std[tr_i][:, sel], y_tr)
        pruned_lr_oof[va_i] = clf_p.decision_function(O_lg_std[va_i][:, sel])
        pruned_lr_test += clf_p.decision_function(T_lg_std[:, sel]) / len(splits)

        print(
            f"  Fold {f_i}/5 (cands={len(cand_idx)}, sel={len(sel)}): "
            f"HC-Logit={roc_auc_score(y[va_i], hc_lg_oof[va_i]):.5f} | "
            f"HC-Probit={roc_auc_score(y[va_i], hc_pr_oof[va_i]):.5f} | "
            f"Pruned-NN={roc_auc_score(y[va_i], pruned_nn_oof[va_i]):.5f} | "
            f"Pruned-LR={roc_auc_score(y[va_i], pruned_lr_oof[va_i]):.5f} ({time.time()-ft0:.1f}s)",
            flush=True,
        )

    avg_w_lg = np.mean(hc_lg_ws, axis=0)
    avg_w_lg /= avg_w_lg.sum()
    hc_lg_test = T_lg_std @ avg_w_lg

    avg_w_pr = np.mean(hc_pr_ws, axis=0)
    avg_w_pr /= avg_w_pr.sum()
    hc_pr_test = T_pr32 @ avg_w_pr

    # 5. Nested Meta-Blend across the 4 pruned direct-AUC learners
    M4_oof = np.column_stack([
        to_probit_rank(hc_lg_oof),
        to_probit_rank(hc_pr_oof),
        to_probit_rank(pruned_nn_oof),
        to_probit_rank(pruned_lr_oof),
    ]).astype(np.float32)
    M4_test = np.column_stack([
        to_probit_rank(hc_lg_test),
        to_probit_rank(hc_pr_test),
        to_probit_rank(pruned_nn_test),
        to_probit_rank(pruned_lr_test),
    ]).astype(np.float32)

    blend_oof = np.zeros(n_train, dtype=np.float64)
    b_ws = []
    for tr_i, va_i in splits:
        w = fast_direct_auc_hill_climb(M4_oof[tr_i], y[tr_i], np.arange(4), n_steps=40)
        b_ws.append(w)
        blend_oof[va_i] = M4_oof[va_i] @ w
    avg_bw = np.mean(b_ws, axis=0)
    avg_bw /= avg_bw.sum()
    blend_test = M4_test @ avg_bw

    cands = [
        ("1. Direct-AUC HC (Logits)", roc_auc_score(y, hc_lg_oof), hc_lg_oof, hc_lg_test),
        ("2. Direct-AUC HC (ProbitRank)", roc_auc_score(y, hc_pr_oof), hc_pr_oof, hc_pr_test),
        ("3. Pruned NonNeg L2-LR", roc_auc_score(y, pruned_nn_oof), pruned_nn_oof, pruned_nn_test),
        ("4. Pruned Free L2-LR", roc_auc_score(y, pruned_lr_oof), pruned_lr_oof, pruned_lr_test),
        ("5. Meta-Blend (1+2+3+4)", roc_auc_score(y, blend_oof), blend_oof, blend_test),
    ]
    print(f"\n--- Summary [{label}] ({time.time()-t0:.1f}s) ---")
    for nm, auc, oo, _ in cands:
        za = roc_auc_score(y, apply_zone_push(oo, zones_tr))
        print(f"  {nm:<32} Raw Nested OOF = {auc:.6f} | +Zones 1-4 = {za:.6f} ({za - auc:+.6f})")

    print(f"  Meta-Blend weights: HC-Logit={avg_bw[0]:.3f}, HC-Probit={avg_bw[1]:.3f}, Pruned-NN={avg_bw[2]:.3f}, Pruned-LR={avg_bw[3]:.3f}")
    print("  Top 18 Selected Models (Direct-AUC HC Logit weights):")
    for idx in np.argsort(-avg_w_lg)[:18]:
        if avg_w_lg[idx] < 1e-4:
            break
        print(f"    {names[idx]:<48} auc={aucs[idx]:.6f} | w_logit={avg_w_lg[idx]:.4f} | w_probit={avg_w_pr[idx]:.4f}")

    best_nm, best_raw, best_oof, best_test = max(cands, key=lambda x: x[1])
    best_z = roc_auc_score(y, apply_zone_push(best_oof, zones_tr))
    fold_str = " | ".join(f"F{i+1}:{roc_auc_score(y[va], apply_zone_push(best_oof, zones_tr)[va]):.5f}" for i, (_, va) in enumerate(splits))
    print(f"  >>> WINNER [{label}]: {best_nm} | Raw={best_raw:.6f} | +Zones={best_z:.6f}", flush=True)
    print(f"  >>> Folds (+Zones): {fold_str}", flush=True)
    return best_oof, best_test, best_raw, best_z


def main() -> None:
    ensure_dirs()
    t0 = time.time()

    zone_cols = [
        "id", "velocity_score", "merchant_risk_score", "cvv_retry_count",
        "prior_disputes", "is_new_merchant", "is_foreign_transaction",
        "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    train_df = pl.read_csv(DATA_RAW_DIR / "train.csv", columns=zone_cols + [TARGET_COL]).to_pandas()
    test_df = pl.read_csv(DATA_RAW_DIR / "test.csv", columns=zone_cols).to_pandas()
    y = train_df[TARGET_COL].to_numpy(dtype=np.int8)
    train_ids = train_df[ID_COL].to_numpy()
    test_ids = test_df[ID_COL].to_numpy()
    n_train, n_test = len(y), len(test_ids)
    zones_tr = get_zero_fraud_zones(train_df)
    zones_te = get_zero_fraud_zones(test_df)

    raw_members = []
    for tag in ("v3", "v4", "v2"):
        for f in sorted(OOF_DIR.glob(f"oof_{tag}_*.npy")):
            short = f.stem.replace(f"oof_{tag}_", "")
            tf = OOF_DIR / f"test_{tag}_{short}.npy"
            if tf.exists():
                oa = np.asarray(np.load(f), dtype=np.float64).squeeze()
                ta = np.asarray(np.load(tf), dtype=np.float64).squeeze()
                if valid_pair(oa, ta, n_train, n_test):
                    raw_members.append((f"own_{tag}:{short}", oa, ta))

    for prefix, slug in PUBLIC_OOF_DATASETS:
        try:
            ds_path = Path(kagglehub.dataset_download(slug))
            raw_members.extend(discover_all_in_dir(ds_path, prefix, n_train, n_test, train_ids, test_ids))
        except Exception as e:
            print(f"[WARN] {slug}: {e}")

    scored = sorted(
        [(fast_auc(y, oa), nm, oa, ta) for nm, oa, ta in raw_members if fast_auc(y, oa) >= 0.942],
        key=lambda x: -x[0],
    )

    names, aucs, pr_o, pr_t, lg_o, lg_t = [], [], [], [], [], []
    sample_idx = np.random.default_rng(SEED).choice(n_train, size=50000, replace=False)
    sample_vecs = []
    for auc, nm, oa, ta in scored:
        po = to_probit_rank(oa)
        sv = po[sample_idx]
        sv_n = (sv - sv.mean()) / (sv.std() + 1e-12)
        if any(float(np.dot(sv_n, prev) / len(sv_n)) > 0.99990 for prev in sample_vecs):
            continue
        sample_vecs.append(sv_n)
        names.append(nm)
        aucs.append(auc)
        pr_o.append(po)
        pr_t.append(to_probit_rank(ta))
        lg_o.append(to_clean_logit(oa))
        lg_t.append(to_clean_logit(ta))

    O_pr = np.column_stack(pr_o)
    T_pr = np.column_stack(pr_t)
    O_lg = np.column_stack(lg_o)
    T_lg = np.column_stack(lg_t)
    del pr_o, pr_t, lg_o, lg_t, raw_members, scored
    gc.collect()
    print(f"Prepared {len(names)} unique models in {time.time()-t0:.1f}s", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(O_pr, y))

    # Track A: Pure Individual Models ONLY
    pure_idx = [
        i for i, nm in enumerate(names)
        if not any(tok in nm.lower() for tok in ("anchor", "stack_ref", "oof_stack", "wide_lm_stack", "ensemble"))
    ]
    pure_oof, pure_test, pure_raw, pure_z = run_pruned_nested_stack(
        O_lg[:, pure_idx], T_lg[:, pure_idx], O_pr[:, pure_idx], T_pr[:, pure_idx],
        y, [names[i] for i in pure_idx], [aucs[i] for i in pure_idx],
        splits, zones_tr, "TRACK A: PURE BASE MODELS ONLY (Private Shield)",
    )

    # Track B: Full 209-Stream Grand Stack
    full_oof, full_test, full_raw, full_z = run_pruned_nested_stack(
        O_lg, T_lg, O_pr, T_pr, y, names, aucs,
        splits, zones_tr, "TRACK B: FULL 209-STREAM GRAND STACK (Max Score)",
    )

    verify_and_save(test_ids, apply_zone_push(pure_test, zones_te) if pure_z > pure_raw else to_unit_rank(pure_test), "sub_v5_2_pure_shield.csv")
    verify_and_save(test_ids, apply_zone_push(full_test, zones_te) if full_z > full_raw else to_unit_rank(full_test), "sub_v5_2_grand_champion.csv")
    verify_and_save(test_ids, to_unit_rank(full_test), "sub_v5_2_grand_champion_nozone.csv")
    print(f"\nAll done in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
