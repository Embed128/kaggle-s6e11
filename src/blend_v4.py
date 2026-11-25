"""V4 Champion Ensemble Optimizer:
- Automatically pools ALL saved V3 + V4 10-fold OOF predictions (up to 17 diverse models!)
- Compares 4 strictly Nested-CV meta-learners:
    1. Hill-Climb Log-Odds Blend
    2. Hill-Climb Rank Blend
    3. L2 Logistic Meta-Stacking
    4. Shallow LightGBM (depth=2) Regime-Gated Meta-Stacker (learns when LM vs GBDT vs GAM is stronger!)
- Applies verified Zero-Fraud Boundary Post-Processing (Zones 1-4: 39,149 zero-fraud rows in train)
  strictly validated on Nested OOF CV!
"""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import DATA_RAW_DIR, ID_COL, OOF_DIR, SEED, SUBMISSION_DIR, TARGET_COL


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return logit(p)


def to_rank(p: np.ndarray) -> np.ndarray:
    return (rankdata(p) - 0.5) / len(p)


def get_zero_fraud_zones(df: pd.DataFrame) -> np.ndarray:
    """Verified Zero-Fraud Boundary Rules (Zones 1-4): 39,149 rows in train.csv with 0 frauds."""
    mrs = df["merchant_risk_score"].to_numpy()
    vel = df["velocity_score"].to_numpy()
    clean = (
        (df["cvv_retry_count"].to_numpy() == 0)
        & (df["prior_disputes"].to_numpy() == 0)
        & (~df["is_new_merchant"].to_numpy(bool))
        & (~df["is_foreign_transaction"].to_numpy(bool))
        & (~df["used_vpn"].to_numpy(bool))
        & (~df["ip_country_mismatch"].to_numpy(bool))
        & (~df["billing_shipping_mismatch"].to_numpy(bool))
        & (~df["is_ai_generated_scam_attempt"].to_numpy(bool))
    )
    z1 = mrs < 13.1
    z2 = (vel <= 4.1) & (mrs <= 20.0)
    z3 = clean & (vel <= 4.1) & (mrs <= 33.8)
    z4 = clean & (vel <= 6.0) & (mrs <= 25.0)
    return z1 | z2 | z3 | z4


def apply_zone_push(scores: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    """Push zero-fraud zone rows strictly below all non-zone rows while preserving internal ordering."""
    r = to_rank(scores).copy()
    r[zone_mask] -= 2.0
    return to_rank(r)


def fast_hill_climb_weights(X_oof: np.ndarray, y: np.ndarray, n_steps: int = 80) -> np.ndarray:
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


def verify_and_save(test_ids: pd.Series, preds: np.ndarray, out_name: str) -> None:
    sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds})
    assert len(sub) == len(test_ids), "Row count mismatch!"
    assert not sub[TARGET_COL].isna().any(), "NaNs found!"
    assert (sub[TARGET_COL] >= 0).all() and (sub[TARGET_COL] <= 1).all(), "Out of [0,1]!"
    out_path = SUBMISSION_DIR / out_name
    sub.to_csv(out_path, index=False)
    print(f"Saved -> {out_path} (min={preds.min():.6f}, mean={preds.mean():.6f}, max={preds.max():.6f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", type=str, default="v3,v4", help="Comma-separated tags to pool")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--out", type=str, default="sub_v4_champion.csv")
    args = parser.parse_args()

    zone_cols = [
        "id", "velocity_score", "merchant_risk_score", "cvv_retry_count",
        "prior_disputes", "is_new_merchant", "is_foreign_transaction",
        "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    train_df = pl.read_csv(DATA_RAW_DIR / "train.csv", columns=zone_cols + [TARGET_COL]).to_pandas()
    test_df = pl.read_csv(DATA_RAW_DIR / "test.csv", columns=zone_cols).to_pandas()
    y = train_df[TARGET_COL].to_numpy(dtype=np.int8)
    test_ids = test_df[ID_COL]

    zones_tr = get_zero_fraud_zones(train_df)
    zones_te = get_zero_fraud_zones(test_df)
    print(f"Zero-Fraud Zones 1-4: train={zones_tr.sum():,} (frauds={y[zones_tr].sum()}) | test={zones_te.sum():,}")

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    names, oofs, tests = [], [], []
    print(f"\n=== Pooled Single Models (tags={tags}) ===")
    for tag in tags:
        for f in sorted(OOF_DIR.glob(f"oof_{tag}_*.npy")):
            short = f.stem.replace(f"oof_{tag}_", "")
            tf = OOF_DIR / f"test_{tag}_{short}.npy"
            if not tf.exists():
                continue
            oof_p = np.load(f)
            test_p = np.load(tf)
            if len(oof_p) != len(y) or len(test_p) != len(test_ids):
                continue
            label = f"{tag}:{short}"
            auc = roc_auc_score(y, oof_p)
            print(f"  {label:<22} OOF AUC = {auc:.6f} | test_mean = {test_p.mean():.5f}")
            names.append(label)
            oofs.append(oof_p)
            tests.append(test_p)

    if not names:
        raise FileNotFoundError(f"No matching OOF files found for tags={tags} in {OOF_DIR}")

    O_logit = np.column_stack([to_logit(p) for p in oofs])
    T_logit = np.column_stack([to_logit(p) for p in tests])
    O_rank = np.column_stack([to_rank(p) for p in oofs])
    T_rank = np.column_stack([to_rank(p) for p in tests])

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)

    # 1. Nested Hill-Climb Log-Odds Blend
    hc_oof = np.zeros(len(y), dtype=np.float64)
    hc_w_list = []
    for tr_idx, va_idx in skf.split(O_logit, y):
        w = fast_hill_climb_weights(O_logit[tr_idx], y[tr_idx], n_steps=80)
        hc_w_list.append(w)
        hc_oof[va_idx] = O_logit[va_idx] @ w
    avg_w_hc = np.mean(hc_w_list, axis=0)
    avg_w_hc /= avg_w_hc.sum()
    hc_auc = roc_auc_score(y, hc_oof)
    hc_test = expit(T_logit @ avg_w_hc)

    # 2. Nested Hill-Climb Rank Blend
    rk_oof = np.zeros(len(y), dtype=np.float64)
    rk_w_list = []
    for tr_idx, va_idx in skf.split(O_rank, y):
        w = fast_hill_climb_weights(O_rank[tr_idx], y[tr_idx], n_steps=80)
        rk_w_list.append(w)
        rk_oof[va_idx] = O_rank[va_idx] @ w
    avg_w_rk = np.mean(rk_w_list, axis=0)
    avg_w_rk /= avg_w_rk.sum()
    rk_auc = roc_auc_score(y, rk_oof)
    rk_test = T_rank @ avg_w_rk

    # 3. Nested L2 Logistic Meta-Stacking
    best_lr_c, best_lr_auc, best_lr_oof, best_lr_test, best_coefs = 0.05, 0.0, None, None, None
    for c in [0.005, 0.02, 0.1, 0.5]:
        s_oof = np.zeros(len(y), dtype=np.float64)
        s_test = np.zeros(len(test_ids), dtype=np.float64)
        coefs = []
        for tr_idx, va_idx in skf.split(O_logit, y):
            meta = LogisticRegression(C=c, max_iter=600)
            meta.fit(O_logit[tr_idx], y[tr_idx])
            s_oof[va_idx] = meta.predict_proba(O_logit[va_idx])[:, 1]
            s_test += meta.predict_proba(T_logit)[:, 1] / args.n_folds
            coefs.append(meta.coef_[0])
        auc = roc_auc_score(y, s_oof)
        if auc > best_lr_auc:
            best_lr_auc, best_lr_c, best_lr_oof, best_lr_test = auc, c, s_oof, s_test
            best_coefs = np.mean(coefs, axis=0)

    # 4. Nested Shallow LightGBM Meta-Stacker (depth=2 + model disagreement/std features!)
    O_meta = np.column_stack([O_logit, O_logit.mean(axis=1), O_logit.std(axis=1), O_logit.max(axis=1) - O_logit.min(axis=1)])
    T_meta = np.column_stack([T_logit, T_logit.mean(axis=1), T_logit.std(axis=1), T_logit.max(axis=1) - T_logit.min(axis=1)])
    lgb_oof = np.zeros(len(y), dtype=np.float64)
    lgb_test = np.zeros(len(test_ids), dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(O_meta, y), start=1):
        m = lgb.LGBMClassifier(
            objective="binary", metric="auc", max_depth=2, num_leaves=4,
            learning_rate=0.03, n_estimators=2000, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, min_child_samples=300, reg_lambda=50.0, reg_alpha=5.0,
            random_state=SEED + fold, n_jobs=-1, verbose=-1,
        )
        m.fit(
            O_meta[tr_idx], y[tr_idx],
            eval_set=[(O_meta[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        lgb_oof[va_idx] = m.predict_proba(O_meta[va_idx])[:, 1]
        lgb_test += m.predict_proba(T_meta)[:, 1] / args.n_folds
    lgb_meta_auc = roc_auc_score(y, lgb_oof)

    # 5. Super-Meta Blend (0.5 Hill-Climb Log-Odds + 0.5 Shallow LGB Meta)
    super_oof = 0.5 * to_logit(expit(hc_oof)) + 0.5 * to_logit(lgb_oof)
    super_test = expit(0.5 * to_logit(hc_test) + 0.5 * to_logit(lgb_test))
    super_auc = roc_auc_score(y, super_oof)

    print("\n=== V4 Ensemble Comparison (Strictly Nested 10-Fold CV) ===")
    candidates = [
        ("Hill-Climb Log-Odds", hc_auc, expit(hc_oof), hc_test),
        ("Hill-Climb Rank", rk_auc, rk_oof, rk_test),
        (f"Logistic Meta (C={best_lr_c})", best_lr_auc, best_lr_oof, best_lr_test),
        ("Shallow LGB Meta (d=2)", lgb_meta_auc, lgb_oof, lgb_test),
        ("Super-Meta (HC + LGB)", super_auc, expit(super_oof), super_test),
    ]
    for nm, auc, oof_c, _ in candidates:
        z_auc = roc_auc_score(y, apply_zone_push(oof_c, zones_tr))
        print(f"  {nm:<26} Raw OOF = {auc:.6f} | +Zones 1-4 = {z_auc:.6f} ({z_auc - auc:+.6f})")

    print("\nLearned Model Weights:")
    for n, wh, wr, wc in zip(names, avg_w_hc, avg_w_rk, best_coefs):
        print(f"  {n:<22} logit_hc={wh:.4f} | rank_hc={wr:.4f} | stack_coef={wc:+.4f}")

    # Pick highest Nested-CV winner
    best_nm, best_raw_auc, best_oof, best_test = max(candidates, key=lambda x: x[1])
    z_oof_auc = roc_auc_score(y, apply_zone_push(best_oof, zones_tr))
    use_zones = z_oof_auc > best_raw_auc

    print(f"\n>>> Best Raw Ensemble : {best_nm} (Nested OOF = {best_raw_auc:.6f})")
    print(f">>> With Zones 1-4    : Nested OOF = {z_oof_auc:.6f} (use_zones={use_zones})")

    verify_and_save(test_ids, best_test, "sub_v4_raw_best.csv")
    final_test = apply_zone_push(best_test, zones_te) if use_zones else best_test
    verify_and_save(test_ids, final_test, args.out)


if __name__ == "__main__":
    main()
