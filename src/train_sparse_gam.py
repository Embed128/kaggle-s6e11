"""Joint Sparse Exact-Value + Digit + Spline + Interaction Logistic GAM for Kaggle S6E11.
Directly learns exact conditional log-odds weights for every discrete value, digit, exact numeric value,
and key 2-way interaction jointly via L2-regularized Logistic Regression on sparse CSR matrices!
"""

import gc
import time

import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from config import DATA_RAW_DIR, ID_COL, N_FOLDS, OOF_DIR, SEED, SUBMISSION_DIR, TARGET_COL, ensure_dirs
from features import BOOL_COLS, CAT_COLS, NUM_COLS, TOP_TE_PAIRS


def build_sparse_gam_matrices() -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, pd.Series]:
    t0 = time.time()
    train_pl = pl.read_csv(DATA_RAW_DIR / "train.csv")
    test_pl = pl.read_csv(DATA_RAW_DIR / "test.csv")
    n_train = train_pl.height
    y = train_pl[TARGET_COL].to_numpy().astype(np.int8)
    test_ids = test_pl[ID_COL].to_pandas()

    raw_cols = [c for c in test_pl.columns if c != ID_COL]
    df = pl.concat([train_pl.select(raw_cols), test_pl.select(raw_cols)], how="vertical").with_columns(
        [pl.col(c).cast(pl.Float32) for c in NUM_COLS]
        + [pl.col(c).cast(pl.Int8) for c in BOOL_COLS]
    )
    del train_pl, test_pl
    gc.collect()

    # Build multi-scale digits, rounding, exact-value strings, and flag bitmask
    bitmask = (
        pl.col("is_foreign_transaction").cast(pl.Int16) * 1
        + pl.col("is_new_merchant").cast(pl.Int16) * 2
        + pl.col("used_vpn").cast(pl.Int16) * 4
        + pl.col("ip_country_mismatch").cast(pl.Int16) * 8
        + pl.col("billing_shipping_mismatch").cast(pl.Int16) * 16
        + pl.col("is_ai_generated_scam_attempt").cast(pl.Int16) * 32
    ).alias("flag_bitmask")

    exprs = [bitmask]
    ohe_cols = list(CAT_COLS) + [
        "cvv_retry_count", "prior_disputes", "time_of_day_hour", "day_of_week",
        "txn_count_last_24h", "customer_age", "card_age_months", "flag_bitmask",
    ]

    # Exact values of continuous cols (captured as discrete tokens for joint MLE!)
    exact_cols = [
        "merchant_risk_score", "velocity_score",
        "amount_usd", "account_balance_usd",
        "hours_since_last_txn", "distance_from_home_km",
    ]
    for col in exact_cols:
        exprs.append(pl.col(col).round(2).cast(pl.String).alias(f"{col}_exact"))
        ohe_cols.append(f"{col}_exact")

    # Multi-scale digits & rounded buckets
    for col in ["amount_usd", "account_balance_usd", "hours_since_last_txn", "distance_from_home_km"]:
        scaled100 = (pl.col(col) * 100.0).round(0).cast(pl.Int64)
        scaled10 = (pl.col(col) * 10.0).round(0).cast(pl.Int64)
        int_part = pl.col(col).floor().cast(pl.Int64)
        new_names = [
            f"{col}_cents", f"{col}_dec1", f"{col}_dec2",
            f"{col}_unit1", f"{col}_tens", f"{col}_hund",
            f"{col}_int", f"{col}_r1", f"{col}_div10",
        ]
        exprs.extend([
            (scaled100 % 100).cast(pl.Int32).alias(new_names[0]),
            ((scaled100 // 10) % 10).cast(pl.Int32).alias(new_names[1]),
            (scaled100 % 10).cast(pl.Int32).alias(new_names[2]),
            (int_part % 10).cast(pl.Int32).alias(new_names[3]),
            ((int_part // 10) % 10).cast(pl.Int32).alias(new_names[4]),
            ((int_part // 100) % 10).cast(pl.Int32).alias(new_names[5]),
            int_part.cast(pl.Int32).alias(new_names[6]),
            scaled10.cast(pl.Int32).alias(new_names[7]),
            (int_part // 10).cast(pl.Int32).alias(new_names[8]),
        ])
        ohe_cols.extend(new_names)

    for col in ["amount_usd", "account_balance_usd"]:
        int_part = pl.col(col).floor().cast(pl.Int64)
        exprs.extend([
            (int_part // 100).cast(pl.Int32).alias(f"{col}_div100"),
            ((int_part // 1000) % 10).cast(pl.Int32).alias(f"{col}_thou"),
        ])
        ohe_cols.extend([f"{col}_div100", f"{col}_thou"])

    for col in ["merchant_risk_score", "velocity_score"]:
        scaled10 = (pl.col(col) * 10.0).round(0).cast(pl.Int64)
        int_part = pl.col(col).floor().cast(pl.Int64)
        new_names = [f"{col}_dec1", f"{col}_unit1", f"{col}_tens", f"{col}_int", f"{col}_div5"]
        exprs.extend([
            (scaled10 % 10).cast(pl.Int32).alias(new_names[0]),
            (int_part % 10).cast(pl.Int32).alias(new_names[1]),
            ((int_part // 10) % 10).cast(pl.Int32).alias(new_names[2]),
            int_part.cast(pl.Int32).alias(new_names[3]),
            (int_part // 5).cast(pl.Int32).alias(new_names[4]),
        ])
        ohe_cols.extend(new_names)

    df = df.with_columns(exprs)

    # Key 2-way interactions as sparse one-hot tokens
    pair_exprs = []
    for a, b in TOP_TE_PAIRS:
        k = f"p__{a}__{b}"
        ohe_cols.append(k)
        pair_exprs.append((pl.col(a).cast(pl.String) + "_" + pl.col(b).cast(pl.String)).alias(k))
    df = df.with_columns(pair_exprs)

    # Smooth spline + linear basis for numeric cols + raw booleans
    num_mat = df.select(NUM_COLS).to_numpy().astype(np.float32)
    bool_mat = df.select(BOOL_COLS).to_numpy().astype(np.float32)

    sc = StandardScaler()
    num_sc = sc.fit_transform(num_mat)
    sp = SplineTransformer(n_knots=10, degree=3, include_bias=False)
    num_sp = sp.fit_transform(num_sc)
    dense_part = sparse.csr_matrix(np.hstack([num_sc, num_sp, bool_mat]).astype(np.float32))

    # Sparse OneHot over all discrete, exact-value, digit, rounded, and pair columns (min_frequency=5 avoids singleton noise)
    ohe_df = df.select(ohe_cols).to_pandas()
    del df
    gc.collect()

    ohe = OneHotEncoder(min_frequency=5, handle_unknown="ignore", dtype=np.float32)
    sparse_part = ohe.fit_transform(ohe_df)
    del ohe_df
    gc.collect()

    X_all = sparse.hstack([dense_part, sparse_part], format="csr")
    X_tr = X_all[:n_train]
    X_te = X_all[n_train:]
    print(f"Sparse GAM matrix ready in {time.time()-t0:.1f}s | X_train={X_tr.shape}, nnz={X_tr.nnz:,}")
    return X_tr, X_te, y, test_ids


def main() -> None:
    ensure_dirs()
    X_tr, X_te, y, test_ids = build_sparse_gam_matrices()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Quick Fold-1 C sweep first (takes ~15s!)
    tr1, va1 = next(skf.split(X_tr, y))
    print("\n=== Fold-1 Regularization C Sweep ===")
    best_c = 0.05
    best_auc = 0.0
    for c in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        t0 = time.time()
        clf = LogisticRegression(C=c, solver="lbfgs", max_iter=500)
        clf.fit(X_tr[tr1], y[tr1])
        p = clf.predict_proba(X_tr[va1])[:, 1]
        auc = roc_auc_score(y[va1], p)
        print(f"  C={c:<6} Fold 1 AUC = {auc:.6f} ({time.time()-t0:.1f}s)")
        if auc > best_auc:
            best_auc = auc
            best_c = c

    print(f"\n=== Running Full 5-Fold Sparse GAM (best C={best_c}) ===")
    oof = np.zeros(X_tr.shape[0], dtype=np.float64)
    test_pred = np.zeros(X_te.shape[0], dtype=np.float64)
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y), start=1):
        t0 = time.time()
        clf = LogisticRegression(C=best_c, solver="lbfgs", max_iter=600)
        clf.fit(X_tr[tr_idx], y[tr_idx])
        va_p = clf.predict_proba(X_tr[va_idx])[:, 1]
        oof[va_idx] = va_p
        test_pred += clf.predict_proba(X_te)[:, 1] / N_FOLDS
        f_auc = roc_auc_score(y[va_idx], va_p)
        fold_aucs.append(f_auc)
        print(f"  [sparse_gam] Fold {fold}/{N_FOLDS} AUC = {f_auc:.6f} ({time.time()-t0:.1f}s)")

    overall_auc = roc_auc_score(y, oof)
    print("\n========================================================")
    print(f"  sparse_gam | Mean={np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f} | Overall OOF={overall_auc:.6f}")
    print("========================================================")

    np.save(OOF_DIR / "oof_v2_sparse_gam.npy", oof)
    np.save(OOF_DIR / "test_v2_sparse_gam.npy", test_pred)
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_pred}).to_csv(
        SUBMISSION_DIR / "sub_v2_sparse_gam.csv", index=False
    )


if __name__ == "__main__":
    main()
