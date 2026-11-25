"""Breakthrough Fold-1 Probe:
1. Unsupervised Exact-Value Cluster Profile (aggregate key risk features over exact continuous values across ALL 1M rows!)
2. Strictly leak-free Leave-One-Out (LOO) Base-Logit Propagation over exact values (reconstructs latent original-record risk!)
"""

import time
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_RAW_DIR, ID_COL, N_FOLDS, SEED, TARGET_COL
from features import BOOL_COLS, CAT_COLS, NUM_COLS, add_oof_target_encoding, build_base_feature_frames


def main() -> None:
    t0 = time.time()
    train_pl = pl.read_csv(DATA_RAW_DIR / "train.csv")
    test_pl = pl.read_csv(DATA_RAW_DIR / "test.csv")
    n_train = train_pl.height
    y = train_pl[TARGET_COL].to_numpy().astype(np.int8)
    raw_cols = [c for c in test_pl.columns if c != ID_COL]

    df_all = pl.concat([train_pl.select(raw_cols), test_pl.select(raw_cols)], how="vertical").with_columns(
        [pl.col(c).cast(pl.Float32) for c in NUM_COLS]
        + [pl.col(c).cast(pl.Int8) for c in BOOL_COLS]
        + [(pl.col("auth_method") == "No Authentication").cast(pl.Float32).alias("is_no_auth")]
    )

    # First train a simple raw-only Stage-0 risk predictor strictly inside Fold 1 (NO exact-value TE, pure raw!)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(skf.split(np.zeros(n_train), y))
    y_tr, y_va = y[tr_idx], y[va_idx]

    # Build V2 FE base for Fold 1 comparison
    X_train_base, X_test_base, _, _, meta = build_base_feature_frames()
    X_tr_v2, X_va_v2, _ = add_oof_target_encoding(
        X_train_base.iloc[tr_idx], y_tr, X_train_base.iloc[va_idx], None,
        te_cols=meta["te_cols"], keep_base_cols=meta["model_base_cols"], smooth=15.0, seed=SEED + 1
    )

    # Reference V2 lr_gam Fold 1
    num_v2 = [c for c in X_tr_v2.columns if c not in CAT_COLS]
    pre_v2 = ColumnTransformer([("num", StandardScaler(), num_v2), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])
    lr_v2 = LogisticRegression(C=0.05, max_iter=500)
    lr_v2.fit(pre_v2.fit_transform(X_tr_v2), y_tr)
    p_v2 = lr_v2.predict_proba(pre_v2.transform(X_va_v2))[:, 1]
    print(f"0. Reference V2 lr_gam (146 feats)        Fold 1 AUC = {roc_auc_score(y_va, p_v2):.6f}")

    # Stage-0 pure-raw linear risk score (fit strictly on tr_idx, scored on ALL 1M rows -> ZERO target leakage!)
    pre_s0 = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS + BOOL_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    df_all_pd = df_all.select(raw_cols).to_pandas()
    s0_clf = LogisticRegression(C=1.0, max_iter=500)
    s0_clf.fit(pre_s0.fit_transform(df_all_pd.iloc[tr_idx]), y_tr)
    all_s0_logit = logit(np.clip(s0_clf.predict_proba(pre_s0.transform(df_all_pd))[:, 1], 1e-6, 1 - 1e-6)).astype(np.float32)
    print(f"   Stage-0 Raw LR                         Fold 1 AUC = {roc_auc_score(y_va, all_s0_logit[va_idx]):.6f}")

    # Attach Stage-0 logit to ALL 1M rows and compute Leave-One-Out (LOO) mean over EVERY exact-value column!
    # LOO_mean(v) = (sum(s0_logit over v) - self_s0_logit) / (count(v) - 1)  (smoothed with global prior!)
    df_p = df_all.with_columns(pl.Series("s0_logit", all_s0_logit))
    global_s0 = float(all_s0_logit.mean())

    exact_keys = [
        "account_balance_usd",
        "amount_usd",
        "hours_since_last_txn",
        "distance_from_home_km",
        "merchant_risk_score",
        "velocity_score",
        "customer_age",
        "card_age_months",
    ]

    # Also add rounded keys & cents keys!
    df_p = df_p.with_columns([
        ((pl.col("hours_since_last_txn") * 100.0).round(0).cast(pl.Int64) % 100).alias("h_cents"),
        ((pl.col("amount_usd") * 100.0).round(0).cast(pl.Int64) % 100).alias("a_cents"),
        ((pl.col("distance_from_home_km") * 100.0).round(0).cast(pl.Int64) % 100).alias("d_cents"),
        ((pl.col("account_balance_usd") * 100.0).round(0).cast(pl.Int64) % 100).alias("b_cents"),
        pl.col("amount_usd").floor().cast(pl.Int32).alias("amount_int"),
        pl.col("account_balance_usd").floor().cast(pl.Int32).alias("balance_int"),
    ])
    all_keys = exact_keys + ["h_cents", "a_cents", "d_cents", "b_cents", "amount_int", "balance_int"]

    # Build LOO smoothed cluster profile features (ZERO target leakage!)
    profile_cols = ["s0_logit", "cvv_retry_count", "merchant_risk_score", "velocity_score", "is_new_merchant", "ip_country_mismatch", "is_no_auth"]
    new_exprs = []
    new_names = []
    for key in all_keys:
        cnt = pl.len().over(key).cast(pl.Float32)
        for pcol in profile_cols:
            if pcol == key:
                continue
            g_mean = float(df_p[pcol].mean())
            # Exact Leave-One-Out smoothed mean across all 1M rows!
            loo_sum = pl.col(pcol).sum().over(key).cast(pl.Float32) - pl.col(pcol).cast(pl.Float32)
            loo_cnt = (cnt - 1.0).clip(lower_bound=0.0)
            # Smooth with m=5 for high-SNR continuous/multi-feature cluster mean!
            loo_val = ((loo_sum + 5.0 * g_mean) / (loo_cnt + 5.0)).cast(pl.Float32)
            cname = f"loo_{pcol}_by_{key}"
            new_exprs.append(loo_val.alias(cname))
            new_names.append(cname)

    df_feat = df_p.select(new_exprs).slice(0, n_train).to_pandas()
    print(f"Built {len(new_names)} LOO unsupervised cluster features across 1M rows in {time.time()-t0:.1f}s")

    # Inspect single-feature AUC of LOO s0_logit features on Fold 1!
    print("\n=== Single-Feature AUC of LOO s0_logit Propagation (Fold 1) ===")
    for key in all_keys:
        c = f"loo_s0_logit_by_{key}"
        print(f"  {c:<42} Fold 1 AUC = {roc_auc_score(y_va, df_feat[c].iloc[va_idx]):.5f}")

    # Test adding these LOO cluster features to V2 lr_gam AND LightGBM!
    X_tr_new = pd.concat([X_tr_v2, df_feat.iloc[tr_idx].reset_index(drop=True)], axis=1)
    X_va_new = pd.concat([X_va_v2, df_feat.iloc[va_idx].reset_index(drop=True)], axis=1)

    num_new = [c for c in X_tr_new.columns if c not in CAT_COLS]
    pre_new = ColumnTransformer([("num", StandardScaler(), num_new), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])
    X_tr_s = pre_new.fit_transform(X_tr_new)
    X_va_s = pre_new.transform(X_va_new)

    print("\n=== Fold 1 Results WITH LOO Cluster Profile Features ===")
    for c in [0.01, 0.05, 0.2]:
        m = LogisticRegression(C=c, max_iter=600)
        m.fit(X_tr_s, y_tr)
        p = m.predict_proba(X_va_s)[:, 1]
        print(f"  1. lr_gam + LOO Cluster (C={c:<4})        Fold 1 AUC = {roc_auc_score(y_va, p):.6f}")

    # Test ONLY adding loo_s0_logit_* (+14 super clean cols) vs ALL (+96 cols)
    s0_only = [f"loo_s0_logit_by_{k}" for k in all_keys]
    X_tr_14 = pd.concat([X_tr_v2, df_feat[s0_only].iloc[tr_idx].reset_index(drop=True)], axis=1)
    X_va_14 = pd.concat([X_va_v2, df_feat[s0_only].iloc[va_idx].reset_index(drop=True)], axis=1)
    num_14 = [c for c in X_tr_14.columns if c not in CAT_COLS]
    pre_14 = ColumnTransformer([("num", StandardScaler(), num_14), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])
    m14 = LogisticRegression(C=0.05, max_iter=600)
    m14.fit(pre_14.fit_transform(X_tr_14), y_tr)
    p14 = m14.predict_proba(pre_14.transform(X_va_14))[:, 1]
    print(f"  2. lr_gam + ONLY 14 loo_s0_logit cols   Fold 1 AUC = {roc_auc_score(y_va, p14):.6f}")

    # Test LGB depth=2 on X_tr_new
    m_lgb = lgb.LGBMClassifier(
        objective="binary", metric="auc", max_depth=2, num_leaves=4,
        learning_rate=0.03, n_estimators=8000, colsample_bytree=0.25,
        subsample=0.8, subsample_freq=1, min_child_samples=200,
        reg_lambda=35.0, reg_alpha=2.0, max_bin=1023,
        random_state=SEED + 1, n_jobs=-1, verbose=-1,
    )
    m_lgb.fit(X_tr_new, y_tr, eval_set=[(X_va_new, y_va)], categorical_feature=CAT_COLS, callbacks=[lgb.early_stopping(200, verbose=False)])
    p_lgb = m_lgb.predict_proba(X_va_new)[:, 1]
    print(f"  3. LGB_d2 + LOO Cluster ({X_tr_new.shape[1]} feats)    Fold 1 AUC = {roc_auc_score(y_va, p_lgb):.6f} (iter={m_lgb.best_iteration_})")

    # Blend new lr + new lgb
    b = 0.65 * logit(np.clip(p14, 1e-6, 1 - 1e-6)) + 0.35 * logit(np.clip(p_lgb, 1e-6, 1 - 1e-6))
    print(f"  >>> Blend (lr_14 + lgb_new)             Fold 1 AUC = {roc_auc_score(y_va, b):.6f}")


if __name__ == "__main__":
    main()
