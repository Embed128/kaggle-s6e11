"""25-second Fold-1 test of Original Dataset (uditjain13/credit-card-fraud-detection-2026, 20k rows)
+ LOO Unsupervised Cluster Profile on top of FE v3!
"""

import time
from pathlib import Path

import kagglehub
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
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
    orig_dir = Path(kagglehub.dataset_download("uditjain13/credit-card-fraud-detection-2026"))
    orig_pl = pl.read_csv(orig_dir / "credit_card_fraud_2026.csv")
    train_pl = pl.read_csv(DATA_RAW_DIR / "train.csv")
    test_pl = pl.read_csv(DATA_RAW_DIR / "test.csv")

    print(f"Original dataset: shape={orig_pl.shape}, fraud_rate={orig_pl[TARGET_COL].mean()*100:.4f}%")
    print(f"Train dataset   : shape={train_pl.shape}, fraud_rate={train_pl[TARGET_COL].mean()*100:.4f}%")

    # Check exact value overlap between competition test and the 20k ORIGINAL dataset!
    print("\n=== Exact Value Overlap: Competition Test -> Original 20k Dataset ===")
    for col in ["amount_usd", "account_balance_usd", "hours_since_last_txn", "distance_from_home_km", "merchant_risk_score", "velocity_score"]:
        orig_set = set(orig_pl[col].cast(pl.Float64).round(2).unique().to_list())
        te_vals = test_pl[col].cast(pl.Float64).round(2).to_list()
        ov = sum(v in orig_set for v in te_vals) / len(te_vals) * 100
        print(f"  {col:<24}: {ov:.2f}% of test exact values exist in 20k orig!")

    # Build external Original-Dataset Target Encoding (100% leak-free external prior!)
    orig_cast = orig_pl.with_columns(
        [pl.col(c).cast(pl.Float32) for c in NUM_COLS]
        + [pl.col(c).cast(pl.Int8) for c in BOOL_COLS]
        + [pl.col(TARGET_COL).cast(pl.Float32)]
    )
    orig_prior = float(orig_cast[TARGET_COL].mean())

    # Prepare train/test base
    X_train_base, X_test_base, y, _, meta = build_base_feature_frames()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(skf.split(X_train_base, y))
    y_tr, y_va = y[tr_idx], y[va_idx]

    # Build V2 Fold-1 TE features
    X_tr_v2, X_va_v2, _ = add_oof_target_encoding(
        X_train_base.iloc[tr_idx], y_tr, X_train_base.iloc[va_idx], None,
        te_cols=meta["te_cols"], keep_base_cols=meta["model_base_cols"], smooth=15.0, seed=SEED + 1
    )

    # Compute Original-dataset exact-value TE & count & mean-risk lookups for ALL 24 cols!
    raw_cols = [c for c in test_pl.columns if c != ID_COL]
    tr_df_orig_feats = {}
    orig_pd = orig_cast.select(raw_cols + [TARGET_COL]).to_pandas()
    tr_raw_pd = X_train_base[raw_cols].copy()

    for col in raw_cols:
        stats = orig_pd.groupby(col, observed=False)[TARGET_COL].agg(["count", "sum"])
        # Smooth with m=5 and m=20 against orig_prior
        p5 = (stats["sum"] + 5.0 * orig_prior) / (stats["count"] + 5.0)
        p5 = np.clip(p5, 1e-5, 1 - 1e-5)
        log_enc = np.log(p5 / (1 - p5))
        tr_df_orig_feats[f"orig_te_{col}"] = tr_raw_pd[col].map(log_enc).fillna(np.log(orig_prior / (1 - orig_prior))).to_numpy(dtype=np.float32)
        tr_df_orig_feats[f"orig_cnt_{col}"] = np.log1p(tr_raw_pd[col].map(stats["count"]).fillna(0.0).to_numpy(dtype=np.float32))

    # Also check orig_te single-feature AUCs on Fold 1!
    orig_feat_df = pd.DataFrame(tr_df_orig_feats)
    print("\n=== Top Original-Dataset External TE Single-Feature AUCs (Fold 1) ===")
    aucs = [(c, roc_auc_score(y_va, orig_feat_df[c].iloc[va_idx])) for c in orig_feat_df.columns if c.startswith("orig_te_")]
    for c, a in sorted(aucs, key=lambda x: -x[1])[:10]:
        print(f"  {c:<38} Fold 1 AUC = {a:.5f}")

    # Evaluate V2 lr_gam vs V2 + Orig external TE features!
    def eval_lr(Xtr, Xva, label):
        num_c = [c for c in Xtr.columns if c not in CAT_COLS]
        pre = ColumnTransformer([("num", StandardScaler(), num_c), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])
        m = LogisticRegression(C=0.05, max_iter=500)
        m.fit(pre.fit_transform(Xtr), y_tr)
        p = m.predict_proba(pre.transform(Xva))[:, 1]
        print(f"  {label:<48} Fold 1 AUC = {roc_auc_score(y_va, p):.6f}")
        return p

    print("\n=== Fold 1 Comparison: V2 vs V2 + Original Dataset Features ===")
    p_v2 = eval_lr(X_tr_v2, X_va_v2, "0. V2 lr_gam (146 feats)")

    X_tr_orig = pd.concat([X_tr_v2, orig_feat_df.iloc[tr_idx].reset_index(drop=True)], axis=1)
    X_va_orig = pd.concat([X_va_v2, orig_feat_df.iloc[va_idx].reset_index(drop=True)], axis=1)
    p_orig = eval_lr(X_tr_orig, X_va_orig, f"1. V2 + Orig External TE+Cnt ({X_tr_orig.shape[1]} feats)")

    # Also test LGB_d2 on V2 + Orig!
    m_lgb = lgb.LGBMClassifier(
        objective="binary", metric="auc", max_depth=2, num_leaves=4,
        learning_rate=0.03, n_estimators=8000, colsample_bytree=0.25,
        subsample=0.8, subsample_freq=1, min_child_samples=200,
        reg_lambda=35.0, reg_alpha=2.0, max_bin=1023,
        random_state=SEED + 1, n_jobs=-1, verbose=-1,
    )
    m_lgb.fit(X_tr_orig, y_tr, eval_set=[(X_va_orig, y_va)], categorical_feature=CAT_COLS, callbacks=[lgb.early_stopping(200, verbose=False)])
    p_lgb = m_lgb.predict_proba(X_va_orig)[:, 1]
    print(f"  {'2. LGB_d2 + Orig External TE+Cnt':<48} Fold 1 AUC = {roc_auc_score(y_va, p_lgb):.6f} (iter={m_lgb.best_iteration_})")

    # Test Residual Boosting (init_score from lr_gam -> LGB_d2)!
    # Strictly inner-OOF init_score for tr_idx, full lr_gam logit for va_idx!
    inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 99)
    tr_init = np.zeros(len(y_tr), dtype=np.float64)
    num_c = [c for c in X_tr_orig.columns if c not in CAT_COLS]
    pre_r = ColumnTransformer([("num", StandardScaler(), num_c), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])
    X_tr_mat = pre_r.fit_transform(X_tr_orig)
    X_va_mat = pre_r.transform(X_va_orig)
    for itr, iva in inner_skf.split(X_tr_mat, y_tr):
        clf_in = LogisticRegression(C=0.05, max_iter=500)
        clf_in.fit(X_tr_mat[itr], y_tr[itr])
        tr_init[iva] = logit(np.clip(clf_in.predict_proba(X_tr_mat[iva])[:, 1], 1e-6, 1 - 1e-6))
    clf_full = LogisticRegression(C=0.05, max_iter=500)
    clf_full.fit(X_tr_mat, y_tr)
    va_init = logit(np.clip(clf_full.predict_proba(X_va_mat)[:, 1], 1e-6, 1 - 1e-6))

    m_res = lgb.LGBMClassifier(
        objective="binary", metric="auc", max_depth=3, num_leaves=7,
        learning_rate=0.02, n_estimators=4000, colsample_bytree=0.25,
        subsample=0.8, subsample_freq=1, min_child_samples=300,
        reg_lambda=80.0, reg_alpha=5.0, max_bin=1023,
        random_state=SEED + 1, n_jobs=-1, verbose=-1,
    )
    m_res.fit(
        X_tr_orig, y_tr,
        init_score=tr_init,
        eval_set=[(X_va_orig, y_va)],
        eval_init_score=[va_init],
        categorical_feature=CAT_COLS,
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    p_res_logit = va_init + m_res.predict(X_va_orig, raw_score=True)
    print(f"  {'3. Residual-Boost LGB_d3 on lr_gam init_score':<48} Fold 1 AUC = {roc_auc_score(y_va, p_res_logit):.6f} (iter={m_res.best_iteration_})")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
