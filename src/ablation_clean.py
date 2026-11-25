"""Clean single-group ablation on top of Raw-24 (Fold 1) + Raw CatBoost & tuning probes."""

import time
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import N_FOLDS, SEED
from features import (
    BOOL_COLS,
    CAT_COLS,
    NUM_COLS,
    add_oof_target_encoding,
    load_base_polars,
    to_aligned_pandas,
)

BASE_LGB = dict(
    objective="binary",
    metric="auc",
    max_depth=2,
    num_leaves=4,
    learning_rate=0.05,
    n_estimators=8000,
    colsample_bytree=0.3,
    subsample=0.8,
    subsample_freq=1,
    min_child_samples=200,
    reg_lambda=30.0,
    max_bin=1023,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


def eval_lgb(X_tr, y_tr, X_va, y_va, cats, label, overrides=None):
    t0 = time.time()
    p_cfg = {**BASE_LGB, **(overrides or {})}
    m = lgb.LGBMClassifier(**p_cfg)
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        categorical_feature=cats,
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    p = m.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, p)
    print(f"  {label:<48} AUC={auc:.6f} (iter={m.best_iteration_}, {time.time()-t0:.1f}s)")
    return p, auc


def main() -> None:
    train_pl, test_pl, y, _ = load_base_polars()
    n_train = train_pl.height
    feat_cols = [c for c in test_pl.columns if c != "id"]
    df_all = pl.concat([train_pl.select(feat_cols), test_pl.select(feat_cols)], how="vertical").with_columns(
        [pl.col(c).cast(pl.Float32) for c in NUM_COLS] + [pl.col(c).cast(pl.Int8) for c in BOOL_COLS]
    )

    X_raw, _ = to_aligned_pandas(df_all.slice(0, n_train), df_all.slice(n_train, test_pl.height), CAT_COLS)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(skf.split(X_raw, y))
    y_tr, y_va = y[tr_idx], y[va_idx]

    print("=== 1. LGB Hyperparameter Probes on Raw-24 ===")
    p_base, _ = eval_lgb(X_raw.iloc[tr_idx], y_tr, X_raw.iloc[va_idx], y_va, CAT_COLS, "0. Raw-24 (d=2, bin=1023, col=0.3)")
    p_bin4k, _ = eval_lgb(X_raw.iloc[tr_idx], y_tr, X_raw.iloc[va_idx], y_va, CAT_COLS, "1. Raw-24 (d=2, bin=4095, col=0.3)", dict(max_bin=4095))
    p_col02, _ = eval_lgb(X_raw.iloc[tr_idx], y_tr, X_raw.iloc[va_idx], y_va, CAT_COLS, "2. Raw-24 (d=2, bin=2047, col=0.2, l2=50)", dict(max_bin=2047, colsample_bytree=0.2, reg_lambda=50.0, learning_rate=0.03))
    p_l1, _ = eval_lgb(X_raw.iloc[tr_idx], y_tr, X_raw.iloc[va_idx], y_va, CAT_COLS, "3. Raw-24 (d=3, leaves=5, bin=2047, col=0.2)", dict(max_depth=3, num_leaves=5, max_bin=2047, colsample_bytree=0.2, reg_lambda=50.0, learning_rate=0.03))

    print("\n=== 2. Clean Single-Group Feature Ablations (added to Raw-24) ===")
    # Group A: ONLY risk flag sum & bitmask (2 cols)
    flag_sum = sum(pl.col(c).cast(pl.Int8) for c in BOOL_COLS)
    bitmask = (
        pl.col("is_foreign_transaction").cast(pl.Int16) * 1
        + pl.col("is_new_merchant").cast(pl.Int16) * 2
        + pl.col("used_vpn").cast(pl.Int16) * 4
        + pl.col("ip_country_mismatch").cast(pl.Int16) * 8
        + pl.col("billing_shipping_mismatch").cast(pl.Int16) * 16
        + pl.col("is_ai_generated_scam_attempt").cast(pl.Int16) * 32
    )
    df_flags = df_all.with_columns([flag_sum.alias("flag_sum"), bitmask.alias("flag_bitmask")])
    Xf, _ = to_aligned_pandas(df_flags.slice(0, n_train), df_flags.slice(n_train, test_pl.height), CAT_COLS)
    eval_lgb(Xf.iloc[tr_idx], y_tr, Xf.iloc[va_idx], y_va, CAT_COLS, "A. Raw + flag_sum & bitmask (+2)")

    # Group B: ONLY clean financial/behavioral ratios (5 cols)
    df_rat = df_all.with_columns([
        (pl.col("amount_usd") / (pl.col("account_balance_usd") + 1.0)).cast(pl.Float32).alias("amt_to_bal"),
        (pl.col("distance_from_home_km") / (pl.col("hours_since_last_txn") + 0.05)).cast(pl.Float32).alias("speed_kmh"),
        (pl.col("merchant_risk_score") * pl.col("velocity_score")).cast(pl.Float32).alias("risk_x_vel"),
        (pl.col("txn_count_last_24h") / (pl.col("hours_since_last_txn") + 0.1)).cast(pl.Float32).alias("txn_rate"),
        (pl.col("amount_usd") * pl.col("txn_count_last_24h")).cast(pl.Float32).alias("spend_24h"),
    ])
    Xr, _ = to_aligned_pandas(df_rat.slice(0, n_train), df_rat.slice(n_train, test_pl.height), CAT_COLS)
    eval_lgb(Xr.iloc[tr_idx], y_tr, Xr.iloc[va_idx], y_va, CAT_COLS, "B. Raw + 5 domain ratios (+5)")

    # Group C: ONLY Frequency Encoding of numerics (13 cols)
    df_freq = df_all.with_columns([pl.len().over(c).log1p().cast(pl.Float32).alias(f"{c}_freq") for c in NUM_COLS])
    Xfr, _ = to_aligned_pandas(df_freq.slice(0, n_train), df_freq.slice(n_train, test_pl.height), CAT_COLS)
    eval_lgb(Xfr.iloc[tr_idx], y_tr, Xfr.iloc[va_idx], y_va, CAT_COLS, "C. Raw + numeric freq encoding (+13)")

    # Group D: ONLY Inner-OOF TE on high-signal single cols (no pairs)
    te_singles = ["merchant_risk_score", "velocity_score", "customer_age", "card_age_months", "time_of_day_hour", "amount_usd", "hours_since_last_txn", "distance_from_home_km"]
    Xtr_te, Xva_te, _ = add_oof_target_encoding(X_raw.iloc[tr_idx], y_tr, X_raw.iloc[va_idx], None, te_cols=te_singles, smooth=30.0)
    eval_lgb(Xtr_te, y_tr, Xva_te, y_va, CAT_COLS, "D. Raw + single-col OOF TE (+8)")

    # Group E: ONLY rounded/digit features of continuous cols (synthetic-data artifact check!)
    df_dig = df_all.with_columns([
        ((pl.col("amount_usd") * 100.0).round(0) % 100).cast(pl.Float32).alias("amt_cents"),
        (pl.col("amount_usd").floor() % 10).cast(pl.Float32).alias("amt_last_digit"),
        ((pl.col("account_balance_usd") * 100.0).round(0) % 100).cast(pl.Float32).alias("bal_cents"),
        ((pl.col("merchant_risk_score") * 10.0).round(0) % 10).cast(pl.Float32).alias("mrs_dec"),
        ((pl.col("velocity_score") * 10.0).round(0) % 10).cast(pl.Float32).alias("vel_dec"),
    ])
    Xdg, _ = to_aligned_pandas(df_dig.slice(0, n_train), df_dig.slice(n_train, test_pl.height), CAT_COLS)
    eval_lgb(Xdg.iloc[tr_idx], y_tr, Xdg.iloc[va_idx], y_va, CAT_COLS, "E. Raw + digit/decimal features (+5)")

    print("\n=== 3. Raw-24 Multi-Model Diversity (LGB + XGB + CatBoost) ===")
    # XGBoost Raw-24 tuned
    t0 = time.time()
    m_xgb = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="hist", enable_categorical=True,
        max_depth=2, learning_rate=0.03, n_estimators=8000, subsample=0.8, colsample_bytree=0.25,
        min_child_weight=2.0, reg_lambda=40.0, reg_alpha=2.0, max_bin=2048,
        random_state=SEED, n_jobs=-1, early_stopping_rounds=150,
    )
    m_xgb.fit(X_raw.iloc[tr_idx], y_tr, eval_set=[(X_raw.iloc[va_idx], y_va)], verbose=False)
    p_xgb = m_xgb.predict_proba(X_raw.iloc[va_idx])[:, 1]
    print(f"  {'XGB Raw-24 (d=2, bin=2048, col=0.25)':<48} AUC={roc_auc_score(y_va, p_xgb):.6f} (iter={m_xgb.best_iteration}, {time.time()-t0:.1f}s)")

    # CatBoost Raw-24 fast (one_hot_max_size=20 so 5 low-card cats are instant!)
    Xtr_cb = X_raw.iloc[tr_idx].copy()
    Xva_cb = X_raw.iloc[va_idx].copy()
    for c in CAT_COLS:
        Xtr_cb[c] = Xtr_cb[c].astype(str)
        Xva_cb[c] = Xva_cb[c].astype(str)
    t0 = time.time()
    m_cb = CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC", iterations=6000, learning_rate=0.04,
        depth=3, l2_leaf_reg=30.0, one_hot_max_size=20, border_count=254,
        random_seed=SEED, thread_count=-1, verbose=0,
    )
    m_cb.fit(Xtr_cb, y_tr, cat_features=CAT_COLS, eval_set=(Xva_cb, y_va), early_stopping_rounds=200)
    p_cb = m_cb.predict_proba(Xva_cb)[:, 1]
    print(f"  {'CatBoost Raw-24 (d=3, OH=20)':<48} AUC={roc_auc_score(y_va, p_cb):.6f} (iter={m_cb.get_best_iteration()}, {time.time()-t0:.1f}s)")

    def lg(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return logit(p)

    b_all = 0.40 * lg(p_col02) + 0.35 * lg(p_xgb) + 0.25 * lg(p_cb)
    print(f"\n  >>> Fold 1 Raw-24 Blend (LGB+XGB+CB)             AUC={roc_auc_score(y_va, b_all):.6f}")


if __name__ == "__main__":
    main()
