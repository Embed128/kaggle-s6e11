"""FE v3 Fold-1 probe: multi-scale digits/rounding, pair numeric-only OOF TE, freq, smoothing, and 4-model diversity."""

import time
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import N_FOLDS, SEED
from features import BOOL_COLS, CAT_COLS, NUM_COLS, add_oof_target_encoding, load_base_polars, to_aligned_pandas


def build_fe_v3_base(df_all: pl.DataFrame) -> tuple[pl.DataFrame, list[str], list[str], list[str], list[str]]:
    """Build digit, multi-scale rounded, and pair-key columns (pair keys are ONLY for OOF TE, dropped before tree fit)."""
    exprs = []
    digit_cols = []
    round_cols = []

    # 2-decimal continuous cols
    for col in ["amount_usd", "account_balance_usd", "hours_since_last_txn", "distance_from_home_km"]:
        scaled100 = (pl.col(col) * 100.0).round(0).cast(pl.Int64)
        scaled10 = (pl.col(col) * 10.0).round(0).cast(pl.Int64)
        int_part = pl.col(col).floor().cast(pl.Int64)

        exprs.extend([
            (scaled100 % 100).cast(pl.Float32).alias(f"{col}_cents"),
            ((scaled100 // 10) % 10).cast(pl.Float32).alias(f"{col}_dec1"),
            (scaled100 % 10).cast(pl.Float32).alias(f"{col}_dec2"),
            (int_part % 10).cast(pl.Float32).alias(f"{col}_unit1"),
            ((int_part // 10) % 10).cast(pl.Float32).alias(f"{col}_tens"),
            ((int_part // 100) % 10).cast(pl.Float32).alias(f"{col}_hund"),
            int_part.cast(pl.Int32).alias(f"{col}_int"),
            scaled10.cast(pl.Int32).alias(f"{col}_r1"),
            (int_part // 10).cast(pl.Int32).alias(f"{col}_div10"),
        ])
        digit_cols.extend([f"{col}_cents", f"{col}_dec1", f"{col}_dec2", f"{col}_unit1", f"{col}_tens", f"{col}_hund"])
        round_cols.extend([f"{col}_int", f"{col}_r1", f"{col}_div10"])

    # Extra coarse buckets for wide-range balance & amount
    for col in ["amount_usd", "account_balance_usd"]:
        int_part = pl.col(col).floor().cast(pl.Int64)
        exprs.extend([
            (int_part // 100).cast(pl.Int32).alias(f"{col}_div100"),
            ((int_part // 1000) % 10).cast(pl.Float32).alias(f"{col}_thou"),
        ])
        round_cols.append(f"{col}_div100")
        digit_cols.append(f"{col}_thou")

    # 1-decimal continuous cols
    for col in ["merchant_risk_score", "velocity_score"]:
        scaled10 = (pl.col(col) * 10.0).round(0).cast(pl.Int64)
        int_part = pl.col(col).floor().cast(pl.Int64)
        exprs.extend([
            (scaled10 % 10).cast(pl.Float32).alias(f"{col}_dec1"),
            (int_part % 10).cast(pl.Float32).alias(f"{col}_unit1"),
            ((int_part // 10) % 10).cast(pl.Float32).alias(f"{col}_tens"),
            int_part.cast(pl.Int32).alias(f"{col}_int"),
            (int_part // 5).cast(pl.Int32).alias(f"{col}_div5"),
        ])
        digit_cols.extend([f"{col}_dec1", f"{col}_unit1", f"{col}_tens"])
        round_cols.extend([f"{col}_int", f"{col}_div5"])

    # Integer cols digits
    for col in ["customer_age", "card_age_months"]:
        int_part = pl.col(col).cast(pl.Int64)
        exprs.extend([
            (int_part % 10).cast(pl.Float32).alias(f"{col}_unit1"),
            (int_part // 10).cast(pl.Float32).alias(f"{col}_tens"),
        ])
        digit_cols.extend([f"{col}_unit1", f"{col}_tens"])

    df_ext = df_all.with_columns(exprs)

    # Frequency encoding for exact numeric & integer-rounded values (log-count)
    freq_targets = [
        "merchant_risk_score", "velocity_score", "amount_usd", "account_balance_usd",
        "hours_since_last_txn", "distance_from_home_km", "customer_age", "card_age_months",
        "amount_usd_int", "account_balance_usd_int", "hours_since_last_txn_int", "distance_from_home_km_int",
    ]
    freq_cols = [f"{c}_freq" for c in freq_targets]
    df_ext = df_ext.with_columns([pl.len().over(c).log1p().cast(pl.Float32).alias(f"{c}_freq") for c in freq_targets])

    # Top-signal 2-way pair keys (STRICTLY for numeric OOF TE — NOT passed as raw strings!)
    top_pairs = [
        ("merchant_risk_score_div5", "velocity_score_div5"),
        ("merchant_risk_score_div5", "cvv_retry_count"),
        ("merchant_risk_score_div5", "auth_method"),
        ("velocity_score_div5", "cvv_retry_count"),
        ("cvv_retry_count", "auth_method"),
        ("cvv_retry_count", "merchant_category"),
        ("cvv_retry_count", "is_new_merchant"),
        ("auth_method", "merchant_category"),
        ("auth_method", "is_new_merchant"),
        ("ip_country_mismatch", "is_foreign_transaction"),
        ("ip_country_mismatch", "used_vpn"),
        ("billing_shipping_mismatch", "is_new_merchant"),
        ("customer_age", "merchant_category"),
        ("merchant_category", "is_new_merchant"),
    ]
    pair_keys = []
    pair_exprs = []
    for a, b in top_pairs:
        k = f"p__{a}__{b}"
        pair_keys.append(k)
        pair_exprs.append((pl.col(a).cast(pl.String) + "_" + pl.col(b).cast(pl.String)).alias(k))
    df_ext = df_ext.with_columns(pair_exprs)

    return df_ext, digit_cols, round_cols, freq_cols, pair_keys


def eval_lgb(X_tr, y_tr, X_va, y_va, cats, label, overrides=None):
    t0 = time.time()
    cfg = dict(
        objective="binary", metric="auc", max_depth=2, num_leaves=4,
        learning_rate=0.05, n_estimators=8000, colsample_bytree=0.25,
        subsample=0.8, subsample_freq=1, min_child_samples=200,
        reg_lambda=30.0, max_bin=1023, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    if overrides:
        cfg.update(overrides)
    m = lgb.LGBMClassifier(**cfg)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], categorical_feature=cats, callbacks=[lgb.early_stopping(150, verbose=False)])
    p = m.predict_proba(X_va)[:, 1]
    print(f"  {label:<54} AUC={roc_auc_score(y_va, p):.6f} (iter={m.best_iteration_}, {time.time()-t0:.1f}s)")
    return p


def main() -> None:
    train_pl, test_pl, y, _ = load_base_polars()
    n_train = train_pl.height
    raw_cols = [c for c in test_pl.columns if c != "id"]
    df_all = pl.concat([train_pl.select(raw_cols), test_pl.select(raw_cols)], how="vertical").with_columns(
        [pl.col(c).cast(pl.Float32) for c in NUM_COLS] + [pl.col(c).cast(pl.Int8) for c in BOOL_COLS]
    )

    df_ext, digit_cols, round_cols, freq_cols, pair_keys = build_fe_v3_base(df_all)
    X_all, _ = to_aligned_pandas(df_ext.slice(0, n_train), df_ext.slice(n_train, test_pl.height), CAT_COLS)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(skf.split(X_all, y))
    y_tr, y_va = y[tr_idx], y[va_idx]

    # Build strict inner-OOF TE (smooth=15 for finer exact-value resolution!)
    t0 = time.time()
    te_single = NUM_COLS + CAT_COLS + round_cols + digit_cols
    te_full = te_single + pair_keys
    Xtr_all, Xva_all, _ = add_oof_target_encoding(
        X_all.iloc[tr_idx], y_tr, X_all.iloc[va_idx], None, te_cols=te_full, smooth=15.0
    )
    print(f"Built {len(te_full)} inner-OOF TE cols in {time.time()-t0:.1f}s")

    te_single_names = [f"te_{c}" for c in te_single]
    te_pair_names = [f"te_{c}" for c in pair_keys]

    print("\n=== 1. FE v3 Incremental Ablation (Fold 1) ===")
    cols_1 = raw_cols + digit_cols + te_single_names
    p1 = eval_lgb(Xtr_all[cols_1], y_tr, Xva_all[cols_1], y_va, CAT_COLS, f"1. Raw+Digits+MultiRound TE (s=15, {len(cols_1)})")

    cols_2 = cols_1 + freq_cols
    p2 = eval_lgb(Xtr_all[cols_2], y_tr, Xva_all[cols_2], y_va, CAT_COLS, f"2. #1 + Freq encoding ({len(cols_2)})")

    cols_3 = cols_2 + te_pair_names
    p3 = eval_lgb(Xtr_all[cols_3], y_tr, Xva_all[cols_3], y_va, CAT_COLS, f"3. #2 + Top-Pair numeric OOF TE ({len(cols_3)})")

    # Pick best feature set among 1, 2, 3
    best_cols, best_p_lgb, best_name = max(
        [(cols_1, p1, "#1"), (cols_2, p2, "#2"), (cols_3, p3, "#3")],
        key=lambda x: roc_auc_score(y_va, x[1]),
    )
    print(f"  >>> Best FE set: {best_name} ({len(best_cols)} feats)")

    # Test LGB depth=1 (pure additive!) on best_cols!
    p_lgb_d1 = eval_lgb(
        Xtr_all[best_cols], y_tr, Xva_all[best_cols], y_va, CAT_COLS,
        f"4. Best FE + LGB depth=1 (pure additive!)",
        overrides=dict(max_depth=1, num_leaves=2, colsample_bytree=0.5, reg_lambda=20.0),
    )

    print("\n=== 2. 4-Family Model Diversity on Best FE (Fold 1) ===")
    # A. LogisticRegression on OOF TE + bools (pure log-odds linear GAM!)
    t0 = time.time()
    num_only = [c for c in best_cols if c not in CAT_COLS]
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr_all[num_only].to_numpy(dtype=np.float32))
    Xva_s = sc.transform(Xva_all[num_only].to_numpy(dtype=np.float32))
    lr = LogisticRegression(C=0.05, max_iter=500)
    lr.fit(Xtr_s, y_tr)
    p_lr = lr.predict_proba(Xva_s)[:, 1]
    print(f"  {'A. LogisticRegression on FE+TE (GAM)':<54} AUC={roc_auc_score(y_va, p_lr):.6f} ({time.time()-t0:.1f}s)")

    # B. XGBoost depth=2 on best_cols
    t0 = time.time()
    m_xgb = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="hist", enable_categorical=True,
        max_depth=2, learning_rate=0.05, n_estimators=6000, subsample=0.8, colsample_bytree=0.25,
        min_child_weight=2.0, reg_lambda=30.0, reg_alpha=2.0, max_bin=1024,
        random_state=SEED, n_jobs=-1, early_stopping_rounds=150,
    )
    m_xgb.fit(Xtr_all[best_cols], y_tr, eval_set=[(Xva_all[best_cols], y_va)], verbose=False)
    p_xgb = m_xgb.predict_proba(Xva_all[best_cols])[:, 1]
    print(f"  {'B. XGBoost depth=2 on Best FE':<54} AUC={roc_auc_score(y_va, p_xgb):.6f} (iter={m_xgb.best_iteration}, {time.time()-t0:.1f}s)")

    # C. CatBoost depth=3 fast on best_cols
    t0 = time.time()
    Xtr_cb = Xtr_all[best_cols].copy()
    Xva_cb = Xva_all[best_cols].copy()
    for c in CAT_COLS:
        Xtr_cb[c] = Xtr_cb[c].astype(str)
        Xva_cb[c] = Xva_cb[c].astype(str)
    m_cb = CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC", iterations=4000, learning_rate=0.05,
        depth=3, l2_leaf_reg=25.0, one_hot_max_size=20, border_count=254,
        random_seed=SEED, thread_count=-1, verbose=0,
    )
    m_cb.fit(Xtr_cb, y_tr, cat_features=CAT_COLS, eval_set=(Xva_cb, y_va), early_stopping_rounds=150)
    p_cb = m_cb.predict_proba(Xva_cb)[:, 1]
    print(f"  {'C. CatBoost depth=3 on Best FE':<54} AUC={roc_auc_score(y_va, p_cb):.6f} (iter={m_cb.get_best_iteration()}, {time.time()-t0:.1f}s)")

    def lg(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return logit(p)

    # Check blend!
    blend_3 = 0.40 * lg(best_p_lgb) + 0.35 * lg(p_xgb) + 0.25 * lg(p_cb)
    blend_4 = 0.35 * lg(best_p_lgb) + 0.30 * lg(p_xgb) + 0.20 * lg(p_cb) + 0.15 * lg(p_lr)
    blend_5 = 0.30 * lg(best_p_lgb) + 0.15 * lg(p_lgb_d1) + 0.25 * lg(p_xgb) + 0.18 * lg(p_cb) + 0.12 * lg(p_lr)
    print(f"\n  >>> Fold 1 Blend (LGB+XGB+CB)                          AUC={roc_auc_score(y_va, blend_3):.6f}")
    print(f"  >>> Fold 1 Blend (LGB+XGB+CB+LR_GAM)                   AUC={roc_auc_score(y_va, blend_4):.6f}")
    print(f"  >>> Fold 1 Blend (LGBd2+LGBd1+XGB+CB+LR_GAM)           AUC={roc_auc_score(y_va, blend_5):.6f}")


if __name__ == "__main__":
    main()
