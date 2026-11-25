"""Winning FE v3 pipeline (146 clean features) for Kaggle S6E11:
- 24 raw features (float32 / int8 / category)
- 36 multi-scale digit & decimal features (synthesizer signal)
- 72 single-column + multi-scale rounded + digit strict inner-OOF Target Encoding (log-odds)
- 14 top-signal 2-way pair strict inner-OOF Target Encoding (numeric only, zero string overfitting)
"""

import gc
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedKFold

from config import DATA_RAW_DIR, ID_COL, TARGET_COL

BOOL_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

CAT_COLS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]

NUM_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]

TOP_TE_PAIRS = [
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


def load_base_polars() -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, pd.Series]:
    train_pl = pl.read_csv(DATA_RAW_DIR / "train.csv")
    test_pl = pl.read_csv(DATA_RAW_DIR / "test.csv")
    y = train_pl[TARGET_COL].to_numpy().astype(np.int8)
    test_ids = test_pl[ID_COL].to_pandas()
    return train_pl, test_pl, y, test_ids


def build_base_feature_frames() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Series, dict]:
    """Build deterministic base + digit + rounding + pair-key tables (low-memory Polars -> aligned pandas)."""
    train_pl, test_pl, y, test_ids = load_base_polars()
    n_train = train_pl.height
    raw_cols = [c for c in test_pl.columns if c != ID_COL]

    df_all = pl.concat(
        [train_pl.select(raw_cols), test_pl.select(raw_cols)], how="vertical"
    ).with_columns(
        [pl.col(c).cast(pl.Float32) for c in NUM_COLS]
        + [pl.col(c).cast(pl.Int8) for c in BOOL_COLS]
    )
    del train_pl, test_pl
    gc.collect()

    exprs = []
    digit_cols = []
    round_cols = []

    # 2-decimal continuous features
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
        digit_cols.extend([
            f"{col}_cents", f"{col}_dec1", f"{col}_dec2",
            f"{col}_unit1", f"{col}_tens", f"{col}_hund",
        ])
        round_cols.extend([f"{col}_int", f"{col}_r1", f"{col}_div10"])

    # Wide-range coarse buckets & thousands digit
    for col in ["amount_usd", "account_balance_usd"]:
        int_part = pl.col(col).floor().cast(pl.Int64)
        exprs.extend([
            (int_part // 100).cast(pl.Int32).alias(f"{col}_div100"),
            ((int_part // 1000) % 10).cast(pl.Float32).alias(f"{col}_thou"),
        ])
        round_cols.append(f"{col}_div100")
        digit_cols.append(f"{col}_thou")

    # 1-decimal continuous features
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

    # Integer digits
    for col in ["customer_age", "card_age_months"]:
        int_part = pl.col(col).cast(pl.Int64)
        exprs.extend([
            (int_part % 10).cast(pl.Float32).alias(f"{col}_unit1"),
            (int_part // 10).cast(pl.Float32).alias(f"{col}_tens"),
        ])
        digit_cols.extend([f"{col}_unit1", f"{col}_tens"])

    df_all = df_all.with_columns(exprs)

    # Top-signal 2-way pair keys (strictly for numeric OOF TE — NOT fed as raw strings to trees!)
    pair_keys = []
    pair_exprs = []
    for a, b in TOP_TE_PAIRS:
        k = f"p__{a}__{b}"
        pair_keys.append(k)
        pair_exprs.append((pl.col(a).cast(pl.String) + "_" + pl.col(b).cast(pl.String)).alias(k))
    df_all = df_all.with_columns(pair_exprs)

    X_train = df_all.slice(0, n_train).to_pandas()
    X_test = df_all.slice(n_train, len(test_ids)).to_pandas()
    del df_all
    gc.collect()

    # Align categorical dtypes for base 5 categories
    for col in CAT_COLS:
        cats = pd.Index(X_train[col].dropna().unique()).union(
            pd.Index(X_test[col].dropna().unique())
        )
        cat_type = pd.CategoricalDtype(categories=cats, ordered=False)
        X_train[col] = X_train[col].astype(cat_type)
        X_test[col] = X_test[col].astype(cat_type)

    te_cols = NUM_COLS + CAT_COLS + round_cols + digit_cols + pair_keys
    model_base_cols = raw_cols + digit_cols

    meta = {
        "raw_cols": raw_cols,
        "cat_cols": CAT_COLS,
        "digit_cols": digit_cols,
        "round_cols": round_cols,
        "pair_keys": pair_keys,
        "te_cols": te_cols,
        "model_base_cols": model_base_cols,
    }
    return X_train, X_test, y, test_ids, meta


def add_oof_target_encoding(
    X_tr_df: pd.DataFrame,
    y_tr: np.ndarray,
    X_va_df: pd.DataFrame,
    X_te_df: pd.DataFrame | None,
    te_cols: list[str],
    keep_base_cols: list[str],
    smooth: float = 15.0,
    n_inner_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Strictly leak-free inner-OOF Smoothed Target Encoding (log-odds scale) fitted INSIDE each outer fold."""
    prior = float(np.mean(y_tr))
    inner_skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=seed)
    inner_splits = list(inner_skf.split(X_tr_df, y_tr))

    tr_new = {}
    va_new = {}
    te_new = {} if X_te_df is not None else None

    for col in te_cols:
        tr_vals = X_tr_df[col].to_numpy()
        va_vals = X_va_df[col].to_numpy()
        te_vals = X_te_df[col].to_numpy() if X_te_df is not None else None

        # Inner OOF for training fold
        oof_te = np.full(len(tr_vals), prior, dtype=np.float32)
        for in_tr, in_va in inner_splits:
            s_df = pd.DataFrame({"k": tr_vals[in_tr], "y": y_tr[in_tr]})
            stats = s_df.groupby("k", observed=False)["y"].agg(["count", "sum"])
            enc = (stats["sum"] + smooth * prior) / (stats["count"] + smooth)
            oof_te[in_va] = pd.Series(tr_vals[in_va]).map(enc).fillna(prior).to_numpy(dtype=np.float32)

        # Full outer training fold stats for validation & test
        full_df = pd.DataFrame({"k": tr_vals, "y": y_tr})
        full_stats = full_df.groupby("k", observed=False)["y"].agg(["count", "sum"])
        full_enc = (full_stats["sum"] + smooth * prior) / (full_stats["count"] + smooth)

        va_mapped = pd.Series(va_vals).map(full_enc).fillna(prior).to_numpy(dtype=np.float32)
        p_tr = np.clip(oof_te, 1e-6, 1.0 - 1e-6)
        p_va = np.clip(va_mapped, 1e-6, 1.0 - 1e-6)
        tr_new[f"te_{col}"] = np.log(p_tr / (1.0 - p_tr)).astype(np.float32)
        va_new[f"te_{col}"] = np.log(p_va / (1.0 - p_va)).astype(np.float32)

        if X_te_df is not None and te_new is not None:
            te_mapped = pd.Series(te_vals).map(full_enc).fillna(prior).to_numpy(dtype=np.float32)
            p_te = np.clip(te_mapped, 1e-6, 1.0 - 1e-6)
            te_new[f"te_{col}"] = np.log(p_te / (1.0 - p_te)).astype(np.float32)

    X_tr_out = pd.concat([X_tr_df[keep_base_cols].reset_index(drop=True), pd.DataFrame(tr_new)], axis=1)
    X_va_out = pd.concat([X_va_df[keep_base_cols].reset_index(drop=True), pd.DataFrame(va_new)], axis=1)
    X_te_out = (
        pd.concat([X_te_df[keep_base_cols].reset_index(drop=True), pd.DataFrame(te_new)], axis=1)
        if X_te_df is not None and te_new is not None
        else None
    )
    return X_tr_out, X_va_out, X_te_out
