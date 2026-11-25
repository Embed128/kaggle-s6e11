"""Super GAM (Multi-Smooth OOF TE + Cubic Splines + Clean Low-Card OneHot + 35 Top-Pair TEs)
+ 2-Layer Neural Net (MLPClassifier on log-odds TE features) for Kaggle S6E11.
Trains in ~1-2 minutes total for all 5 folds on CPU!
"""

import gc
import time

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from config import DATA_RAW_DIR, ID_COL, N_FOLDS, OOF_DIR, SEED, SUBMISSION_DIR, TARGET_COL, ensure_dirs
from features import BOOL_COLS, CAT_COLS, NUM_COLS

EXPANDED_PAIRS = [
    ("merchant_risk_score_div5", "velocity_score_div5"),
    ("merchant_risk_score_div5", "cvv_retry_count"),
    ("merchant_risk_score_div5", "auth_method"),
    ("merchant_risk_score_div5", "is_new_merchant"),
    ("merchant_risk_score_div5", "merchant_category"),
    ("merchant_risk_score_div5", "ip_country_mismatch"),
    ("velocity_score_div5", "cvv_retry_count"),
    ("velocity_score_div5", "auth_method"),
    ("velocity_score_div5", "is_new_merchant"),
    ("velocity_score_div5", "merchant_category"),
    ("cvv_retry_count", "auth_method"),
    ("cvv_retry_count", "merchant_category"),
    ("cvv_retry_count", "is_new_merchant"),
    ("cvv_retry_count", "ip_country_mismatch"),
    ("cvv_retry_count", "billing_shipping_mismatch"),
    ("cvv_retry_count", "is_foreign_transaction"),
    ("cvv_retry_count", "used_vpn"),
    ("cvv_retry_count", "flag_bitmask"),
    ("auth_method", "merchant_category"),
    ("auth_method", "is_new_merchant"),
    ("auth_method", "channel"),
    ("auth_method", "flag_bitmask"),
    ("merchant_category", "is_new_merchant"),
    ("merchant_category", "channel"),
    ("merchant_category", "flag_bitmask"),
    ("ip_country_mismatch", "is_foreign_transaction"),
    ("ip_country_mismatch", "used_vpn"),
    ("ip_country_mismatch", "is_new_merchant"),
    ("billing_shipping_mismatch", "is_new_merchant"),
    ("billing_shipping_mismatch", "is_foreign_transaction"),
    ("customer_age", "merchant_category"),
    ("customer_age", "auth_method"),
    ("customer_age", "cvv_retry_count"),
    ("hours_since_last_txn_int", "txn_count_last_24h"),
    ("amount_usd_div10", "merchant_category"),
]

CLEAN_OHE_COLS = CAT_COLS + [
    "cvv_retry_count",
    "prior_disputes",
    "time_of_day_hour",
    "day_of_week",
    "txn_count_last_24h",
    "flag_bitmask",
]


def build_super_gam_base() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Series, list[str], list[str]]:
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

    bitmask = (
        pl.col("is_foreign_transaction").cast(pl.Int16) * 1
        + pl.col("is_new_merchant").cast(pl.Int16) * 2
        + pl.col("used_vpn").cast(pl.Int16) * 4
        + pl.col("ip_country_mismatch").cast(pl.Int16) * 8
        + pl.col("billing_shipping_mismatch").cast(pl.Int16) * 16
        + pl.col("is_ai_generated_scam_attempt").cast(pl.Int16) * 32
    ).alias("flag_bitmask")

    exprs = [bitmask]
    te_low_card = list(CAT_COLS) + [
        "cvv_retry_count", "prior_disputes", "time_of_day_hour", "day_of_week",
        "txn_count_last_24h", "customer_age", "card_age_months", "flag_bitmask",
    ]
    te_high_card = [
        "merchant_risk_score", "velocity_score",
        "amount_usd", "account_balance_usd",
        "hours_since_last_txn", "distance_from_home_km",
    ]

    for col in ["amount_usd", "account_balance_usd", "hours_since_last_txn", "distance_from_home_km"]:
        scaled100 = (pl.col(col) * 100.0).round(0).cast(pl.Int64)
        scaled10 = (pl.col(col) * 10.0).round(0).cast(pl.Int64)
        int_part = pl.col(col).floor().cast(pl.Int64)
        names_lc = [f"{col}_cents", f"{col}_dec1", f"{col}_dec2", f"{col}_unit1", f"{col}_tens", f"{col}_hund"]
        names_hc = [f"{col}_int", f"{col}_r1", f"{col}_div10"]
        exprs.extend([
            (scaled100 % 100).cast(pl.Int32).alias(names_lc[0]),
            ((scaled100 // 10) % 10).cast(pl.Int32).alias(names_lc[1]),
            (scaled100 % 10).cast(pl.Int32).alias(names_lc[2]),
            (int_part % 10).cast(pl.Int32).alias(names_lc[3]),
            ((int_part // 10) % 10).cast(pl.Int32).alias(names_lc[4]),
            ((int_part // 100) % 10).cast(pl.Int32).alias(names_lc[5]),
            int_part.cast(pl.Int32).alias(names_hc[0]),
            scaled10.cast(pl.Int32).alias(names_hc[1]),
            (int_part // 10).cast(pl.Int32).alias(names_hc[2]),
        ])
        te_low_card.extend(names_lc)
        te_high_card.extend(names_hc)

    for col in ["amount_usd", "account_balance_usd"]:
        int_part = pl.col(col).floor().cast(pl.Int64)
        exprs.extend([
            (int_part // 100).cast(pl.Int32).alias(f"{col}_div100"),
            ((int_part // 1000) % 10).cast(pl.Int32).alias(f"{col}_thou"),
        ])
        te_high_card.append(f"{col}_div100")
        te_low_card.append(f"{col}_thou")

    for col in ["merchant_risk_score", "velocity_score"]:
        scaled10 = (pl.col(col) * 10.0).round(0).cast(pl.Int64)
        int_part = pl.col(col).floor().cast(pl.Int64)
        exprs.extend([
            (scaled10 % 10).cast(pl.Int32).alias(f"{col}_dec1"),
            (int_part % 10).cast(pl.Int32).alias(f"{col}_unit1"),
            ((int_part // 10) % 10).cast(pl.Int32).alias(f"{col}_tens"),
            int_part.cast(pl.Int32).alias(f"{col}_int"),
            (int_part // 5).cast(pl.Int32).alias(f"{col}_div5"),
        ])
        te_low_card.extend([f"{col}_dec1", f"{col}_unit1", f"{col}_tens", f"{col}_int", f"{col}_div5"])

    for col in ["customer_age", "card_age_months"]:
        int_part = pl.col(col).cast(pl.Int64)
        exprs.extend([
            (int_part % 10).cast(pl.Int32).alias(f"{col}_unit1"),
            (int_part // 10).cast(pl.Int32).alias(f"{col}_tens"),
        ])
        te_low_card.extend([f"{col}_unit1", f"{col}_tens"])

    df = df.with_columns(exprs)

    pair_exprs = []
    for a, b in EXPANDED_PAIRS:
        k = f"p__{a}__{b}"
        te_high_card.append(k)
        pair_exprs.append((pl.col(a).cast(pl.String) + "_" + pl.col(b).cast(pl.String)).alias(k))
    df = df.with_columns(pair_exprs)

    X_tr = df.slice(0, n_train).to_pandas()
    X_te = df.slice(n_train, len(test_ids)).to_pandas()
    del df
    gc.collect()
    return X_tr, X_te, y, test_ids, te_low_card, te_high_card


def compute_multi_smooth_te(
    X_tr_df: pd.DataFrame,
    y_tr: np.ndarray,
    X_va_df: pd.DataFrame,
    X_te_df: pd.DataFrame,
    te_low_card: list[str],
    te_high_card: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute strictly leak-free inner-OOF log-odds TE with adaptive multi-smoothing."""
    prior = float(np.mean(y_tr))
    inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    inner_splits = list(inner_skf.split(X_tr_df, y_tr))

    # Low-card cols: smooth=10; High-card cols: BOTH smooth=5 and smooth=40 so LR/MLP can weight rare vs common!
    specs = [(c, [10.0]) for c in te_low_card] + [(c, [5.0, 40.0]) for c in te_high_card]

    tr_cols, va_cols, te_cols_list = [], [], []
    for col, smooths in specs:
        tr_v = X_tr_df[col].to_numpy()
        va_v = X_va_df[col].to_numpy()
        te_v = X_te_df[col].to_numpy()

        # Precompute inner fold stats once per column!
        oof_s = {s: np.full(len(tr_v), prior, dtype=np.float32) for s in smooths}
        for in_tr, in_va in inner_splits:
            stats = pd.DataFrame({"k": tr_v[in_tr], "y": y_tr[in_tr]}).groupby("k", observed=False)["y"].agg(["count", "sum"])
            va_s = pd.Series(tr_v[in_va])
            for s in smooths:
                enc = (stats["sum"] + s * prior) / (stats["count"] + s)
                oof_s[s][in_va] = va_s.map(enc).fillna(prior).to_numpy(dtype=np.float32)

        full_stats = pd.DataFrame({"k": tr_v, "y": y_tr}).groupby("k", observed=False)["y"].agg(["count", "sum"])
        va_series = pd.Series(va_v)
        te_series = pd.Series(te_v)
        for s in smooths:
            enc = (full_stats["sum"] + s * prior) / (full_stats["count"] + s)
            p_tr = np.clip(oof_s[s], 1e-6, 1.0 - 1e-6)
            p_va = np.clip(va_series.map(enc).fillna(prior).to_numpy(dtype=np.float32), 1e-6, 1.0 - 1e-6)
            p_te = np.clip(te_series.map(enc).fillna(prior).to_numpy(dtype=np.float32), 1e-6, 1.0 - 1e-6)
            tr_cols.append(np.log(p_tr / (1.0 - p_tr)).astype(np.float32))
            va_cols.append(np.log(p_va / (1.0 - p_va)).astype(np.float32))
            te_cols_list.append(np.log(p_te / (1.0 - p_te)).astype(np.float32))

    return (
        np.column_stack(tr_cols),
        np.column_stack(va_cols),
        np.column_stack(te_cols_list),
    )


def main() -> None:
    ensure_dirs()
    t0 = time.time()
    X_tr_base, X_te_base, y, test_ids, te_lc, te_hc = build_super_gam_base()
    print(f"Base ready in {time.time()-t0:.1f}s | te_low={len(te_lc)}, te_high={len(te_hc)}")

    # Pre-encode shared spline + clean low-card OneHot block (unsupervised, fit on train fold inside loop)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_lr = np.zeros(len(y), dtype=np.float64)
    test_lr = np.zeros(len(test_ids), dtype=np.float64)
    oof_mlp = np.zeros(len(y), dtype=np.float64)
    test_mlp = np.zeros(len(test_ids), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_base, y), start=1):
        ft0 = time.time()
        te_tr, te_va, te_te = compute_multi_smooth_te(
            X_tr_base.iloc[tr_idx], y[tr_idx], X_tr_base.iloc[va_idx], X_te_base, te_lc, te_hc, seed=SEED + fold
        )

        # Clean Splines + Booleans + Low-Card OneHot
        pre = ColumnTransformer([
            ("num", Pipeline([("sc", StandardScaler()), ("sp", SplineTransformer(n_knots=8, degree=3, include_bias=False))]), NUM_COLS),
            ("bool", "passthrough", BOOL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CLEAN_OHE_COLS),
        ])
        base_tr = pre.fit_transform(X_tr_base.iloc[tr_idx]).astype(np.float32)
        base_va = pre.transform(X_tr_base.iloc[va_idx]).astype(np.float32)
        base_te = pre.transform(X_te_base).astype(np.float32)

        te_sc = StandardScaler()
        te_tr_s = te_sc.fit_transform(te_tr).astype(np.float32)
        te_va_s = te_sc.transform(te_va).astype(np.float32)
        te_te_s = te_sc.transform(te_te).astype(np.float32)

        F_tr = np.hstack([te_tr_s, base_tr])
        F_va = np.hstack([te_va_s, base_va])
        F_te = np.hstack([te_te_s, base_te])

        # 1. Super GAM (L2 Logistic Regression)
        t1 = time.time()
        lr = LogisticRegression(C=0.02, solver="lbfgs", max_iter=600)
        lr.fit(F_tr, y[tr_idx])
        va_p_lr = lr.predict_proba(F_va)[:, 1]
        oof_lr[va_idx] = va_p_lr
        test_lr += lr.predict_proba(F_te)[:, 1] / N_FOLDS
        print(f"  [super_gam] Fold {fold}/{N_FOLDS} AUC = {roc_auc_score(y[va_idx], va_p_lr):.6f} (feats={F_tr.shape[1]}, lr={time.time()-t1:.1f}s, total={time.time()-ft0:.1f}s)")

        # 2. Compact 2-Layer Neural Network (MLP with strong L2 alpha + early stopping)
        t2 = time.time()
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 32),
            activation="relu",
            alpha=0.5,
            batch_size=1024,
            learning_rate_init=1e-3,
            max_iter=25,
            early_stopping=True,
            n_iter_no_change=4,
            validation_fraction=0.1,
            random_state=SEED + fold,
        )
        mlp.fit(F_tr, y[tr_idx])
        va_p_mlp = mlp.predict_proba(F_va)[:, 1]
        oof_mlp[va_idx] = va_p_mlp
        test_mlp += mlp.predict_proba(F_te)[:, 1] / N_FOLDS
        print(f"  [mlp_gam  ] Fold {fold}/{N_FOLDS} AUC = {roc_auc_score(y[va_idx], va_p_mlp):.6f} (epochs={mlp.n_iter_}, {time.time()-t2:.1f}s)")

        del F_tr, F_va, F_te, base_tr, base_va, base_te, te_tr, te_va, te_te
        gc.collect()

    print("\n========================================================")
    print(f"  super_gam  | Overall OOF = {roc_auc_score(y, oof_lr):.6f}")
    print(f"  mlp_gam    | Overall OOF = {roc_auc_score(y, oof_mlp):.6f}")
    print("========================================================")

    np.save(OOF_DIR / "oof_v2_super_gam.npy", oof_lr)
    np.save(OOF_DIR / "test_v2_super_gam.npy", test_lr)
    np.save(OOF_DIR / "oof_v2_mlp_gam.npy", oof_mlp)
    np.save(OOF_DIR / "test_v2_mlp_gam.npy", test_mlp)


if __name__ == "__main__":
    main()
