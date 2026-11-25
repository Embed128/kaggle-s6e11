"""Test combined OOF TE + Digit/Decimal features and inspect which specific digits/TEs carry signal."""

import time
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import N_FOLDS, SEED
from features import BOOL_COLS, CAT_COLS, NUM_COLS, add_oof_target_encoding, load_base_polars, to_aligned_pandas


def build_digit_and_round_features(df_all: pl.DataFrame) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Extract digit/decimal/fractional & rounded-bucket features across continuous numeric columns."""
    # Exact decimal / digit extractions (classic Playground Series synthesizer signal)
    exprs = []
    digit_cols = []
    round_cols = []

    # 2-decimal continuous cols: amount_usd, account_balance_usd, hours_since_last_txn, distance_from_home_km
    for col in ["amount_usd", "account_balance_usd", "hours_since_last_txn", "distance_from_home_km"]:
        scaled100 = (pl.col(col) * 100.0).round(0).cast(pl.Int64)
        scaled10 = (pl.col(col) * 10.0).round(0).cast(pl.Int64)
        int_part = pl.col(col).floor().cast(pl.Int64)

        c_cents = f"{col}_cents"          # 0..99
        c_d1 = f"{col}_dec1"              # 1st decimal digit 0..9
        c_d2 = f"{col}_dec2"              # 2nd decimal digit 0..9
        c_u1 = f"{col}_unit1"             # last integer digit 0..9
        c_u2 = f"{col}_tens"              # tens digit 0..9
        c_int = f"{col}_int"              # integer floor (for OOF TE)
        c_r1 = f"{col}_r1"                # rounded to 0.1 (for OOF TE)

        exprs.extend([
            (scaled100 % 100).cast(pl.Float32).alias(c_cents),
            ((scaled100 // 10) % 10).cast(pl.Float32).alias(c_d1),
            (scaled100 % 10).cast(pl.Float32).alias(c_d2),
            (int_part % 10).cast(pl.Float32).alias(c_u1),
            ((int_part // 10) % 10).cast(pl.Float32).alias(c_u2),
            int_part.cast(pl.Int32).alias(c_int),
            scaled10.cast(pl.Int32).alias(c_r1),
        ])
        digit_cols.extend([c_cents, c_d1, c_d2, c_u1, c_u2])
        round_cols.extend([c_int, c_r1])

    # 1-decimal continuous cols: merchant_risk_score, velocity_score
    for col in ["merchant_risk_score", "velocity_score"]:
        scaled10 = (pl.col(col) * 10.0).round(0).cast(pl.Int64)
        int_part = pl.col(col).floor().cast(pl.Int64)

        c_d1 = f"{col}_dec1"
        c_u1 = f"{col}_unit1"
        c_int = f"{col}_int"
        exprs.extend([
            (scaled10 % 10).cast(pl.Float32).alias(c_d1),
            (int_part % 10).cast(pl.Float32).alias(c_u1),
            int_part.cast(pl.Int32).alias(c_int),
        ])
        digit_cols.extend([c_d1, c_u1])
        round_cols.append(c_int)

    # Integer cols digits: customer_age, card_age_months
    for col in ["customer_age", "card_age_months"]:
        int_part = pl.col(col).cast(pl.Int64)
        c_u1 = f"{col}_unit1"
        exprs.append((int_part % 10).cast(pl.Float32).alias(c_u1))
        digit_cols.append(c_u1)

    return df_all.with_columns(exprs), digit_cols, round_cols


def eval_lgb(X_tr, y_tr, X_va, y_va, cats, label, overrides=None):
    t0 = time.time()
    cfg = dict(
        objective="binary", metric="auc", max_depth=2, num_leaves=4,
        learning_rate=0.05, n_estimators=8000, colsample_bytree=0.3,
        subsample=0.8, subsample_freq=1, min_child_samples=200,
        reg_lambda=30.0, max_bin=1023, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    if overrides:
        cfg.update(overrides)
    m = lgb.LGBMClassifier(**cfg)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], categorical_feature=cats, callbacks=[lgb.early_stopping(150, verbose=False)])
    p = m.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, p)
    print(f"  {label:<52} AUC={auc:.6f} (iter={m.best_iteration_}, {time.time()-t0:.1f}s)")
    return m, p, auc


def main() -> None:
    train_pl, test_pl, y, _ = load_base_polars()
    n_train = train_pl.height
    feat_cols = [c for c in test_pl.columns if c != "id"]
    df_all = pl.concat([train_pl.select(feat_cols), test_pl.select(feat_cols)], how="vertical").with_columns(
        [pl.col(c).cast(pl.Float32) for c in NUM_COLS] + [pl.col(c).cast(pl.Int8) for c in BOOL_COLS]
    )

    df_ext, digit_cols, round_cols = build_digit_and_round_features(df_all)
    X_all, _ = to_aligned_pandas(df_ext.slice(0, n_train), df_ext.slice(n_train, test_pl.height), CAT_COLS)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(skf.split(X_all, y))
    y_tr, y_va = y[tr_idx], y[va_idx]

    # Inspect individual digit single-feature AUCs!
    print("=== 1. Top Digit Features by Single-Feature Fraud-Rate Spread ===")
    spreads = []
    for c in digit_cols:
        s = pd.DataFrame({"k": X_all[c].iloc[tr_idx].to_numpy(), "y": y_tr}).groupby("k")["y"].agg(["count", "mean"])
        s = s[s["count"] >= 200]
        if len(s) >= 2:
            spreads.append((c, s["mean"].min() * 100, s["mean"].max() * 100, (s["mean"].max() - s["mean"].min()) * 100))
    for c, mn, mx, sp in sorted(spreads, key=lambda x: -x[3])[:12]:
        print(f"  {c:<30} min={mn:.2f}% max={mx:.2f}% spread={sp:.2f}%")

    print("\n=== 2. Combined FE v2 Ablations (Fold 1) ===")
    raw_cols = feat_cols

    # 1. Raw + Digits only (no round_cols)
    cols_dig = raw_cols + digit_cols
    eval_lgb(X_all[cols_dig].iloc[tr_idx], y_tr, X_all[cols_dig].iloc[va_idx], y_va, CAT_COLS, f"1. Raw + All Digits ({len(cols_dig)} feats)")

    # 2. Raw + All-Numeric & Cat OOF TE (exact values + rounded)
    te_base = NUM_COLS + CAT_COLS + round_cols
    Xtr_te1, Xva_te1, _ = add_oof_target_encoding(
        X_all.iloc[tr_idx], y_tr, X_all.iloc[va_idx], None, te_cols=te_base, smooth=25.0
    )
    te1_names = [f"te_{c}" for c in te_base]
    # Keep raw + te1 (drop raw round_cols and digits first)
    eval_lgb(Xtr_te1[raw_cols + te1_names], y_tr, Xva_te1[raw_cols + te1_names], y_va, CAT_COLS, f"2. Raw + All Num/Cat/Round OOF TE ({len(raw_cols + te1_names)})")

    # 3. Raw + Digits + All OOF TE (incl. digit OOF TE!)
    te_all = te_base + digit_cols
    Xtr_te2, Xva_te2, _ = add_oof_target_encoding(
        X_all.iloc[tr_idx], y_tr, X_all.iloc[va_idx], None, te_cols=te_all, smooth=25.0
    )
    keep_cols = raw_cols + digit_cols + [f"te_{c}" for c in te_all]
    m3, p3, _ = eval_lgb(Xtr_te2[keep_cols], y_tr, Xva_te2[keep_cols], y_va, CAT_COLS, f"3. Raw + Digits + All OOF TE ({len(keep_cols)})")

    # 4. Try depth=3 and depth=4 on #3! (Maybe with TE+Digits, deeper trees now help!)
    m4, p4, _ = eval_lgb(
        Xtr_te2[keep_cols], y_tr, Xva_te2[keep_cols], y_va, CAT_COLS,
        f"4. #3 with LGB depth=3, leaves=7, col=0.25",
        overrides=dict(max_depth=3, num_leaves=7, colsample_bytree=0.25, reg_lambda=40.0),
    )
    m5, p5, _ = eval_lgb(
        Xtr_te2[keep_cols], y_tr, Xva_te2[keep_cols], y_va, CAT_COLS,
        f"5. #3 with LGB depth=4, leaves=15, col=0.25",
        overrides=dict(max_depth=4, num_leaves=15, colsample_bytree=0.25, reg_lambda=50.0),
    )

    # Print top 20 features by gain in best model
    best_m = max([(m3, roc_auc_score(y_va, p3)), (m4, roc_auc_score(y_va, p4)), (m5, roc_auc_score(y_va, p5))], key=lambda x: x[1])[0]
    imp = pd.DataFrame({"f": keep_cols, "gain": best_m.booster_.feature_importance("gain")}).sort_values("gain", ascending=False)
    print("\nTop 20 Features (Gain):")
    for r in imp.head(20).itertuples(index=False):
        print(f"  {r.f:<36} {r.gain:>14,.1f}")


if __name__ == "__main__":
    main()
