"""Diagnostic script: inspect single-feature fraud rates, baseline errors, and model-structure probes."""

import time
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from config import DATA_RAW_DIR, ID_COL, N_FOLDS, OOF_DIR, SEED, TARGET_COL


def main() -> None:
    train = pl.read_csv(DATA_RAW_DIR / "train.csv")
    test = pl.read_csv(DATA_RAW_DIR / "test.csv")
    oof_base = np.load(OOF_DIR / "oof_lgb_baseline.npy")
    y = train[TARGET_COL].to_numpy().astype(np.int8)

    feature_cols = [c for c in test.columns if c != ID_COL]
    cat_cols = [c for c in feature_cols if train[c].dtype == pl.String]
    bool_cols = [c for c in feature_cols if train[c].dtype == pl.Boolean]
    num_cols = [c for c in feature_cols if train[c].dtype.is_numeric()]

    print("=== 1. Single-feature ROC AUC (numeric & bool) ===")
    aucs = []
    for c in num_cols + bool_cols:
        vals = train[c].cast(pl.Float64).to_numpy()
        auc = roc_auc_score(y, vals)
        aucs.append((c, max(auc, 1 - auc), "+" if auc >= 0.5 else "-"))
    for c, auc, sign in sorted(aucs, key=lambda x: -x[1]):
        print(f"  {c:<30} AUC={auc:.5f} ({sign})")

    print("\n=== 2. Fraud rate by boolean flags ===")
    for c in bool_cols:
        g = train.group_by(c).agg(pl.len().alias("n"), pl.col(TARGET_COL).mean().alias("rate")).sort(c)
        rows = ", ".join(f"{r[0]}: {r[2]*100:.2f}% (n={r[1]:,})" for r in g.iter_rows())
        print(f"  {c:<30} {rows}")

    print("\n=== 3. Fraud rate by key discrete/categorical cols ===")
    for c in ["cvv_retry_count", "prior_disputes", "auth_method", "channel", "merchant_category", "card_type", "device_type"]:
        g = train.group_by(c).agg(pl.len().alias("n"), pl.col(TARGET_COL).mean().alias("rate")).sort("rate", descending=True)
        rows = " | ".join(f"{r[0]}:{r[2]*100:.2f}%({r[1]:,})" for r in g.head(6).iter_rows())
        print(f"  {c:<20} {rows}")

    print("\n=== 4. Risk flag count vs fraud rate ===")
    flag_sum = sum(pl.col(c).cast(pl.Int8) for c in bool_cols)
    g = train.with_columns(flag_sum.alias("flag_sum")).group_by("flag_sum").agg(
        pl.len().alias("n"), pl.col(TARGET_COL).mean().alias("rate")
    ).sort("flag_sum")
    for r in g.iter_rows():
        print(f"  flag_sum={r[0]}: n={r[1]:>7,}, fraud_rate={r[2]*100:.3f}%")

    print("\n=== 5. Check ID trend / drift (AUC of id vs target, and decile rates) ===")
    print(f"  id vs is_fraud AUC: {roc_auc_score(y, train[ID_COL].to_numpy()):.5f}")

    print("\n=== 6. Fold-1 Structure Probes (LR vs Shallow LGB vs Deep LGB) ===")
    cast_exprs = [pl.col(c).cast(pl.Float32) for c in num_cols] + [pl.col(c).cast(pl.Int8) for c in bool_cols]
    X_pd = train.select(feature_cols).with_columns(cast_exprs).to_pandas()
    for c in cat_cols:
        X_pd[c] = X_pd[c].astype("category")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(skf.split(X_pd, y))
    print(f"  Baseline LGB (leaves=63, lr=0.05) Fold 1 AUC: {roc_auc_score(y[va_idx], oof_base[va_idx]):.6f}")

    # Probe A: Logistic Regression (raw)
    t0 = time.time()
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols + bool_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    lr_pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=300, C=1.0))])
    lr_pipe.fit(X_pd.iloc[tr_idx], y[tr_idx])
    lr_pred = lr_pipe.predict_proba(X_pd.iloc[va_idx])[:, 1]
    print(f"  LogisticRegression (raw)          Fold 1 AUC: {roc_auc_score(y[va_idx], lr_pred):.6f} ({time.time()-t0:.1f}s)")

    # Probe B: Shallow LGB configs
    probes = [
        ("LGB depth=2, leaves=4, lr=0.05", dict(max_depth=2, num_leaves=4, learning_rate=0.05, colsample_bytree=0.7, subsample=0.8, subsample_freq=1, min_child_samples=100, reg_lambda=10.0)),
        ("LGB depth=3, leaves=8, lr=0.05", dict(max_depth=3, num_leaves=8, learning_rate=0.05, colsample_bytree=0.7, subsample=0.8, subsample_freq=1, min_child_samples=100, reg_lambda=10.0)),
        ("LGB depth=4, leaves=15, lr=0.05", dict(max_depth=4, num_leaves=15, learning_rate=0.05, colsample_bytree=0.6, subsample=0.8, subsample_freq=1, min_child_samples=200, reg_lambda=20.0)),
        ("LGB depth=5, leaves=23, lr=0.03, col=0.5", dict(max_depth=5, num_leaves=23, learning_rate=0.03, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=200, reg_lambda=20.0)),
        ("LGB depth=3, leaves=8, lr=0.05, col=0.4", dict(max_depth=3, num_leaves=8, learning_rate=0.05, colsample_bytree=0.4, subsample=0.8, subsample_freq=1, min_child_samples=200, reg_lambda=30.0)),
    ]
    for name, cfg in probes:
        t0 = time.time()
        m = lgb.LGBMClassifier(objective="binary", metric="auc", n_estimators=4000, random_state=SEED, n_jobs=-1, verbose=-1, **cfg)
        m.fit(
            X_pd.iloc[tr_idx], y[tr_idx],
            eval_set=[(X_pd.iloc[va_idx], y[va_idx])],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        p = m.predict_proba(X_pd.iloc[va_idx])[:, 1]
        print(f"  {name:<42} Fold 1 AUC: {roc_auc_score(y[va_idx], p):.6f} (iter={m.best_iteration_}, {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
