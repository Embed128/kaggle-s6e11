"""Fast Fold-1 probe: depth=1 (pure additive), max_bin, Spline GAM, shallow XGB/CatBoost, and nonlinear curves."""

import time
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, SplineTransformer, StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from config import DATA_RAW_DIR, ID_COL, N_FOLDS, SEED, TARGET_COL


def main() -> None:
    train = pl.read_csv(DATA_RAW_DIR / "train.csv")
    y = train[TARGET_COL].to_numpy().astype(np.int8)
    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
    cat_cols = [c for c in feature_cols if train[c].dtype == pl.String]
    bool_cols = [c for c in feature_cols if train[c].dtype == pl.Boolean]
    num_cols = [c for c in feature_cols if train[c].dtype.is_numeric()]

    cast_exprs = [pl.col(c).cast(pl.Float32) for c in num_cols] + [pl.col(c).cast(pl.Int8) for c in bool_cols]
    X_pd = train.select(feature_cols).with_columns(cast_exprs).to_pandas()
    for c in cat_cols:
        X_pd[c] = X_pd[c].astype("category")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(skf.split(X_pd, y))
    X_tr, y_tr = X_pd.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X_pd.iloc[va_idx], y[va_idx]

    preds = {}

    # 1. Spline Logistic Regression (Smooth GAM!)
    t0 = time.time()
    pre_spline = ColumnTransformer([
        ("num", Pipeline([("sc", StandardScaler()), ("sp", SplineTransformer(n_knots=12, degree=3))]), num_cols),
        ("bool", "passthrough", bool_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    gam = Pipeline([("pre", pre_spline), ("clf", LogisticRegression(max_iter=500, C=0.5))])
    gam.fit(X_tr, y_tr)
    p = gam.predict_proba(X_va)[:, 1]
    preds["Spline_GAM"] = p
    print(f"Spline_GAM (12 knots)                AUC={roc_auc_score(y_va, p):.6f} ({time.time()-t0:.1f}s)")

    # 2. KBins + Log1p Logistic Regression
    t0 = time.time()
    pre_bins = ColumnTransformer([
        ("num", KBinsDiscretizer(n_bins=40, encode="onehot", strategy="quantile"), num_cols),
        ("bool", "passthrough", bool_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    bgam = Pipeline([("pre", pre_bins), ("clf", LogisticRegression(max_iter=500, C=0.2))])
    bgam.fit(X_tr, y_tr)
    p = bgam.predict_proba(X_va)[:, 1]
    preds["Binned_GAM"] = p
    print(f"Binned_GAM (40 quantile bins)        AUC={roc_auc_score(y_va, p):.6f} ({time.time()-t0:.1f}s)")

    # 3. LGB probes: depth=1, depth=2 with high max_bin, depth=2 low colsample
    lgb_probes = [
        ("LGB_d1_stump", dict(max_depth=1, num_leaves=2, learning_rate=0.05, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, min_child_samples=100, reg_lambda=10.0, max_bin=511)),
        ("LGB_d2_bin511_c04", dict(max_depth=2, num_leaves=4, learning_rate=0.05, colsample_bytree=0.4, subsample=0.8, subsample_freq=1, min_child_samples=150, reg_lambda=20.0, max_bin=511)),
        ("LGB_d2_bin1023_c03", dict(max_depth=2, num_leaves=4, learning_rate=0.05, colsample_bytree=0.3, subsample=0.8, subsample_freq=1, min_child_samples=200, reg_lambda=30.0, max_bin=1023)),
        ("LGB_d3_bin511_c03", dict(max_depth=3, num_leaves=6, learning_rate=0.05, colsample_bytree=0.3, subsample=0.8, subsample_freq=1, min_child_samples=200, reg_lambda=30.0, max_bin=511)),
    ]
    for name, cfg in lgb_probes:
        t0 = time.time()
        m = lgb.LGBMClassifier(objective="binary", metric="auc", n_estimators=6000, random_state=SEED, n_jobs=-1, verbose=-1, **cfg)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], categorical_feature=cat_cols, callbacks=[lgb.early_stopping(150, verbose=False)])
        p = m.predict_proba(X_va)[:, 1]
        preds[name] = p
        print(f"{name:<36} AUC={roc_auc_score(y_va, p):.6f} (iter={m.best_iteration_}, {time.time()-t0:.1f}s)")

    # 4. XGBoost shallow probes
    for name, depth, col in [("XGB_d2_c04", 2, 0.4), ("XGB_d3_c04", 3, 0.4)]:
        t0 = time.time()
        m = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="auc", tree_method="hist", enable_categorical=True,
            max_depth=depth, learning_rate=0.05, n_estimators=5000, subsample=0.8, colsample_bytree=col,
            min_child_weight=50, reg_lambda=20.0, reg_alpha=1.0, max_bin=512,
            random_state=SEED, n_jobs=-1, early_stopping_rounds=150,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p = m.predict_proba(X_va)[:, 1]
        preds[name] = p
        print(f"{name:<36} AUC={roc_auc_score(y_va, p):.6f} (iter={m.best_iteration}, {time.time()-t0:.1f}s)")

    # 5. CatBoost shallow probe (depth=3, 4)
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_va_cb[c] = X_va_cb[c].astype(str)
    for name, depth in [("CB_d3", 3), ("CB_d4", 4)]:
        t0 = time.time()
        m = CatBoostClassifier(
            loss_function="Logloss", eval_metric="AUC", iterations=4000, learning_rate=0.05,
            depth=depth, l2_leaf_reg=15.0, random_seed=SEED, thread_count=-1, verbose=0,
        )
        m.fit(X_tr_cb, y_tr, cat_features=cat_cols, eval_set=(X_va_cb, y_va), early_stopping_rounds=150)
        p = m.predict_proba(X_va_cb)[:, 1]
        preds[name] = p
        print(f"{name:<36} AUC={roc_auc_score(y_va, p):.6f} (iter={m.get_best_iteration()}, {time.time()-t0:.1f}s)")

    # Quick blend check on Fold 1
    from scipy.special import logit
    def to_logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return logit(p)

    blend = (
        0.25 * to_logit(preds["Spline_GAM"])
        + 0.25 * to_logit(preds["LGB_d2_bin1023_c03"])
        + 0.25 * to_logit(preds["XGB_d2_c04"])
        + 0.25 * to_logit(preds["CB_d3"])
    )
    print(f"\nEqual Log-Odds Blend (GAM+LGB+XGB+CB) Fold 1 AUC: {roc_auc_score(y_va, blend):.6f}")


if __name__ == "__main__":
    main()
