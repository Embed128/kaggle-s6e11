"""20-second Fold-1 verification of V3 ideas:
1. Poly2 Log-Odds GAM (degree-2 interactions among top TE log-odds to capture smooth tail saturation)
2. Dual-representation exact-value categoricals for LightGBM with strong cat_smooth
"""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from config import N_FOLDS, SEED
from features import CAT_COLS, NUM_COLS, add_oof_target_encoding, build_base_feature_frames


def main() -> None:
    X_train_base, X_test_base, y, _, meta = build_base_feature_frames()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(skf.split(X_train_base, y))
    y_tr, y_va = y[tr_idx], y[va_idx]

    X_tr, X_va, _ = add_oof_target_encoding(
        X_train_base.iloc[tr_idx], y_tr, X_train_base.iloc[va_idx], None,
        te_cols=meta["te_cols"], keep_base_cols=meta["model_base_cols"], smooth=15.0, seed=SEED + 1
    )

    num_cols = [c for c in X_tr.columns if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])

    # 1. Baseline V2 lr_gam (C=0.05)
    t0 = time.time()
    m1 = Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.05, max_iter=500))])
    m1.fit(X_tr, y_tr)
    p1 = m1.predict_proba(X_va)[:, 1]
    print(f"1. V2 lr_gam (C=0.05)                    Fold 1 AUC = {roc_auc_score(y_va, p1):.6f} ({time.time()-t0:.1f}s)")

    # 2. Tune C on lr_gam + try Poly2 on top 18 strongest signals!
    for c in [0.005, 0.01, 0.02]:
        t0 = time.time()
        m = Pipeline([("pre", pre), ("clf", LogisticRegression(C=c, max_iter=500))])
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)[:, 1]
        print(f"   lr_gam (C={c:<5})                      Fold 1 AUC = {roc_auc_score(y_va, p):.6f} ({time.time()-t0:.1f}s)")

    # 3. Poly2-GAM: add degree-2 polynomial interactions of top 20 core log-odds/bool signals
    core_cols = [
        "te_merchant_risk_score", "te_merchant_risk_score_int",
        "te_velocity_score", "te_velocity_score_int",
        "te_cvv_retry_count", "te_auth_method", "te_merchant_category",
        "te_customer_age", "te_hours_since_last_txn", "te_amount_usd",
        "te_account_balance_usd_int", "te_distance_from_home_km",
        "is_new_merchant", "ip_country_mismatch", "billing_shipping_mismatch",
        "used_vpn", "is_foreign_transaction", "is_ai_generated_scam_attempt",
        "te_hours_since_last_txn_cents", "te_amount_usd_cents",
    ]
    pre_poly = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("poly", Pipeline([("sc", StandardScaler()), ("p2", PolynomialFeatures(degree=2, include_bias=False))]), core_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    for c in [0.002, 0.005, 0.01]:
        t0 = time.time()
        mp = Pipeline([("pre", pre_poly), ("clf", LogisticRegression(C=c, max_iter=500))])
        mp.fit(X_tr, y_tr)
        pp = mp.predict_proba(X_va)[:, 1]
        print(f"2. Poly2-GAM (C={c:<5}, +230 poly2)      Fold 1 AUC = {roc_auc_score(y_va, pp):.6f} ({time.time()-t0:.1f}s)")

    # 4. Dual-representation LightGBM: add exact-value categorical copies of top continuous cols with cat_smooth=50
    X_tr_dual = X_tr.copy()
    X_va_dual = X_va.copy()
    dual_cats = list(CAT_COLS)
    for col in ["merchant_risk_score", "velocity_score", "customer_age", "card_age_months", "hours_since_last_txn", "distance_from_home_km"]:
        cname = f"{col}_cat"
        cats = pd.Index(X_tr_dual[col].unique()).union(pd.Index(X_va_dual[col].unique()))
        ctype = pd.CategoricalDtype(categories=cats, ordered=False)
        X_tr_dual[cname] = X_tr_dual[col].astype(ctype)
        X_va_dual[cname] = X_va_dual[col].astype(ctype)
        dual_cats.append(cname)

    t0 = time.time()
    m_lgb = lgb.LGBMClassifier(
        objective="binary", metric="auc", max_depth=2, num_leaves=4,
        learning_rate=0.03, n_estimators=8000, colsample_bytree=0.25,
        subsample=0.8, subsample_freq=1, min_child_samples=200,
        reg_lambda=35.0, reg_alpha=2.0, max_bin=1023,
        cat_smooth=50.0, cat_l2=30.0,
        random_state=SEED + 1, n_jobs=-1, verbose=-1,
    )
    m_lgb.fit(X_tr_dual, y_tr, eval_set=[(X_va_dual, y_va)], categorical_feature=dual_cats, callbacks=[lgb.early_stopping(200, verbose=False)])
    p_dual = m_lgb.predict_proba(X_va_dual)[:, 1]
    print(f"3. LGB_d2 + Dual Exact Cats (smooth=50)  Fold 1 AUC = {roc_auc_score(y_va, p_dual):.6f} (iter={m_lgb.best_iteration_}, {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
