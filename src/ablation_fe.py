"""Fast Fold-1 ablation: test FE groups + fixed XGBoost (min_child_weight=1) + CatBoost (fast)."""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import N_FOLDS, SEED
from features import (
    CAT_COLS,
    INTERACTION_PAIRS,
    add_domain_and_freq_features,
    add_oof_target_encoding,
    load_base_polars,
    to_aligned_pandas,
)

LGB_FAST_PARAMS = dict(
    objective="binary",
    metric="auc",
    max_depth=2,
    num_leaves=4,
    learning_rate=0.05,
    n_estimators=6000,
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


def eval_lgb_fold1(X_tr, y_tr, X_va, y_va, cat_cols, label, extra_params=None):
    t0 = time.time()
    params = {**LGB_FAST_PARAMS, **(extra_params or {})}
    m = lgb.LGBMClassifier(**params)
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    p = m.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, p)
    print(f"  {label:<46} AUC={auc:.6f} (iter={m.best_iteration_}, {time.time()-t0:.1f}s)")
    return p, auc


def main() -> None:
    train_pl, test_pl, y, _ = load_base_polars()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(skf.split(np.zeros(len(y)), y))
    y_tr, y_va = y[tr_idx], y[va_idx]

    print("=== 1. Feature Group Ablation (LGB depth=2, max_bin=1023) ===")
    # A. Raw only
    tr_a, te_a, cats_a = add_domain_and_freq_features(train_pl, test_pl, False, False, False)
    Xa, _ = to_aligned_pandas(tr_a, te_a, cats_a)
    p_raw, _ = eval_lgb_fold1(Xa.iloc[tr_idx], y_tr, Xa.iloc[va_idx], y_va, cats_a, "A. Raw (24 feats)")

    # B. Raw + Domain/Interactions
    tr_b, te_b, cats_b = add_domain_and_freq_features(train_pl, test_pl, True, False, False)
    Xb, _ = to_aligned_pandas(tr_b, te_b, cats_b)
    eval_lgb_fold1(Xb.iloc[tr_idx], y_tr, Xb.iloc[va_idx], y_va, cats_b, f"B. +Domain+Pairs ({Xb.shape[1]} feats)")

    # C. Raw + Domain + Freq + GroupStats
    tr_c, te_c, cats_c = add_domain_and_freq_features(train_pl, test_pl, True, True, True)
    Xc, _ = to_aligned_pandas(tr_c, te_c, cats_c)
    p_fe, _ = eval_lgb_fold1(Xc.iloc[tr_idx], y_tr, Xc.iloc[va_idx], y_va, cats_c, f"C. +Domain+Freq+Grp ({Xc.shape[1]} feats)")

    # D. + Strict Inner-OOF Target Encoding on discrete/high-signal cols + pairs
    te_cols = [
        "merchant_risk_score", "velocity_score", "customer_age", "card_age_months",
        "time_of_day_hour", "merchant_category", "auth_method", "device_type", "flag_bitmask",
    ] + [f"{a}__x__{b}" for a, b in INTERACTION_PAIRS]
    t0 = time.time()
    Xtr_d, Xva_d, _ = add_oof_target_encoding(
        Xc.iloc[tr_idx], y_tr, Xc.iloc[va_idx], None, te_cols=te_cols, smooth=30.0
    )
    print(f"  [TE built in {time.time()-t0:.1f}s -> {Xtr_d.shape[1]} feats]")
    p_fe_te, _ = eval_lgb_fold1(Xtr_d, y_tr, Xva_d, y_va, cats_c, f"D. C + Inner-OOF TE ({Xtr_d.shape[1]} feats)")
    p_fe_te_d3, _ = eval_lgb_fold1(
        Xtr_d, y_tr, Xva_d, y_va, cats_c, f"D2. C+TE LGB depth=3, leaves=7",
        extra_params=dict(max_depth=3, num_leaves=7, colsample_bytree=0.25, reg_lambda=40.0),
    )

    print("\n=== 2. Fixed XGBoost & Fast CatBoost on Best Features ===")
    # Check XGBoost on Raw vs FE+TE with FIXED min_child_weight=1..5!
    for label, Xtr_in, Xva_in, depth, mcw in [
        ("XGB_raw_d2_mcw2", Xa.iloc[tr_idx], Xa.iloc[va_idx], 2, 2.0),
        ("XGB_fe_te_d2_mcw2", Xtr_d, Xva_d, 2, 2.0),
        ("XGB_fe_te_d3_mcw3", Xtr_d, Xva_d, 3, 3.0),
    ]:
        t0 = time.time()
        m = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="auc", tree_method="hist", enable_categorical=True,
            max_depth=depth, learning_rate=0.05, n_estimators=6000, subsample=0.8, colsample_bytree=0.25,
            min_child_weight=mcw, reg_lambda=30.0, reg_alpha=2.0, max_bin=1024,
            random_state=SEED, n_jobs=-1, early_stopping_rounds=150,
        )
        m.fit(Xtr_in, y_tr, eval_set=[(Xva_in, y_va)], verbose=False)
        p = m.predict_proba(Xva_in)[:, 1]
        print(f"  {label:<46} AUC={roc_auc_score(y_va, p):.6f} (iter={m.best_iteration}, {time.time()-t0:.1f}s)")
        if label == "XGB_fe_te_d2_mcw2":
            p_xgb = p

    # Fast CatBoost (drop high-card string pairs since we already have their OOF TE, keep 5 base cats with one_hot_max_size=20!)
    cb_drop = [f"{a}__x__{b}" for a, b in INTERACTION_PAIRS]
    Xtr_cb = Xtr_d.drop(columns=cb_drop).copy()
    Xva_cb = Xva_d.drop(columns=cb_drop).copy()
    for c in CAT_COLS:
        Xtr_cb[c] = Xtr_cb[c].astype(str)
        Xva_cb[c] = Xva_cb[c].astype(str)
    t0 = time.time()
    cb = CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC", iterations=4000, learning_rate=0.05,
        depth=4, l2_leaf_reg=25.0, one_hot_max_size=20, border_count=254,
        random_seed=SEED, thread_count=-1, verbose=0,
    )
    cb.fit(Xtr_cb, y_tr, cat_features=CAT_COLS, eval_set=(Xva_cb, y_va), early_stopping_rounds=150)
    p_cb = cb.predict_proba(Xva_cb)[:, 1]
    print(f"  {'CB_fe_te_d4 (fast OH)':<46} AUC={roc_auc_score(y_va, p_cb):.6f} (iter={cb.get_best_iteration()}, {time.time()-t0:.1f}s)")

    # Blend check
    def lg(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return logit(p)

    blend = 0.35 * lg(p_fe_te) + 0.35 * lg(p_xgb) + 0.30 * lg(p_cb)
    print(f"\n  >>> Fold 1 Log-Odds Blend (LGB+XGB+CB)         AUC={roc_auc_score(y_va, blend):.6f}")


if __name__ == "__main__":
    main()
