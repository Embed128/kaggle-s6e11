"""Residual Hunter: find the strongest missed signals on top of V2 lr_gam (OOF 0.950413)."""

import itertools
import numpy as np
import pandas as pd
import polars as pl
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

from config import DATA_RAW_DIR, ID_COL, N_FOLDS, OOF_DIR, SEED, TARGET_COL
from features import BOOL_COLS, CAT_COLS, NUM_COLS


def main() -> None:
    train_pl = pl.read_csv(DATA_RAW_DIR / "train.csv")
    test_pl = pl.read_csv(DATA_RAW_DIR / "test.csv")
    y = train_pl[TARGET_COL].to_numpy().astype(np.int8)
    oof_lr = np.load(OOF_DIR / "oof_v2_lr_gam.npy")
    oof_xgb = np.load(OOF_DIR / "oof_v2_xgb_d2.npy")
    base_logit = logit(np.clip(oof_lr, 1e-6, 1 - 1e-6))
    resid = y - oof_lr

    print(f"Base lr_gam OOF AUC: {roc_auc_score(y, oof_lr):.6f}")
    print(f"Residual stats: mean={resid.mean():.6f}, std={resid.std():.6f}")

    # 1. Check Adversarial Validation (train vs test distribution shift)
    raw_cols = [c for c in test_pl.columns if c != ID_COL]
    df_all = pl.concat([train_pl.select(raw_cols), test_pl.select(raw_cols)], how="vertical").with_columns(
        [pl.col(c).cast(pl.Float32) for c in NUM_COLS] + [pl.col(c).cast(pl.Int8) for c in BOOL_COLS]
    ).to_pandas()
    for c in CAT_COLS:
        df_all[c] = df_all[c].astype("category")
    y_adv = np.concatenate([np.zeros(train_pl.height, dtype=np.int8), np.ones(test_pl.height, dtype=np.int8)])
    adv_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    tr_a, va_a = next(adv_skf.split(df_all, y_adv))
    m_adv = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, verbose=-1, n_jobs=-1, random_state=SEED)
    m_adv.fit(df_all.iloc[tr_a], y_adv[tr_a], categorical_feature=CAT_COLS)
    adv_auc = roc_auc_score(y_adv[va_a], m_adv.predict_proba(df_all.iloc[va_a])[:, 1])
    print(f"\n=== 1. Adversarial Validation (Train vs Test) AUC = {adv_auc:.5f} ===")

    # 2. Check where residuals concentrate (False Negatives vs False Positives)
    X_tr = df_all.iloc[:train_pl.height].copy()
    print("\n=== 2. Residual Correlation with Raw Features & Quadratic Terms ===")
    corrs = []
    for c in NUM_COLS + BOOL_COLS:
        v = X_tr[c].to_numpy(dtype=np.float64)
        r1 = np.corrcoef(v, resid)[0, 1]
        r2 = np.corrcoef((v - v.mean()) ** 2, resid)[0, 1]
        corrs.append((c, r1, r2))
    for c, r1, r2 in sorted(corrs, key=lambda x: -max(abs(x[1]), abs(x[2])))[:10]:
        print(f"  {c:<30} corr(x,r)={r1:+.5f}  corr((x-m)^2,r)={r2:+.5f}")

    # 3. Scan ALL 276 2-way pairs of binned/discrete raw features for residual predictive power!
    print("\n=== 3. Scanning ALL 276 2-Way Pairs for Uncaptured Residual Signal ===")
    # Discretize continuous columns into ~15 quantile bins for clean interaction scan
    disc = {}
    for c in raw_cols:
        if c in CAT_COLS or c in BOOL_COLS or train_pl[c].n_unique() <= 25:
            disc[c] = train_pl[c].cast(pl.String).to_numpy()
        else:
            vals = train_pl[c].to_numpy()
            q = pd.qcut(vals, q=15, labels=False, duplicates="drop")
            disc[c] = q.astype(str)

    # Fold-1 test of adding smoothed residual mean of each pair to base_logit
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(skf.split(X_tr, y))
    base_va_auc = roc_auc_score(y[va_idx], base_logit[va_idx])
    print(f"  Fold-1 base lr_gam AUC = {base_va_auc:.6f}")

    pair_gains = []
    y_tr_r = resid[tr_idx]
    for a, b in itertools.combinations(raw_cols, 2):
        key_tr = pd.Series(disc[a][tr_idx]) + "_" + pd.Series(disc[b][tr_idx])
        key_va = pd.Series(disc[a][va_idx]) + "_" + pd.Series(disc[b][va_idx])
        stats = pd.DataFrame({"k": key_tr, "r": y_tr_r}).groupby("k")["r"].agg(["count", "sum"])
        enc = stats["sum"] / (stats["count"] + 50.0)
        va_adj = key_va.map(enc).fillna(0.0).to_numpy()
        new_auc = roc_auc_score(y[va_idx], base_logit[va_idx] + 3.0 * va_adj)
        gain = new_auc - base_va_auc
        if gain > 0.00003:
            pair_gains.append((f"{a} x {b}", gain, new_auc))

    pair_gains.sort(key=lambda x: -x[1])
    print(f"  Pairs with positive Fold-1 residual gain > 3e-5: {len(pair_gains)}")
    for name, g, auc in pair_gains[:20]:
        print(f"    {name:<55} gain={g:+.6f} -> {auc:.6f}")

    # 4. Check if ANY single raw column's EXACT values STILL have residual signal not captured by smooth=15!
    print("\n=== 4. Exact-Value Residual Check (Testing Lower Smoothing smooth=3 on Exact Values) ===")
    for c in NUM_COLS:
        tr_v = train_pl[c].to_numpy()[tr_idx]
        va_v = train_pl[c].to_numpy()[va_idx]
        stats = pd.DataFrame({"k": tr_v, "r": y_tr_r}).groupby("k")["r"].agg(["count", "sum"])
        enc = stats["sum"] / (stats["count"] + 5.0)
        va_adj = pd.Series(va_v).map(enc).fillna(0.0).to_numpy()
        gain = roc_auc_score(y[va_idx], base_logit[va_idx] + 2.0 * va_adj) - base_va_auc
        print(f"  exact {c:<28} residual gain = {gain:+.6f}")

    # 5. Check exact value overlap between train and test for high-card numeric cols
    print("\n=== 5. Exact Value Overlap (Test rows whose exact value appears in Train) ===")
    for c in ["amount_usd", "account_balance_usd", "hours_since_last_txn", "distance_from_home_km", "merchant_risk_score", "velocity_score"]:
        tr_set = set(train_pl[c].unique().to_list())
        te_vals = test_pl[c].to_list()
        overlap = sum(v in tr_set for v in te_vals) / len(te_vals) * 100
        print(f"  {c:<28} test exact-value seen in train: {overlap:.2f}%")


if __name__ == "__main__":
    main()
