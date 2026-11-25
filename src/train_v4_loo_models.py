"""V4 10-Fold LOO-Cluster-Enhanced Models (lr_gam_loo, xgb_d2_loo, lgb_d2_loo).
Combines FE v3 (146 clean cols) + 82 strictly leak-free 1M-row LOO Unsupervised Cluster Profile features
(verified +0.00024 single-model / +0.00063 blend on Fold 1!).
"""

import argparse
import gc
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import ID_COL, OOF_DIR, SEED, SUBMISSION_DIR, TARGET_COL, ensure_dirs
from features import CAT_COLS, add_oof_target_encoding, build_base_feature_frames
from train_realmlp_gpu import build_unsupervised_loo_cluster_features

MODELS = ["lr_gam_loo", "xgb_d2_loo", "lgb_d2_loo"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=",".join(MODELS))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--tag", type=str, default="v4")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = args.device if args.device in ("cpu", "cuda") else ("cuda" if shutil.which("nvidia-smi") else "cpu")
    req = [m.strip() for m in args.models.split(",") if m.strip()]
    todo = [
        m for m in req
        if args.force or not ((OOF_DIR / f"oof_{args.tag}_{m}.npy").exists() and (OOF_DIR / f"test_{args.tag}_{m}.npy").exists())
    ]
    if not todo:
        print("[SKIP] All V4 LOO models already completed!")
        return

    t0 = time.time()
    X_train_base, X_test_base, y, test_ids, meta = build_base_feature_frames()
    tr_loo, te_loo = build_unsupervised_loo_cluster_features()
    print(f"=== V4 LOO Models | device={device} | folds={args.n_folds} | models={todo} | loo_feats={tr_loo.shape[1]} ({time.time()-t0:.1f}s) ===")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    oof_dict = {m: np.zeros(len(y), dtype=np.float64) for m in todo}
    test_dict = {m: np.zeros(len(test_ids), dtype=np.float64) for m in todo}
    fold_aucs = {m: [] for m in todo}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_base, y), start=1):
        ft0 = time.time()
        X_tr, X_va, X_te = add_oof_target_encoding(
            X_train_base.iloc[tr_idx], y[tr_idx], X_train_base.iloc[va_idx], X_test_base,
            te_cols=meta["te_cols"], keep_base_cols=meta["model_base_cols"],
            smooth=15.0, n_inner_splits=5, seed=SEED + fold,
        )
        X_tr = pd.concat([X_tr, tr_loo.iloc[tr_idx].reset_index(drop=True)], axis=1)
        X_va = pd.concat([X_va, tr_loo.iloc[va_idx].reset_index(drop=True)], axis=1)
        X_te = pd.concat([X_te, te_loo.reset_index(drop=True)], axis=1)
        print(f"\n--- Fold {fold}/{args.n_folds} ({X_tr.shape[1]} feats, prep={time.time()-ft0:.1f}s) ---")

        if "lr_gam_loo" in todo:
            t1 = time.time()
            num_cols = [c for c in X_tr.columns if c not in CAT_COLS]
            pre = ColumnTransformer([
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ])
            Xtr_m = pre.fit_transform(X_tr)
            Xva_m = pre.transform(X_va)
            Xte_m = pre.transform(X_te)
            va_p = np.zeros(len(va_idx), dtype=np.float64)
            te_p = np.zeros(len(test_ids), dtype=np.float64)
            for c in (0.03, 0.15):
                clf = LogisticRegression(C=c, max_iter=600)
                clf.fit(Xtr_m, y[tr_idx])
                va_p += 0.5 * clf.predict_proba(Xva_m)[:, 1]
                te_p += 0.5 * clf.predict_proba(Xte_m)[:, 1]
            oof_dict["lr_gam_loo"][va_idx] = va_p
            test_dict["lr_gam_loo"] += te_p / args.n_folds
            fa = roc_auc_score(y[va_idx], va_p)
            fold_aucs["lr_gam_loo"].append(fa)
            print(f"  [lr_gam_loo] Fold {fold} AUC = {fa:.6f} ({time.time()-t1:.1f}s)")

        if "xgb_d2_loo" in todo:
            t1 = time.time()
            m = xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="auc", tree_method="hist",
                device="cuda" if device == "cuda" else "cpu", enable_categorical=True,
                max_depth=2, learning_rate=0.025, n_estimators=12000, subsample=0.8,
                colsample_bytree=0.22, min_child_weight=2.0, reg_lambda=40.0, reg_alpha=2.5,
                max_bin=2048 if device == "cuda" else 1024,
                random_state=SEED + 50 + fold, n_jobs=-1, early_stopping_rounds=250,
            )
            m.fit(X_tr, y[tr_idx], eval_set=[(X_va, y[va_idx])], verbose=False)
            va_p = m.predict_proba(X_va)[:, 1]
            te_p = m.predict_proba(X_te)[:, 1]
            oof_dict["xgb_d2_loo"][va_idx] = va_p
            test_dict["xgb_d2_loo"] += te_p / args.n_folds
            fa = roc_auc_score(y[va_idx], va_p)
            fold_aucs["xgb_d2_loo"].append(fa)
            print(f"  [xgb_d2_loo] Fold {fold} AUC = {fa:.6f} (iter={m.best_iteration}, {time.time()-t1:.1f}s)")

        if "lgb_d2_loo" in todo:
            t1 = time.time()
            m = lgb.LGBMClassifier(
                objective="binary", metric="auc", max_depth=2, num_leaves=4,
                learning_rate=0.025, n_estimators=12000, colsample_bytree=0.22,
                subsample=0.8, subsample_freq=1, min_child_samples=200,
                reg_lambda=40.0, reg_alpha=2.5, max_bin=1023,
                random_state=SEED + 50 + fold, n_jobs=-1, verbose=-1,
            )
            m.fit(
                X_tr, y[tr_idx], eval_set=[(X_va, y[va_idx])], categorical_feature=CAT_COLS,
                callbacks=[lgb.early_stopping(250, verbose=False)],
            )
            va_p = m.predict_proba(X_va)[:, 1]
            te_p = m.predict_proba(X_te)[:, 1]
            oof_dict["lgb_d2_loo"][va_idx] = va_p
            test_dict["lgb_d2_loo"] += te_p / args.n_folds
            fa = roc_auc_score(y[va_idx], va_p)
            fold_aucs["lgb_d2_loo"].append(fa)
            print(f"  [lgb_d2_loo] Fold {fold} AUC = {fa:.6f} (iter={m.best_iteration_}, {time.time()-t1:.1f}s)")

        del X_tr, X_va, X_te
        gc.collect()

    print("\n================ V4 LOO Models Summary ================")
    for m in todo:
        ov = roc_auc_score(y, oof_dict[m])
        print(f"  {m:<12} | Mean={np.mean(fold_aucs[m]):.6f} ± {np.std(fold_aucs[m]):.6f} | Overall OOF={ov:.6f}")
        np.save(OOF_DIR / f"oof_{args.tag}_{m}.npy", oof_dict[m])
        np.save(OOF_DIR / f"test_{args.tag}_{m}.npy", test_dict[m])
        pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_dict[m]}).to_csv(
            SUBMISSION_DIR / f"sub_{args.tag}_{m}.csv", index=False
        )


if __name__ == "__main__":
    main()
