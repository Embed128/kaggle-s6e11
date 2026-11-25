"""Multi-model 5-Fold (or N-Fold) trainer on FE v3 (146 clean features).
Supports MacBook Air (CPU) and 5060 Ti desktop (CUDA GPU) via --device cpu|cuda.

Example usage:
  uv run python src/train_v2.py --models lr_gam,lgb_d1,lgb_d2
  uv run python src/train_v2.py --models xgb_d1,xgb_d2,cb_d2
  uv run python src/train_v2.py --models lr_gam,lgb_d1,lgb_d2,xgb_d1,xgb_d2,cb_d2 --device cuda --n-folds 10
"""

import argparse
import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config import ID_COL, N_FOLDS, OOF_DIR, SEED, SUBMISSION_DIR, TARGET_COL, ensure_dirs
from features import CAT_COLS, add_oof_target_encoding, build_base_feature_frames

ALL_MODELS = ["lr_gam", "lgb_d1", "lgb_d2", "xgb_d1", "xgb_d2", "cb_d2"]


def fit_predict_model(
    model_name: str,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    X_te: pd.DataFrame,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    t0 = time.time()

    if model_name == "lr_gam":
        num_cols = [c for c in X_tr.columns if c not in CAT_COLS]
        pre = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ])
        pipe = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(C=0.05, max_iter=500)),
        ])
        pipe.fit(X_tr, y_tr)
        va_pred = pipe.predict_proba(X_va)[:, 1]
        te_pred = pipe.predict_proba(X_te)[:, 1]
        return va_pred, te_pred, f"elapsed={time.time()-t0:.1f}s"

    if model_name == "lgb_d1":
        m = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            max_depth=1,
            num_leaves=2,
            learning_rate=0.05,
            n_estimators=8000,
            colsample_bytree=0.5,
            subsample=0.8,
            subsample_freq=1,
            min_child_samples=200,
            reg_lambda=25.0,
            reg_alpha=1.0,
            max_bin=1023,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        va_pred = m.predict_proba(X_va)[:, 1]
        te_pred = m.predict_proba(X_te)[:, 1]
        return va_pred, te_pred, f"iter={m.best_iteration_}, elapsed={time.time()-t0:.1f}s"

    if model_name == "lgb_d2":
        m = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            max_depth=2,
            num_leaves=4,
            learning_rate=0.03,
            n_estimators=10000,
            colsample_bytree=0.25,
            subsample=0.8,
            subsample_freq=1,
            min_child_samples=200,
            reg_lambda=35.0,
            reg_alpha=2.0,
            max_bin=1023,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        va_pred = m.predict_proba(X_va)[:, 1]
        te_pred = m.predict_proba(X_te)[:, 1]
        return va_pred, te_pred, f"iter={m.best_iteration_}, elapsed={time.time()-t0:.1f}s"

    if model_name in ("xgb_d1", "xgb_d2"):
        depth = 1 if model_name == "xgb_d1" else 2
        col = 0.5 if depth == 1 else 0.25
        m = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda" if device == "cuda" else "cpu",
            enable_categorical=True,
            max_depth=depth,
            learning_rate=0.04,
            n_estimators=8000,
            subsample=0.8,
            colsample_bytree=col,
            min_child_weight=2.0,
            reg_lambda=30.0,
            reg_alpha=2.0,
            max_bin=1024,
            random_state=seed,
            n_jobs=-1,
            early_stopping_rounds=150,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        va_pred = m.predict_proba(X_va)[:, 1]
        te_pred = m.predict_proba(X_te)[:, 1]
        return va_pred, te_pred, f"iter={m.best_iteration}, elapsed={time.time()-t0:.1f}s"

    if model_name == "cb_d2":
        X_tr_cb = X_tr.copy()
        X_va_cb = X_va.copy()
        X_te_cb = X_te.copy()
        for c in CAT_COLS:
            X_tr_cb[c] = X_tr_cb[c].astype(str)
            X_va_cb[c] = X_va_cb[c].astype(str)
            X_te_cb[c] = X_te_cb[c].astype(str)

        cb_kwargs = dict(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=4500,
            learning_rate=0.05,
            depth=2,
            l2_leaf_reg=25.0,
            one_hot_max_size=20,
            border_count=254,
            random_seed=seed,
            verbose=0,
        )
        if device == "cuda":
            cb_kwargs["task_type"] = "GPU"
        else:
            cb_kwargs["thread_count"] = -1

        m = CatBoostClassifier(**cb_kwargs)
        m.fit(
            X_tr_cb, y_tr,
            cat_features=CAT_COLS,
            eval_set=(X_va_cb, y_va),
            early_stopping_rounds=200,
        )
        va_pred = m.predict_proba(X_va_cb)[:, 1]
        te_pred = m.predict_proba(X_te_cb)[:, 1]
        return va_pred, te_pred, f"iter={m.get_best_iteration()}, elapsed={time.time()-t0:.1f}s"

    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="lr_gam,lgb_d1,lgb_d2", help="Comma-separated model names")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--smooth", type=float, default=15.0)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in ALL_MODELS:
            raise ValueError(f"Unknown model '{m}'. Choose from {ALL_MODELS}")

    ensure_dirs()
    t_start = time.time()
    print(f"Building FE v3 base tables for models={models} (device={args.device}, folds={args.n_folds})...")
    X_train_base, X_test_base, y, test_ids, meta = build_base_feature_frames()
    print(
        f"Base tables ready in {time.time()-t_start:.1f}s | "
        f"train={X_train_base.shape}, test={X_test_base.shape}, te_cols={len(meta['te_cols'])}"
    )

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    oof_dict = {m: np.zeros(len(X_train_base), dtype=np.float64) for m in models}
    test_dict = {m: np.zeros(len(X_test_base), dtype=np.float64) for m in models}
    fold_aucs = {m: [] for m in models}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_base, y), start=1):
        ft0 = time.time()
        print(f"\n--- Fold {fold}/{args.n_folds} ---")
        X_tr, X_va, X_te = add_oof_target_encoding(
            X_train_base.iloc[tr_idx],
            y[tr_idx],
            X_train_base.iloc[va_idx],
            X_test_base,
            te_cols=meta["te_cols"],
            keep_base_cols=meta["model_base_cols"],
            smooth=args.smooth,
            n_inner_splits=5,
            seed=SEED + fold,
        )
        print(f"  Inner-OOF TE ready ({X_tr.shape[1]} feats) in {time.time()-ft0:.1f}s")

        for m in models:
            va_p, te_p, info = fit_predict_model(
                m, X_tr, y[tr_idx], X_va, y[va_idx], X_te, device=args.device, seed=SEED + fold
            )
            oof_dict[m][va_idx] = va_p
            test_dict[m] += te_p / args.n_folds
            f_auc = roc_auc_score(y[va_idx], va_p)
            fold_aucs[m].append(f_auc)
            print(f"  [{m:<8}] Fold {fold} AUC = {f_auc:.6f} ({info})")

        del X_tr, X_va, X_te
        gc.collect()

    print("\n========================================================")
    tag = f"f{args.n_folds}" if args.n_folds != 5 else "v2"
    for m in models:
        oof_auc = roc_auc_score(y, oof_dict[m])
        mean_auc = float(np.mean(fold_aucs[m]))
        std_auc = float(np.std(fold_aucs[m]))
        print(f"  {m:<10} | Mean={mean_auc:.6f} ± {std_auc:.6f} | Overall OOF={oof_auc:.6f}")

        np.save(OOF_DIR / f"oof_{tag}_{m}.npy", oof_dict[m])
        np.save(OOF_DIR / f"test_{tag}_{m}.npy", test_dict[m])

        # Also save single-model submission for reference
        sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_dict[m]})
        sub.to_csv(SUBMISSION_DIR / f"sub_{tag}_{m}.csv", index=False)

    print(f"Total elapsed: {time.time()-t_start:.1f}s")
    print("========================================================")


if __name__ == "__main__":
    main()
