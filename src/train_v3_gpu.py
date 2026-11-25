"""V3 10-Fold GPU/CPU Multi-Model Suite for Kaggle S6E11 (RTX 5060 Ti 16GB VRAM + 16GB RAM safe).

Features:
- Strictly leak-free 10-Fold StratifiedKFold (+12.5% exact-value support per fold for inner-OOF TE!)
- Resumable per-model checkpointing: skips models whose oof_v3_<model>.npy already exists (use --force to rerun)
- Explicit float32 + gc.collect() memory hygiene for 16GB RAM desktops

Models:
  lr_gam     : Multi-C L2 Logistic GAM on 146 FE v3 cols
  super_gam  : Multi-smooth (5/40) OOF TE + Cubic Splines + Clean OneHot + 35 Top Pairs
  lgb_d1     : LightGBM depth=1 (pure additive stumps)
  lgb_d2     : LightGBM depth=2 (2-way interactions, max_bin=1023)
  lgb_d3_et  : LightGBM depth=3 with extra_trees=True (randomized thresholds for diversity)
  xgb_d2_gpu : XGBoost CUDA/CPU depth=2 (max_bin=2048, min_child_weight=2)
  xgb_d3_gpu : XGBoost CUDA/CPU depth=3 (max_bin=2048, min_child_weight=3, strong L2)
  cb_d3_gpu  : CatBoost GPU/CPU depth=3 (oblivious trees)
  cb_d4_gpu  : CatBoost GPU/CPU depth=4 (oblivious trees, higher interaction capacity)
"""

import argparse
import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from config import ID_COL, OOF_DIR, SEED, SUBMISSION_DIR, TARGET_COL, ensure_dirs
from features import BOOL_COLS, CAT_COLS, NUM_COLS, add_oof_target_encoding, build_base_feature_frames
from train_super_gam import CLEAN_OHE_COLS, build_super_gam_base, compute_multi_smooth_te

FE_V3_MODELS = [
    "lr_gam",
    "lgb_d1",
    "lgb_d2",
    "lgb_d3_et",
    "xgb_d2_gpu",
    "xgb_d3_gpu",
    "cb_d3_gpu",
    "cb_d4_gpu",
]
ALL_V3_MODELS = FE_V3_MODELS + ["super_gam"]


def detect_device(requested: str) -> str:
    if requested in ("cpu", "cuda"):
        return requested
    # auto-detect CUDA via xgboost / nvidia-smi
    import shutil
    if shutil.which("nvidia-smi"):
        return "cuda"
    return "cpu"


def fit_predict_fe_model(
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
        X_tr_m = pre.fit_transform(X_tr)
        X_va_m = pre.transform(X_va)
        X_te_m = pre.transform(X_te)
        # Blend C=0.05 and C=0.25 in probability space (zero extra feature cost, tighter calibration)
        va_p = np.zeros(len(y_va), dtype=np.float64)
        te_p = np.zeros(len(X_te), dtype=np.float64)
        for c in (0.05, 0.25):
            clf = LogisticRegression(C=c, max_iter=600)
            clf.fit(X_tr_m, y_tr)
            va_p += 0.5 * clf.predict_proba(X_va_m)[:, 1]
            te_p += 0.5 * clf.predict_proba(X_te_m)[:, 1]
        return va_p, te_p, f"elapsed={time.time()-t0:.1f}s"

    if model_name == "lgb_d1":
        m = lgb.LGBMClassifier(
            objective="binary", metric="auc", max_depth=1, num_leaves=2,
            learning_rate=0.04, n_estimators=10000, colsample_bytree=0.5,
            subsample=0.8, subsample_freq=1, min_child_samples=200,
            reg_lambda=25.0, reg_alpha=1.0, max_bin=1023,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        m.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)], categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1], f"iter={m.best_iteration_}, {time.time()-t0:.1f}s"

    if model_name == "lgb_d2":
        m = lgb.LGBMClassifier(
            objective="binary", metric="auc", max_depth=2, num_leaves=4,
            learning_rate=0.025, n_estimators=12000, colsample_bytree=0.25,
            subsample=0.8, subsample_freq=1, min_child_samples=200,
            reg_lambda=35.0, reg_alpha=2.0, max_bin=1023,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        m.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)], categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(250, verbose=False)],
        )
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1], f"iter={m.best_iteration_}, {time.time()-t0:.1f}s"

    if model_name == "lgb_d3_et":
        m = lgb.LGBMClassifier(
            objective="binary", metric="auc", max_depth=3, num_leaves=7,
            extra_trees=True, learning_rate=0.025, n_estimators=12000,
            colsample_bytree=0.25, subsample=0.8, subsample_freq=1,
            min_child_samples=250, reg_lambda=50.0, reg_alpha=3.0, max_bin=1023,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        m.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)], categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(250, verbose=False)],
        )
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1], f"iter={m.best_iteration_}, {time.time()-t0:.1f}s"

    if model_name in ("xgb_d2_gpu", "xgb_d3_gpu"):
        depth = 2 if model_name == "xgb_d2_gpu" else 3
        lr = 0.025 if depth == 2 else 0.02
        mcw = 2.0 if depth == 2 else 3.0
        reg_l2 = 35.0 if depth == 2 else 60.0
        col = 0.25 if depth == 2 else 0.20
        m = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda" if device == "cuda" else "cpu",
            enable_categorical=True,
            max_depth=depth,
            learning_rate=lr,
            n_estimators=12000,
            subsample=0.8,
            colsample_bytree=col,
            min_child_weight=mcw,
            reg_lambda=reg_l2,
            reg_alpha=2.0,
            max_bin=2048 if device == "cuda" else 1024,
            random_state=seed,
            n_jobs=-1,
            early_stopping_rounds=250,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1], f"iter={m.best_iteration}, {time.time()-t0:.1f}s"

    if model_name in ("cb_d3_gpu", "cb_d4_gpu"):
        depth = 3 if model_name == "cb_d3_gpu" else 4
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
            iterations=8000 if device == "cuda" else 4500,
            learning_rate=0.03 if device == "cuda" else 0.05,
            depth=depth,
            l2_leaf_reg=30.0 if depth == 3 else 45.0,
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
        m.fit(X_tr_cb, y_tr, cat_features=CAT_COLS, eval_set=(X_va_cb, y_va), early_stopping_rounds=250)
        va_p = m.predict_proba(X_va_cb)[:, 1]
        te_p = m.predict_proba(X_te_cb)[:, 1]
        del X_tr_cb, X_va_cb, X_te_cb
        gc.collect()
        return va_p, te_p, f"iter={m.get_best_iteration()}, {time.time()-t0:.1f}s"

    raise ValueError(f"Unknown model: {model_name}")


def run_super_gam_v3(n_folds: int, tag: str) -> None:
    print(f"\n================ Running {n_folds}-Fold super_gam ================")
    t0 = time.time()
    X_tr_base, X_te_base, y, test_ids, te_lc, te_hc = build_super_gam_base()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    oof_lr = np.zeros(len(y), dtype=np.float64)
    test_lr = np.zeros(len(test_ids), dtype=np.float64)
    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_base, y), start=1):
        ft0 = time.time()
        te_tr, te_va, te_te = compute_multi_smooth_te(
            X_tr_base.iloc[tr_idx], y[tr_idx], X_tr_base.iloc[va_idx], X_te_base, te_lc, te_hc, seed=SEED + fold
        )
        pre = ColumnTransformer([
            ("num", Pipeline([("sc", StandardScaler()), ("sp", SplineTransformer(n_knots=8, degree=3, include_bias=False))]), NUM_COLS),
            ("bool", "passthrough", BOOL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CLEAN_OHE_COLS),
        ])
        base_tr = pre.fit_transform(X_tr_base.iloc[tr_idx]).astype(np.float32)
        base_va = pre.transform(X_tr_base.iloc[va_idx]).astype(np.float32)
        base_te = pre.transform(X_te_base).astype(np.float32)

        te_sc = StandardScaler()
        F_tr = np.hstack([te_sc.fit_transform(te_tr).astype(np.float32), base_tr])
        F_va = np.hstack([te_sc.transform(te_va).astype(np.float32), base_va])
        F_te = np.hstack([te_sc.transform(te_te).astype(np.float32), base_te])

        va_p = np.zeros(len(va_idx), dtype=np.float64)
        te_p = np.zeros(len(test_ids), dtype=np.float64)
        for c in (0.015, 0.06):
            lr = LogisticRegression(C=c, solver="lbfgs", max_iter=600)
            lr.fit(F_tr, y[tr_idx])
            va_p += 0.5 * lr.predict_proba(F_va)[:, 1]
            te_p += 0.5 * lr.predict_proba(F_te)[:, 1]

        oof_lr[va_idx] = va_p
        test_lr += te_p / n_folds
        f_auc = roc_auc_score(y[va_idx], va_p)
        fold_aucs.append(f_auc)
        print(f"  [super_gam] Fold {fold}/{n_folds} AUC = {f_auc:.6f} ({time.time()-ft0:.1f}s)")

        del F_tr, F_va, F_te, base_tr, base_va, base_te, te_tr, te_va, te_te
        gc.collect()

    overall = roc_auc_score(y, oof_lr)
    print(f"  >>> super_gam | Mean={np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f} | Overall OOF={overall:.6f} ({time.time()-t0:.1f}s)")
    np.save(OOF_DIR / f"oof_{tag}_super_gam.npy", oof_lr)
    np.save(OOF_DIR / f"test_{tag}_super_gam.npy", test_lr)
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_lr}).to_csv(SUBMISSION_DIR / f"sub_{tag}_super_gam.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=",".join(ALL_V3_MODELS))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--smooth", type=float, default=15.0)
    parser.add_argument("--tag", type=str, default="v3")
    parser.add_argument("--force", action="store_true", help="Re-train even if OOF already exists")
    args = parser.parse_args()

    ensure_dirs()
    device = detect_device(args.device)
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in requested:
        if m not in ALL_V3_MODELS:
            raise ValueError(f"Unknown model '{m}'. Valid: {ALL_V3_MODELS}")

    # Filter already-completed models unless --force
    todo = []
    for m in requested:
        oof_f = OOF_DIR / f"oof_{args.tag}_{m}.npy"
        test_f = OOF_DIR / f"test_{args.tag}_{m}.npy"
        if not args.force and oof_f.exists() and test_f.exists():
            print(f"[SKIP] {m} already completed ({oof_f.name})")
        else:
            todo.append(m)

    if not todo:
        print("All requested models already completed!")
        return

    print(f"=== V3 Trainer | device={device} | n_folds={args.n_folds} | tag={args.tag} | models={todo} ===")

    fe_todo = [m for m in todo if m in FE_V3_MODELS]
    if fe_todo:
        t_start = time.time()
        X_train_base, X_test_base, y, test_ids, meta = build_base_feature_frames()
        print(f"FE v3 base ready in {time.time()-t_start:.1f}s | train={X_train_base.shape}, te_cols={len(meta['te_cols'])}")

        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
        oof_dict = {m: np.zeros(len(y), dtype=np.float64) for m in fe_todo}
        test_dict = {m: np.zeros(len(test_ids), dtype=np.float64) for m in fe_todo}
        fold_aucs = {m: [] for m in fe_todo}

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_base, y), start=1):
            ft0 = time.time()
            print(f"\n--- Fold {fold}/{args.n_folds} ---")
            X_tr, X_va, X_te = add_oof_target_encoding(
                X_train_base.iloc[tr_idx], y[tr_idx], X_train_base.iloc[va_idx], X_test_base,
                te_cols=meta["te_cols"], keep_base_cols=meta["model_base_cols"],
                smooth=args.smooth, n_inner_splits=5, seed=SEED + fold,
            )
            print(f"  Inner-OOF TE ready ({X_tr.shape[1]} feats) in {time.time()-ft0:.1f}s")

            for m in fe_todo:
                va_p, te_p, info = fit_predict_fe_model(
                    m, X_tr, y[tr_idx], X_va, y[va_idx], X_te, device=device, seed=SEED + fold
                )
                oof_dict[m][va_idx] = va_p
                test_dict[m] += te_p / args.n_folds
                f_auc = roc_auc_score(y[va_idx], va_p)
                fold_aucs[m].append(f_auc)
                print(f"  [{m:<11}] Fold {fold} AUC = {f_auc:.6f} ({info})")
                gc.collect()

            del X_tr, X_va, X_te
            gc.collect()

        print("\n================ FE v3 Models Summary ================")
        for m in fe_todo:
            oof_auc = roc_auc_score(y, oof_dict[m])
            print(f"  {m:<12} | Mean={np.mean(fold_aucs[m]):.6f} ± {np.std(fold_aucs[m]):.6f} | Overall OOF={oof_auc:.6f}")
            np.save(OOF_DIR / f"oof_{args.tag}_{m}.npy", oof_dict[m])
            np.save(OOF_DIR / f"test_{args.tag}_{m}.npy", test_dict[m])
            pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_dict[m]}).to_csv(
                SUBMISSION_DIR / f"sub_{args.tag}_{m}.csv", index=False
            )

        del X_train_base, X_test_base
        gc.collect()

    if "super_gam" in todo:
        run_super_gam_v3(n_folds=args.n_folds, tag=args.tag)


if __name__ == "__main__":
    main()
