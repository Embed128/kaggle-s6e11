"""5-Fold StratifiedKFold LightGBM Baseline for Kaggle S6E11 (Credit Card Fraud)."""

import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import (
    DATA_RAW_DIR,
    ID_COL,
    N_FOLDS,
    OOF_DIR,
    SEED,
    SUBMISSION_DIR,
    TARGET_COL,
    ensure_dirs,
)


def load_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.Series, list[str], str]:
    """Load train/test with Polars and convert to compact float32/category pandas DataFrames."""
    train_pl = pl.read_csv(DATA_RAW_DIR / "train.csv")
    test_pl = pl.read_csv(DATA_RAW_DIR / "test.csv")

    target_col = TARGET_COL
    if target_col not in train_pl.columns:
        diff = [c for c in train_pl.columns if c not in test_pl.columns]
        if not diff:
            raise ValueError("Could not identify target column in train.csv")
        target_col = diff[0]
        print(f"[INFO] Detected target column: {target_col}")

    feature_cols = [c for c in test_pl.columns if c != ID_COL]
    cat_cols = [c for c in feature_cols if train_pl[c].dtype == pl.String]
    bool_cols = [c for c in feature_cols if train_pl[c].dtype == pl.Boolean]

    y = train_pl[target_col].to_numpy().astype(np.int8)
    test_ids = test_pl[ID_COL].to_pandas()

    # Downcast numeric to float32 and booleans to int8 to keep RAM footprint low
    cast_exprs = [
        pl.col(c).cast(pl.Float32) for c in feature_cols if train_pl[c].dtype.is_numeric()
    ] + [pl.col(c).cast(pl.Int8) for c in bool_cols]
    X_train = train_pl.select(feature_cols).with_columns(cast_exprs).to_pandas()
    X_test = test_pl.select(feature_cols).with_columns(cast_exprs).to_pandas()

    del train_pl, test_pl
    gc.collect()

    # Align categorical dtypes across train and test consistently
    for col in cat_cols:
        cats = pd.Index(X_train[col].dropna().unique()).union(
            pd.Index(X_test[col].dropna().unique())
        )
        cat_type = pd.CategoricalDtype(categories=cats, ordered=False)
        X_train[col] = X_train[col].astype(cat_type)
        X_test[col] = X_test[col].astype(cat_type)

    return X_train, y, X_test, test_ids, cat_cols, target_col


def main() -> None:
    ensure_dirs()
    t0 = time.time()

    print("Loading data...")
    X_train, y, X_test, test_ids, cat_cols, target_col = load_data()
    print(
        f"Loaded X_train={X_train.shape}, X_test={X_test.shape}, "
        f"pos_rate={y.mean():.5f}, cat_cols={len(cat_cols)} ({time.time() - t0:.1f}s)"
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X_train), dtype=np.float64)
    test_preds = np.zeros(len(X_test), dtype=np.float64)
    feature_importance = np.zeros(X_train.shape[1], dtype=np.float64)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "n_estimators": 3000,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), start=1):
        ft0 = time.time()
        X_tr, y_tr = X_train.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            categorical_feature=cat_cols if cat_cols else "auto",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=500),
            ],
        )

        va_pred = model.predict_proba(X_va)[:, 1]
        oof_preds[va_idx] = va_pred
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
        feature_importance += model.booster_.feature_importance(importance_type="gain") / N_FOLDS

        fold_auc = roc_auc_score(y_va, va_pred)
        fold_scores.append(fold_auc)
        print(
            f"[Fold {fold}/{N_FOLDS}] ROC AUC = {fold_auc:.6f} | "
            f"best_iter = {model.best_iteration_} | elapsed = {time.time() - ft0:.1f}s"
        )

        del X_tr, y_tr, X_va, y_va, model
        gc.collect()

    oof_auc = roc_auc_score(y, oof_preds)
    print("\n==========================================")
    print(f"5-Fold Mean AUC : {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    print(f"Overall OOF AUC : {oof_auc:.6f}")
    print(f"Total elapsed   : {time.time() - t0:.1f}s")
    print("==========================================")

    # Top 15 features by gain
    imp_df = pd.DataFrame(
        {"feature": X_train.columns, "gain": feature_importance}
    ).sort_values("gain", ascending=False)
    print("\nTop 15 Features (Gain):")
    for row in imp_df.head(15).itertuples(index=False):
        print(f"  {row.feature:<28} {row.gain:>14,.1f}")

    # Save OOF & test predictions for future ensembling
    np.save(OOF_DIR / "oof_lgb_baseline.npy", oof_preds)
    np.save(OOF_DIR / "test_lgb_baseline.npy", test_preds)

    # Save and verify submission
    sub = pd.DataFrame({ID_COL: test_ids, target_col: test_preds})
    assert len(sub) == len(X_test), "Submission row count mismatch!"
    assert not sub[target_col].isna().any(), "Submission contains NaNs!"
    assert (sub[target_col] >= 0).all() and (sub[target_col] <= 1).all(), "Predictions out of [0, 1]!"

    sub_path = SUBMISSION_DIR / "sub_lgb_baseline.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved submission -> {sub_path}")
    print(
        f"Prediction stats: min={test_preds.min():.6f}, "
        f"mean={test_preds.mean():.6f}, max={test_preds.max():.6f}"
    )


if __name__ == "__main__":
    main()
