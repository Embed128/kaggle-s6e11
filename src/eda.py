"""Fast, low-memory EDA using Polars for Kaggle S6E11 (Credit Card Fraud)."""

import polars as pl

from config import DATA_RAW_DIR, ID_COL, TARGET_COL


def main() -> None:
    train_path = DATA_RAW_DIR / "train.csv"
    test_path = DATA_RAW_DIR / "test.csv"
    sub_path = DATA_RAW_DIR / "sample_submission.csv"

    print("Loading datasets with Polars...")
    train = pl.read_csv(train_path)
    test = pl.read_csv(test_path)
    sub = pl.read_csv(sub_path)

    print("\n=== 1. Shapes ===")
    print(f"train: {train.shape}")
    print(f"test : {test.shape}")
    print(f"sub  : {sub.shape}")
    print(f"sub columns: {sub.columns}")
    print("sub head(3):")
    print(sub.head(3))

    # Detect target column if different from default
    target_col = TARGET_COL
    if target_col not in train.columns:
        diff = [c for c in train.columns if c not in test.columns]
        print(f"[WARN] '{TARGET_COL}' not found; inferred target column(s): {diff}")
        target_col = diff[0] if diff else None

    if target_col:
        print(f"\n=== 2. Target Distribution ({target_col}) ===")
        vc = train.group_by(target_col).len().sort(target_col)
        total = train.height
        for row in vc.iter_rows():
            val, cnt = row
            print(f"  {val}: {cnt:,} ({cnt / total * 100:.4f}%)")

    print("\n=== 3. Column Summary (dtype, nulls, n_unique, overlap) ===")
    feature_cols = [c for c in test.columns if c != ID_COL]
    header = (
        f"{'Column':<28} {'Dtype':<10} {'TrNull%':>8} {'TeNull%':>8} "
        f"{'TrNUniq':>9} {'TeNUniq':>9} {'UnseenTe':>9}"
    )
    print(header)
    print("-" * len(header))

    for col in feature_cols:
        dtype = str(train[col].dtype)
        tr_null = train[col].null_count() / train.height * 100
        te_null = test[col].null_count() / test.height * 100
        tr_nuniq = train[col].n_unique()
        te_nuniq = test[col].n_unique()

        if train[col].dtype == pl.String or tr_nuniq <= 1000:
            tr_vals = set(train[col].drop_nulls().unique().to_list())
            te_vals = set(test[col].drop_nulls().unique().to_list())
            unseen = len(te_vals - tr_vals)
            unseen_str = str(unseen)
        else:
            unseen_str = "-"

        print(
            f"{col:<28} {dtype:<10} {tr_null:>7.2f}% {te_null:>7.2f}% "
            f"{tr_nuniq:>9,} {te_nuniq:>9,} {unseen_str:>9}"
        )

    print("\n=== 4. Sample Rows (train head 5) ===")
    with pl.Config(tbl_cols=30, tbl_width_chars=200):
        print(train.head(5))

    print("\n=== 5. Numeric Summary (train) ===")
    num_cols = [c for c in feature_cols if train[c].dtype.is_numeric()]
    if num_cols:
        with pl.Config(tbl_cols=30, tbl_width_chars=200):
            print(train.select(num_cols).describe())

    print("\n=== 6. Categorical Top Values (train) ===")
    cat_cols = [c for c in feature_cols if not train[c].dtype.is_numeric()]
    for col in cat_cols:
        top = train.group_by(col).len().sort("len", descending=True).head(5)
        items = ", ".join(f"{r[0]}:{r[1]:,}" for r in top.iter_rows())
        print(f"  {col} (n_unique={train[col].n_unique():,}): {items}")

    if ID_COL in train.columns and ID_COL in test.columns:
        print("\n=== 7. ID Ranges ===")
        print(f"  train id: min={train[ID_COL].min()}, max={train[ID_COL].max()}")
        print(f"  test  id: min={test[ID_COL].min()}, max={test[ID_COL].max()}")

    print("\n=== 8. Exact Duplicate Feature Rows ===")
    dup_tr = train.height - train.select(feature_cols).n_unique()
    print(f"  Duplicate feature rows in train: {dup_tr:,}")


if __name__ == "__main__":
    main()
