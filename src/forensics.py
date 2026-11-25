"""Data Forensics for Kaggle S6E11: hunt for hidden structural signals explaining the gap to top LB."""

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

from config import DATA_RAW_DIR, ID_COL, OOF_DIR, TARGET_COL
from features import BOOL_COLS, CAT_COLS, NUM_COLS


def main() -> None:
    train = pl.read_csv(DATA_RAW_DIR / "train.csv")
    test = pl.read_csv(DATA_RAW_DIR / "test.csv")
    y = train[TARGET_COL].to_numpy().astype(np.int8)

    print("=== 1. Check Row-Order / ID Autocorrelation (lag 1, 2, 3, 5, 10) ===")
    for lag in [1, 2, 3, 5, 10]:
        corr_y = np.corrcoef(y[:-lag], y[lag:])[0, 1]
        corr_amt = np.corrcoef(train["amount_usd"].to_numpy()[:-lag], train["amount_usd"].to_numpy()[lag:])[0, 1]
        corr_bal = np.corrcoef(train["account_balance_usd"].to_numpy()[:-lag], train["account_balance_usd"].to_numpy()[lag:])[0, 1]
        print(f"  lag={lag:<2}: corr(y)={corr_y:+.5f}, corr(amount)={corr_amt:+.5f}, corr(balance)={corr_bal:+.5f}")

    print("\n=== 2. Check Latent Customer/Card Clustering (do rows with same balance/age share features or fraud?) ===")
    for key in ["account_balance_usd", "amount_usd", "hours_since_last_txn", "distance_from_home_km"]:
        g = train.group_by(key).agg(
            pl.len().alias("n"),
            pl.col(TARGET_COL).mean().alias("fraud_rate"),
            pl.col("customer_age").n_unique().alias("n_ages"),
            pl.col("merchant_risk_score").std().alias("mrs_std"),
        ).filter(pl.col("n") >= 10)
        print(
            f"  group by {key:<24}: groups(n>=10)={g.height:>6,}, "
            f"avg_n={g['n'].mean():.1f}, avg_n_ages={g['n_ages'].mean():.2f}, "
            f"fraud_rate_std={g['fraud_rate'].std():.4f}, max_fraud_rate={g['fraud_rate'].max():.3f}"
        )

    print("\n=== 3. Inspect WHY hours_since_last_txn_cents has 0.65%..2.97% fraud spread! ===")
    df_c = train.with_columns(
        ((pl.col("hours_since_last_txn") * 100.0).round(0).cast(pl.Int64) % 100).alias("h_cents"),
        ((pl.col("amount_usd") * 100.0).round(0).cast(pl.Int64) % 100).alias("a_cents"),
        ((pl.col("distance_from_home_km") * 100.0).round(0).cast(pl.Int64) % 100).alias("d_cents"),
    )
    for col in ["h_cents", "a_cents", "d_cents"]:
        g = df_c.group_by(col).agg(pl.len().alias("n"), pl.col(TARGET_COL).mean().alias("rate")).sort("rate", descending=True)
        top5 = ", ".join(f"{r[0]:02d}:{r[2]*100:.2f}%({r[1]})" for r in g.head(5).iter_rows())
        bot5 = ", ".join(f"{r[0]:02d}:{r[2]*100:.2f}%({r[1]})" for r in g.tail(5).iter_rows())
        print(f"  {col}: TOP5=[{top5}] | BOT5=[{bot5}]")

    print("\n=== 4. Check Exact String Formatting / Decimal Places in raw CSV ===")
    raw_str = pl.read_csv(DATA_RAW_DIR / "train.csv", infer_schema_length=0)
    for col in ["amount_usd", "account_balance_usd", "hours_since_last_txn", "distance_from_home_km", "merchant_risk_score", "velocity_score"]:
        dec_len = raw_str[col].str.split(".").list.get(1, null_on_oob=True).str.len_chars().fill_null(0)
        g = pl.DataFrame({"dlen": dec_len, "y": y}).group_by("dlen").agg(pl.len().alias("n"), pl.col("y").mean().alias("rate")).sort("dlen")
        items = ", ".join(f"len={r[0]}:{r[2]*100:.2f}%({r[1]:,})" for r in g.iter_rows())
        print(f"  {col:<24}: {items}")

    print("\n=== 5. Check Conditional Correlation: does grouping other features by exact balance/amount predict fraud? ===")
    # E.g. mean(merchant_risk_score) over exact account_balance_usd or exact amount_usd
    df_all = pl.concat([train.drop(TARGET_COL), test], how="vertical")
    for gcol in ["account_balance_usd", "amount_usd", "hours_since_last_txn", "distance_from_home_km"]:
        aggs = df_all.with_columns([
            pl.col("merchant_risk_score").mean().over(gcol).alias("g_mrs"),
            pl.col("velocity_score").mean().over(gcol).alias("g_vel"),
            pl.col("cvv_retry_count").mean().over(gcol).alias("g_cvv"),
            pl.col("is_new_merchant").cast(pl.Float32).mean().over(gcol).alias("g_new"),
            pl.len().over(gcol).alias("g_cnt"),
        ]).slice(0, train.height)
        aucs = [f"{c}:{roc_auc_score(y, aggs[c].to_numpy()):.4f}" for c in ["g_mrs", "g_vel", "g_cvv", "g_new", "g_cnt"]]
        print(f"  over({gcol:<22}): " + " | ".join(aucs))

    print("\n=== 6. Check Multi-Column Exact Near-Duplicates (matching top 6 numeric cols) ===")
    key6 = ["amount_usd", "account_balance_usd", "hours_since_last_txn", "distance_from_home_km", "merchant_risk_score", "velocity_score"]
    n_uniq6 = df_all.select(key6).n_unique()
    print(f"  Total rows={df_all.height:,}, unique 6-num tuples={n_uniq6:,}")
    for pair in [("amount_usd", "account_balance_usd"), ("merchant_risk_score", "velocity_score"), ("hours_since_last_txn", "distance_from_home_km")]:
        nu = df_all.select(list(pair)).n_unique()
        print(f"  unique {pair}: {nu:,} (avg {df_all.height / nu:.2f} rows/pair)")

    print("\n=== 7. Inspect Extreme Value Ranges & Threshold Cliffs ===")
    for col in ["merchant_risk_score", "velocity_score", "customer_age", "txn_count_last_24h", "hours_since_last_txn", "amount_usd"]:
        q = pd.qcut(train[col].to_numpy(), q=10, duplicates="drop")
        g = pd.DataFrame({"q": q, "y": y}).groupby("q", observed=False)["y"].agg(["count", "mean"])
        rates = " -> ".join(f"{r*100:.2f}%" for r in g["mean"])
        print(f"  {col:<24} deciles: {rates}")


if __name__ == "__main__":
    main()
