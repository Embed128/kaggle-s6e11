"""V4 GPU RealMLP-PBLD (High-Frequency Periodic Embeddings sigma=5.0 + 8-Member BatchEnsemble
+ SmoothClip + ScalingLayer + LOO Unsupervised Cluster Profile) for Kaggle S6E11.

Fixes what V3 tabm_nn missed:
- High-frequency PBLD periodic numerical embeddings (sigma=5.0, n_freqs=16 -> resolves sharp synthetic value spikes!)
- Robust median/IQR scaling + SmoothClip (x / sqrt(1 + (x/3)^2))
- Learnable per-feature ScalingLayer + Linear residual skip
- Includes 1M-row LOO unsupervised cluster features alongside FE v3 OOF TE
"""

import argparse
import gc
import math
import time

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import DATA_RAW_DIR, ID_COL, OOF_DIR, SEED, SUBMISSION_DIR, TARGET_COL, ensure_dirs
from features import BOOL_COLS, CAT_COLS, NUM_COLS, add_oof_target_encoding, build_base_feature_frames

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def build_unsupervised_loo_cluster_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build 100% leak-free unsupervised Leave-One-Out cluster profile features across all 1M rows."""
    train_pl = pl.read_csv(DATA_RAW_DIR / "train.csv")
    test_pl = pl.read_csv(DATA_RAW_DIR / "test.csv")
    n_train = train_pl.height
    raw_cols = [c for c in test_pl.columns if c != ID_COL]

    df = pl.concat([train_pl.select(raw_cols), test_pl.select(raw_cols)], how="vertical").with_columns(
        [pl.col(c).cast(pl.Float32) for c in NUM_COLS]
        + [pl.col(c).cast(pl.Int8) for c in BOOL_COLS]
        + [(pl.col("auth_method") == "No Authentication").cast(pl.Float32).alias("is_no_auth")]
        + [
            ((pl.col("hours_since_last_txn") * 100.0).round(0).cast(pl.Int64) % 100).alias("h_cents"),
            ((pl.col("amount_usd") * 100.0).round(0).cast(pl.Int64) % 100).alias("a_cents"),
            ((pl.col("distance_from_home_km") * 100.0).round(0).cast(pl.Int64) % 100).alias("d_cents"),
            ((pl.col("account_balance_usd") * 100.0).round(0).cast(pl.Int64) % 100).alias("b_cents"),
            pl.col("amount_usd").floor().cast(pl.Int32).alias("amount_int"),
            pl.col("account_balance_usd").floor().cast(pl.Int32).alias("balance_int"),
        ]
    )
    del train_pl, test_pl
    gc.collect()

    keys = [
        "account_balance_usd", "amount_usd", "hours_since_last_txn", "distance_from_home_km",
        "merchant_risk_score", "velocity_score", "customer_age", "card_age_months",
        "h_cents", "a_cents", "d_cents", "b_cents", "amount_int", "balance_int",
    ]
    profile_cols = [
        "cvv_retry_count", "merchant_risk_score", "velocity_score",
        "is_new_merchant", "ip_country_mismatch", "is_no_auth",
    ]

    exprs = []
    for key in keys:
        cnt = pl.len().over(key).cast(pl.Float32)
        for pcol in profile_cols:
            if pcol == key:
                continue
            g_mean = float(df[pcol].mean())
            loo_sum = pl.col(pcol).sum().over(key).cast(pl.Float32) - pl.col(pcol).cast(pl.Float32)
            loo_cnt = (cnt - 1.0).clip(lower_bound=0.0)
            loo_val = ((loo_sum + 5.0 * g_mean) / (loo_cnt + 5.0)).cast(pl.Float32)
            exprs.append(loo_val.alias(f"loo_{pcol}_by_{key}"))

    out = df.select(exprs)
    tr_loo = out.slice(0, n_train).to_pandas()
    te_loo = out.slice(n_train, out.height - n_train).to_pandas()
    del df, out
    gc.collect()
    return tr_loo, te_loo


def robust_smooth_clip_fit_transform(
    X_tr: np.ndarray, X_va: np.ndarray, X_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RealMLP preprocessing: median center + IQR scale + smooth clip x / sqrt(1 + (x/3)^2)."""
    med = np.median(X_tr, axis=0)
    q75 = np.quantile(X_tr, 0.75, axis=0)
    q25 = np.quantile(X_tr, 0.25, axis=0)
    scale = q75 - q25
    zero_mask = scale < 1e-8
    if np.any(zero_mask):
        std = np.std(X_tr[:, zero_mask], axis=0)
        scale[zero_mask] = np.where(std > 1e-8, std, 1.0)
    inv_scale = (1.0 / scale).astype(np.float32)
    med = med.astype(np.float32)

    def tfm(arr: np.ndarray) -> np.ndarray:
        z = (arr - med) * inv_scale
        return (z / np.sqrt(1.0 + (z / 3.0) ** 2)).astype(np.float32)

    return tfm(X_tr), tfm(X_va), tfm(X_te)


if HAS_TORCH:
    class PBLDEmbedding(nn.Module):
        """High-Frequency Periodic Basis Linear-Activation Embeddings (RealMLP style)."""

        def __init__(self, n_features: int, n_freqs: int = 12, out_dim: int = 4, freq_scale: float = 5.0):
            super().__init__()
            self.freqs = nn.Parameter(torch.randn(n_features, n_freqs) * freq_scale)
            self.phase = nn.Parameter(torch.zeros(n_features, n_freqs))
            self.proj = nn.Parameter(torch.randn(n_features, 2 * n_freqs, out_dim) * (1.0 / math.sqrt(2 * n_freqs)))
            self.bias = nn.Parameter(torch.zeros(n_features, out_dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, F) -> periodic (B, F, 2*n_freqs) -> linear per feature -> (B, F * out_dim)
            angles = x.unsqueeze(-1) * self.freqs.unsqueeze(0) + self.phase.unsqueeze(0)
            pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            out = torch.einsum("bfp,fpo->bfo", pe, self.proj) + self.bias.unsqueeze(0)
            return F.silu(out).flatten(1)

    class EnsembleLinear(nn.Module):
        """True independent K-member parallel linear layer (einsum) like RealMLP n_ens=8!"""

        def __init__(self, k: int, in_dim: int, out_dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(k, in_dim, out_dim))
            self.bias = nn.Parameter(torch.zeros(k, out_dim))
            bound = 1.0 / math.sqrt(in_dim)
            nn.init.uniform_(self.weight, -bound, bound)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, K, in_dim) -> (B, K, out_dim)
            return torch.einsum("bki,kio->bko", x, self.weight) + self.bias.unsqueeze(0)

    class RealMLPPBLD(nn.Module):
        def __init__(
            self,
            n_num: int,
            n_pbld: int,
            cat_cards: list[int],
            k: int = 8,
            hidden_dim: int = 256,
            dropout: float = 0.08,
        ):
            super().__init__()
            self.k = k
            self.n_pbld = n_pbld
            self.pbld = PBLDEmbedding(n_pbld, n_freqs=12, out_dim=4, freq_scale=5.0)
            self.embs = nn.ModuleList([
                nn.Embedding(c + 1, min(12, max(4, (c + 1) // 2))) for c in cat_cards
            ])
            emb_dim = sum(e.embedding_dim for e in self.embs)
            in_dim = n_num + n_pbld * 4 + emb_dim

            self.front_scale = nn.Parameter(torch.ones(k, in_dim))
            self.fc1 = EnsembleLinear(k, in_dim, hidden_dim)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.fc2 = EnsembleLinear(k, hidden_dim, hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
            self.fc3 = EnsembleLinear(k, hidden_dim, hidden_dim // 2)
            self.ln3 = nn.LayerNorm(hidden_dim // 2)
            self.head = EnsembleLinear(k, hidden_dim // 2, 1)
            self.skip = EnsembleLinear(k, in_dim, 1)
            self.drop = nn.Dropout(dropout)

        def forward(self, x_num: torch.Tensor, x_pbld: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
            pe = self.pbld(x_pbld)
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
            x = torch.cat([x_num, pe] + embs, dim=1)
            xk = x.unsqueeze(1) * self.front_scale.unsqueeze(0)  # (B, K, in_dim)

            h = self.drop(F.silu(self.ln1(self.fc1(xk))))
            h = h + self.drop(F.silu(self.ln2(self.fc2(h))))
            h = self.drop(F.silu(self.ln3(self.fc3(h))))
            return (self.head(h) + self.skip(xk)).squeeze(-1)  # (B, K)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--tag", type=str, default="v4")
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--k-ensemble", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    oof_path = OOF_DIR / f"oof_{args.tag}_realmlp_pbld.npy"
    test_path = OOF_DIR / f"test_{args.tag}_realmlp_pbld.npy"
    if not args.force and oof_path.exists() and test_path.exists():
        print(f"[SKIP] realmlp_pbld already completed ({oof_path.name})")
        return

    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for RealMLP-PBLD!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"=== RealMLP-PBLD | device={device} | folds={args.n_folds} | K={args.k_ensemble} ===")

    t0 = time.time()
    X_train_base, X_test_base, y, test_ids, meta = build_base_feature_frames()
    tr_loo, te_loo = build_unsupervised_loo_cluster_features()
    print(f"Loaded base + {tr_loo.shape[1]} LOO cluster features in {time.time()-t0:.1f}s")

    ckpt_dir = OOF_DIR / f"ckpt_{args.tag}_realmlp_pbld"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(test_ids), dtype=np.float64)
    fold_aucs = []

    # Columns that benefit most from high-frequency PBLD periodic embeddings
    pbld_cols = NUM_COLS + [
        "te_merchant_risk_score", "te_velocity_score", "te_amount_usd",
        "te_account_balance_usd", "te_hours_since_last_txn", "te_distance_from_home_km",
        "te_customer_age", "te_cvv_retry_count", "te_auth_method", "te_merchant_category",
    ]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_base, y), start=1):
        f_oof_file = ckpt_dir / f"fold{fold}_oof.npy"
        f_test_file = ckpt_dir / f"fold{fold}_test.npy"
        if not args.force and f_oof_file.exists() and f_test_file.exists():
            va_p = np.load(f_oof_file)
            te_p = np.load(f_test_file)
            oof[va_idx] = va_p
            test_pred += te_p / args.n_folds
            f_auc = roc_auc_score(y[va_idx], va_p)
            fold_aucs.append(f_auc)
            print(f"  [realmlp] Fold {fold}/{args.n_folds} [RESUMED] AUC = {f_auc:.6f}")
            continue

        ft0 = time.time()
        torch.manual_seed(SEED + 300 + fold)
        np.random.seed(SEED + 300 + fold)

        X_tr, X_va, X_te = add_oof_target_encoding(
            X_train_base.iloc[tr_idx], y[tr_idx], X_train_base.iloc[va_idx], X_test_base,
            te_cols=meta["te_cols"], keep_base_cols=meta["model_base_cols"],
            smooth=15.0, n_inner_splits=5, seed=SEED + fold,
        )

        # Attach LOO unsupervised cluster features
        X_tr = pd.concat([X_tr, tr_loo.iloc[tr_idx].reset_index(drop=True)], axis=1)
        X_va = pd.concat([X_va, tr_loo.iloc[va_idx].reset_index(drop=True)], axis=1)
        X_te = pd.concat([X_te, te_loo.reset_index(drop=True)], axis=1)

        num_cols = [c for c in X_tr.columns if c not in CAT_COLS]
        cat_cards = [len(X_tr[c].cat.categories) for c in CAT_COLS]

        tr_cat = np.column_stack([X_tr[c].cat.codes.to_numpy(dtype=np.int64) + 1 for c in CAT_COLS])
        va_cat = np.column_stack([X_va[c].cat.codes.to_numpy(dtype=np.int64) + 1 for c in CAT_COLS])
        te_cat = np.column_stack([X_te[c].cat.codes.to_numpy(dtype=np.int64) + 1 for c in CAT_COLS])

        tr_num, va_num, te_num = robust_smooth_clip_fit_transform(
            X_tr[num_cols].to_numpy(dtype=np.float32),
            X_va[num_cols].to_numpy(dtype=np.float32),
            X_te[num_cols].to_numpy(dtype=np.float32),
        )
        tr_pb, va_pb, te_pb = robust_smooth_clip_fit_transform(
            X_tr[pbld_cols].to_numpy(dtype=np.float32),
            X_va[pbld_cols].to_numpy(dtype=np.float32),
            X_te[pbld_cols].to_numpy(dtype=np.float32),
        )
        del X_tr, X_va, X_te
        gc.collect()

        tr_ds = TensorDataset(
            torch.from_numpy(tr_num), torch.from_numpy(tr_pb),
            torch.from_numpy(tr_cat), torch.from_numpy(y[tr_idx].astype(np.float32)),
        )
        va_ds = TensorDataset(torch.from_numpy(va_num), torch.from_numpy(va_pb), torch.from_numpy(va_cat))
        te_ds = TensorDataset(torch.from_numpy(te_num), torch.from_numpy(te_pb), torch.from_numpy(te_cat))

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, pin_memory=use_amp)
        va_loader = DataLoader(va_ds, batch_size=4096, shuffle=False, pin_memory=use_amp)
        te_loader = DataLoader(te_ds, batch_size=4096, shuffle=False, pin_memory=use_amp)

        model = RealMLPPBLD(
            n_num=tr_num.shape[1], n_pbld=tr_pb.shape[1],
            cat_cards=cat_cards, k=args.k_ensemble, hidden_dim=256, dropout=0.08,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

        best_auc = 0.0
        best_va_p = None
        best_state = None
        patience = 5
        no_improve = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            for b_num, b_pb, b_cat, by in tr_loader:
                b_num = b_num.to(device, non_blocking=True)
                b_pb = b_pb.to(device, non_blocking=True)
                b_cat = b_cat.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True).unsqueeze(1).expand(-1, args.k_ensemble)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(b_num, b_pb, b_cat)
                    loss = criterion(logits, by)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

            model.eval()
            chunks = []
            with torch.no_grad():
                for b_num, b_pb, b_cat in va_loader:
                    b_num = b_num.to(device, non_blocking=True)
                    b_pb = b_pb.to(device, non_blocking=True)
                    b_cat = b_cat.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        probs = torch.sigmoid(model(b_num, b_pb, b_cat)).mean(dim=1)
                    chunks.append(probs.float().cpu().numpy())
            va_p = np.concatenate(chunks)
            auc = roc_auc_score(y[va_idx], va_p)
            if auc > best_auc + 1e-6:
                best_auc = auc
                best_va_p = va_p
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience and epoch >= 8:
                    break

        model.load_state_dict(best_state)
        model.eval()
        te_chunks = []
        with torch.no_grad():
            for b_num, b_pb, b_cat in te_loader:
                b_num = b_num.to(device, non_blocking=True)
                b_pb = b_pb.to(device, non_blocking=True)
                b_cat = b_cat.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    probs = torch.sigmoid(model(b_num, b_pb, b_cat)).mean(dim=1)
                te_chunks.append(probs.float().cpu().numpy())
        te_p = np.concatenate(te_chunks)

        np.save(f_oof_file, best_va_p)
        np.save(f_test_file, te_p)
        oof[va_idx] = best_va_p
        test_pred += te_p / args.n_folds
        fold_aucs.append(best_auc)
        print(f"  [realmlp] Fold {fold}/{args.n_folds} AUC = {best_auc:.6f} (ep={epoch}, {time.time()-ft0:.1f}s)", flush=True)

        del model, optimizer, scheduler, tr_ds, va_ds, te_ds, tr_num, va_num, te_num, tr_pb, va_pb, te_pb, best_state
        gc.collect()
        if use_amp:
            torch.cuda.empty_cache()

    overall = roc_auc_score(y, oof)
    print("\n========================================================")
    print(f"  realmlp_pbld | Mean={np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f} | Overall OOF={overall:.6f}")
    print("========================================================")

    np.save(oof_path, oof)
    np.save(test_path, test_pred)
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_pred}).to_csv(
        SUBMISSION_DIR / f"sub_{args.tag}_realmlp_pbld.csv", index=False
    )


if __name__ == "__main__":
    main()
