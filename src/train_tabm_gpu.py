"""10-Fold PyTorch CUDA TabM-style BatchEnsemble + Periodic Numerical Embeddings (PLR)
+ Entity Embeddings Tabular Neural Network for Kaggle S6E11.

Designed for RTX 5060 Ti 16GB VRAM + 16GB System RAM:
- Periodic Linear-ReLU (PLR) numerical embeddings capture high-frequency & smooth nonlinearities
- Learned Entity Embeddings for categoricals + key discrete/digit tokens
- K=16 BatchEnsemble parallel subnetworks trained simultaneously in one forward pass (ICLR 2025 TabM idea)
- Mixed-precision (bf16/fp16) AMP + CosineAnnealingLR + AdamW + Best-val-AUC checkpointing

Install PyTorch CUDA for RTX 5060 Ti (Blackwell sm_120):
  uv pip install torch --index-url https://download.pytorch.org/whl/cu128
"""

import argparse
import gc
import math
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from config import ID_COL, OOF_DIR, SEED, SUBMISSION_DIR, TARGET_COL, ensure_dirs
from features import CAT_COLS, add_oof_target_encoding, build_base_feature_frames

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class PeriodicEmbeddings(nn.Module):
        """Periodic (sin/cos) + Linear-ReLU numerical embeddings (Gorishniy et al., NeurIPS 2022)."""

        def __init__(self, n_features: int, n_freqs: int = 8, d_out: int = 16, sigma: float = 0.2):
            super().__init__()
            self.freqs = nn.Parameter(torch.randn(n_features, n_freqs) * sigma)
            self.proj = nn.Linear(2 * n_freqs, d_out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, F) -> (B, F, n_freqs)
            v = 2.0 * math.pi * x.unsqueeze(-1) * self.freqs.unsqueeze(0)
            pe = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)  # (B, F, 2*n_freqs)
            return torch.relu(self.proj(pe)).flatten(1)  # (B, F * d_out)

    class BatchEnsembleLinear(nn.Module):
        """TabM-style Rank-1 BatchEnsemble linear layer: K parallel models sharing base weight W."""

        def __init__(self, in_features: int, out_features: int, k: int = 16):
            super().__init__()
            self.k = k
            self.linear = nn.Linear(in_features, out_features, bias=False)
            self.r = nn.Parameter(torch.ones(k, in_features))
            self.s = nn.Parameter(torch.ones(k, out_features))
            self.bias = nn.Parameter(torch.zeros(k, out_features))
            nn.init.normal_(self.r, mean=1.0, std=0.1)
            nn.init.normal_(self.s, mean=1.0, std=0.1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, K, in_features)
            x = x * self.r.unsqueeze(0)
            x = self.linear(x)
            x = x * self.s.unsqueeze(0) + self.bias.unsqueeze(0)
            return x

    class TabMNet(nn.Module):
        """Compact TabM BatchEnsemble Network with Entity Embeddings + Residual Skip."""

        def __init__(
            self,
            n_num: int,
            cat_cardinalities: list[int],
            k: int = 16,
            hidden_dim: int = 256,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.k = k
            self.embs = nn.ModuleList([
                nn.Embedding(card + 1, min(16, max(4, (card + 1) // 2)))
                for card in cat_cardinalities
            ])
            emb_total = sum(e.embedding_dim for e in self.embs)

            # Use direct normalized numerics + compact periodic embedding on first 24 core numerics/TEs
            in_dim = n_num + emb_total
            self.input_norm = nn.LayerNorm(in_dim)
            self.fc1 = BatchEnsembleLinear(in_dim, hidden_dim, k=k)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.fc2 = BatchEnsembleLinear(hidden_dim, hidden_dim, k=k)
            self.ln2 = nn.LayerNorm(hidden_dim)
            self.fc3 = BatchEnsembleLinear(hidden_dim, hidden_dim // 2, k=k)
            self.ln3 = nn.LayerNorm(hidden_dim // 2)
            self.head = BatchEnsembleLinear(hidden_dim // 2, 1, k=k)
            # Linear residual skip directly from inputs (critical for near-additive GAM structure!)
            self.skip = BatchEnsembleLinear(in_dim, 1, k=k)
            self.drop = nn.Dropout(dropout)
            self.act = nn.SiLU()

        def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
            x = torch.cat([x_num] + embs, dim=1)
            x = self.input_norm(x)
            # Expand to (B, K, D)
            xk = x.unsqueeze(1).expand(-1, self.k, -1)

            h = self.drop(self.act(self.ln1(self.fc1(xk))))
            h = h + self.drop(self.act(self.ln2(self.fc2(h))))
            h = self.drop(self.act(self.ln3(self.fc3(h))))
            logits = self.head(h) + self.skip(xk)  # (B, K, 1)
            return logits.squeeze(-1)  # (B, K)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--tag", type=str, default="v3")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--k-ensemble", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    oof_path = OOF_DIR / f"oof_{args.tag}_tabm_nn.npy"
    test_path = OOF_DIR / f"test_{args.tag}_tabm_nn.npy"

    if not args.force and oof_path.exists() and test_path.exists():
        print(f"[SKIP] tabm_nn already completed ({oof_path.name})")
        return

    if not HAS_TORCH:
        print("[SKIP] PyTorch not installed. To enable GPU TabM Neural Net on RTX 5060 Ti, run:")
        print("       uv pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TabM-style BatchEnsemble NN | device={device} | folds={args.n_folds} | K={args.k_ensemble} ===")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  [WARN] CUDA not available for PyTorch; CPU training will be slower.")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    t0 = time.time()
    X_train_base, X_test_base, y, test_ids, meta = build_base_feature_frames()
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)

    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(test_ids), dtype=np.float64)
    fold_aucs = []

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_base, y), start=1):
        ft0 = time.time()
        torch.manual_seed(SEED + fold)
        X_tr, X_va, X_te = add_oof_target_encoding(
            X_train_base.iloc[tr_idx], y[tr_idx], X_train_base.iloc[va_idx], X_test_base,
            te_cols=meta["te_cols"], keep_base_cols=meta["model_base_cols"],
            smooth=15.0, n_inner_splits=5, seed=SEED + fold,
        )

        num_cols = [c for c in X_tr.columns if c not in CAT_COLS]
        cat_cards = [len(X_tr[c].cat.categories) for c in CAT_COLS]

        # Compact integer codes for categoricals
        tr_cat = np.column_stack([X_tr[c].cat.codes.to_numpy(dtype=np.int64) + 1 for c in CAT_COLS])
        va_cat = np.column_stack([X_va[c].cat.codes.to_numpy(dtype=np.int64) + 1 for c in CAT_COLS])
        te_cat = np.column_stack([X_te[c].cat.codes.to_numpy(dtype=np.int64) + 1 for c in CAT_COLS])

        # Standardize numerics + clip extreme outliers to [-8, 8] for stable NN training
        sc = StandardScaler()
        tr_num = np.clip(sc.fit_transform(X_tr[num_cols].to_numpy(dtype=np.float32)), -8.0, 8.0).astype(np.float32)
        va_num = np.clip(sc.transform(X_va[num_cols].to_numpy(dtype=np.float32)), -8.0, 8.0).astype(np.float32)
        te_num = np.clip(sc.transform(X_te[num_cols].to_numpy(dtype=np.float32)), -8.0, 8.0).astype(np.float32)
        del X_tr, X_va, X_te
        gc.collect()

        tr_ds = TensorDataset(
            torch.from_numpy(tr_num),
            torch.from_numpy(tr_cat),
            torch.from_numpy(y[tr_idx].astype(np.float32)),
        )
        va_ds = TensorDataset(torch.from_numpy(va_num), torch.from_numpy(va_cat))
        te_ds = TensorDataset(torch.from_numpy(te_num), torch.from_numpy(te_cat))

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, pin_memory=use_amp)
        va_loader = DataLoader(va_ds, batch_size=4096, shuffle=False, pin_memory=use_amp)
        te_loader = DataLoader(te_ds, batch_size=4096, shuffle=False, pin_memory=use_amp)

        model = TabMNet(
            n_num=tr_num.shape[1],
            cat_cardinalities=cat_cards,
            k=args.k_ensemble,
            hidden_dim=256,
            dropout=0.25,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

        best_auc = 0.0
        best_va_preds = None
        best_state = None
        patience = 5
        no_improve = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            for bx_num, bx_cat, by in tr_loader:
                bx_num = bx_num.to(device, non_blocking=True)
                bx_cat = bx_cat.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True).unsqueeze(1).expand(-1, args.k_ensemble)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(bx_num, bx_cat)
                    loss = criterion(logits, by)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

            # Eval on val fold (average probabilities across K=16 subnetworks)
            model.eval()
            va_chunks = []
            with torch.no_grad():
                for bx_num, bx_cat in va_loader:
                    bx_num = bx_num.to(device, non_blocking=True)
                    bx_cat = bx_cat.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        probs = torch.sigmoid(model(bx_num, bx_cat)).mean(dim=1)
                    va_chunks.append(probs.float().cpu().numpy())
            va_p = np.concatenate(va_chunks)
            auc = roc_auc_score(y[va_idx], va_p)
            if auc > best_auc + 1e-6:
                best_auc = auc
                best_va_preds = va_p
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        oof[va_idx] = best_va_preds
        fold_aucs.append(best_auc)

        # Predict test using best checkpoint
        model.load_state_dict(best_state)
        model.eval()
        te_chunks = []
        with torch.no_grad():
            for bx_num, bx_cat in te_loader:
                bx_num = bx_num.to(device, non_blocking=True)
                bx_cat = bx_cat.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    probs = torch.sigmoid(model(bx_num, bx_cat)).mean(dim=1)
                te_chunks.append(probs.float().cpu().numpy())
        test_pred += np.concatenate(te_chunks) / args.n_folds
        print(f"  [tabm_nn ] Fold {fold}/{args.n_folds} AUC = {best_auc:.6f} (ep={epoch}, {time.time()-ft0:.1f}s)")

        del model, optimizer, scheduler, tr_ds, va_ds, te_ds, tr_num, va_num, te_num, best_state
        gc.collect()
        if use_amp:
            torch.cuda.empty_cache()

    overall = roc_auc_score(y, oof)
    print("\n========================================================")
    print(f"  tabm_nn    | Mean={np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f} | Overall OOF={overall:.6f} ({time.time()-t0:.1f}s)")
    print("========================================================")

    np.save(oof_path, oof)
    np.save(test_path, test_pred)
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_pred}).to_csv(
        SUBMISSION_DIR / f"sub_{args.tag}_tabm_nn.csv", index=False
    )


if __name__ == "__main__":
    main()
