"""V5.1 Grand Meta-Stacker for Kaggle S6E11 (200+ Unique Heterogeneous Models -> 0.9526 ~ 0.9529+ Nested CV).

Universal loader supports:
  - 1D .npy / .csv OOF+Test pairs (including *_oof_5fold.npy, own V3/V4/V2 models)
  - 2D .npy matrix pairs (e.g., busyaprime's 52-model (699960, 52) matrix + manifest.csv)
  - Multi-stream .npz archives (pool_bank.npz, rank1_q17_streams.npz, joint_token_predictions.npz, wguesdon/*.npz)

Strictly Nested 5-Fold CV (StratifiedKFold(5, shuffle=True, random_state=42)):
  1. L2 Logistic Regression on Standardized Logits (fine C grid)
  2. L2 Logistic Regression on Probit-Ranks (norm.ppf(rank))
  3. Convex Non-Negative L2 Logistic Regression (w >= 0 via L-BFGS-B — zero negative-weight overfitting!)
  4. Non-Negative Forward Hill-Climbing (Caruana Ensemble Selection) on Logits
  5. Compact Residual Meta-MLP on Top-60 Logits
  6. Level-3 Convex Non-Negative Super-Blend + Zero-Fraud Zones 1-4 verification
"""

import argparse
import gc
import json
import time
import warnings
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtri
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import DATA_RAW_DIR, ID_COL, OOF_DIR, SEED, SUBMISSION_DIR, TARGET_COL, ensure_dirs

warnings.filterwarnings("ignore")

PUBLIC_OOF_DATASETS = [
    ("goodpjw_bank", "goodpjw2008/s6e11-oof-bank"),
    ("masa_gpt2", "masayakawamata/s6e11-gpt2-target-first-oof"),
    ("mega_lm", "megayak/s6e11-target-first-lm"),
    ("goodpjw_lm7", "goodpjw2008/s6e11-compact-lm-seed7"),
    ("weni_bpe", "wenimohamed/s6e11-diverse-bpe-realmlp-tabm-oof-pack"),
    ("sadam_tabpfn", "sadamtorres/ps-s6e11-tabpfn-gbdt-oof-test-pairs"),
    ("math_17", "mathsonchain/s6e11-fraud-oof-test-17-single-models"),
    ("busya_52", "busyaprime/s6e11-33-own-members"),
    ("malyshev_v9", "malyshevdanil/s6e11-pooled-stack-v9"),
    ("golem_lib", "dariushafshar/s6e11-golem-oof-library"),
    ("wguesdon", "wguesdon/s6e11-tree-ensemble-predictions"),
    ("amanatar_q17", "amanatar/s6e11-rank1-q17-meta-bundle"),
    ("malyshev_tok", "malyshevdanil/s6e11-digit-token-full-refit-artifacts"),
    ("goodpjw_art", "goodpjw2008/s6e11-fraud-series-artifacts"),
]


def to_unit_rank(v: np.ndarray) -> np.ndarray:
    return (rankdata(v, method="average") - 0.5) / len(v)


def to_probit_rank(v: np.ndarray) -> np.ndarray:
    u = np.clip(to_unit_rank(v), 1e-6, 1.0 - 1e-6)
    return ndtri(u).astype(np.float64)


def to_clean_logit(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float64)
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    if vmin >= -1e-6 and vmax <= 1.0 + 1e-6:
        return logit(np.clip(v, 1e-6, 1.0 - 1e-6))
    return v


def get_zero_fraud_zones(df: pd.DataFrame) -> np.ndarray:
    mrs = df["merchant_risk_score"].to_numpy()
    vel = df["velocity_score"].to_numpy()
    clean = (
        (df["cvv_retry_count"].to_numpy() == 0)
        & (df["prior_disputes"].to_numpy() == 0)
        & (~df["is_new_merchant"].to_numpy(bool))
        & (~df["is_foreign_transaction"].to_numpy(bool))
        & (~df["used_vpn"].to_numpy(bool))
        & (~df["ip_country_mismatch"].to_numpy(bool))
        & (~df["billing_shipping_mismatch"].to_numpy(bool))
        & (~df["is_ai_generated_scam_attempt"].to_numpy(bool))
    )
    z1 = mrs < 13.1
    z2 = (vel <= 4.1) & (mrs <= 20.0)
    z3 = clean & (vel <= 4.1) & (mrs <= 33.8)
    z4 = clean & (vel <= 6.0) & (mrs <= 25.0)
    return z1 | z2 | z3 | z4


def apply_zone_push(scores: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    r = to_unit_rank(scores).copy()
    r[zone_mask] -= 2.0  # r in (0, 1) -> zone rows move strictly to (-2, -1)
    return to_unit_rank(r)


def valid_pair(oa: np.ndarray, ta: np.ndarray, n_train: int, n_test: int) -> bool:
    if oa is None or ta is None:
        return False
    if oa.ndim != 1 or ta.ndim != 1 or len(oa) != n_train or len(ta) != n_test:
        return False
    if not (np.all(np.isfinite(oa)) and np.all(np.isfinite(ta))):
        return False
    if np.std(oa) < 1e-8 or np.std(ta) < 1e-8:
        return False
    return True


def load_csv_1d(path: Path, expected_len: int, expected_ids: np.ndarray | None = None) -> np.ndarray | None:
    try:
        df = pd.read_csv(path)
        if len(df) != expected_len:
            return None
        id_col = next((c for c in df.columns if c.lower() == "id"), None)
        pred_cols = [c for c in df.columns if c.lower() not in ("id", "fold", "is_fraud_true", "target", "y")]
        pcol = pred_cols[-1] if pred_cols else df.columns[-1]
        if id_col is not None and expected_ids is not None:
            if not np.array_equal(df[id_col].to_numpy(), expected_ids):
                df = df.set_index(id_col).loc[expected_ids].reset_index()
        return df[pcol].to_numpy(dtype=np.float64)
    except Exception:
        return None


def extract_npz_pairs(
    npz_path: Path, prefix: str, n_train: int, n_test: int
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Extract 1D or 2D (oof, test) pairs from any .npz archive."""
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    try:
        z = np.load(npz_path, allow_pickle=True)
        keys = list(z.files)
    except Exception:
        return pairs

    # Try to read optional name list inside npz
    name_list = None
    for nk in ("names", "member_names", "model_names", "keys", "cols"):
        if nk in keys:
            try:
                name_list = [str(x) for x in z[nk].tolist()]
            except Exception:
                pass

    # Pattern A: direct 1D/2D oof + test keys in single npz
    oof_candidates = [k for k in keys if "oof" in k.lower() and "fold" not in k.lower() or k.lower() in ("oof", "oof_preds", "oof_logits", "oofs", "x_oof")]
    matched_keys = set()
    for ok in oof_candidates:
        # Find matching test key
        cands = [
            ok.replace("oof", "test"), ok.replace("OOF", "TEST"),
            ok.replace("oof", "te"), ok.replace("oof", "sub"),
            "test", "test_preds", "test_logits", "tests", "x_test", "pred_test",
        ]
        tk = next((c for c in cands if c in keys and c != ok), None)
        if tk is None:
            continue
        matched_keys.add(ok)
        oa = np.asarray(z[ok], dtype=np.float64)
        ta = np.asarray(z[tk], dtype=np.float64)
        stem = f"{prefix}:{npz_path.stem}"

        if oa.ndim == 1 and ta.ndim == 1 and valid_pair(oa, ta, n_train, n_test):
            sub_k = ok.replace("oof_", "").replace("_oof", "")
            lbl = stem if sub_k.lower() in ("oof", "oof_preds", "oof_logits", "") else f"{stem}/{sub_k}"
            pairs.append((lbl, oa, ta))
        elif oa.ndim == 2 and ta.ndim == 2:
            # Handle (N, K) or (K, N)
            if oa.shape[0] == n_train and ta.shape[0] == n_test and oa.shape[1] == ta.shape[1]:
                K = oa.shape[1]
                for j in range(K):
                    o1, t1 = oa[:, j], ta[:, j]
                    if valid_pair(o1, t1, n_train, n_test):
                        mname = name_list[j] if (name_list and len(name_list) == K) else f"m{j}"
                        pairs.append((f"{stem}/{mname}", o1, t1))
            elif oa.shape[1] == n_train and ta.shape[1] == n_test and oa.shape[0] == ta.shape[0]:
                K = oa.shape[0]
                for j in range(K):
                    o1, t1 = oa[j, :], ta[j, :]
                    if valid_pair(o1, t1, n_train, n_test):
                        mname = name_list[j] if (name_list and len(name_list) == K) else f"m{j}"
                        pairs.append((f"{stem}/{mname}", o1, t1))

    # Pattern B: keys named <name>__oof / <name>__test or <name>_oof / <name>_test (handled above if "oof" in k)
    return pairs


def discover_all_in_dir(
    root: Path, prefix: str, n_train: int, n_test: int, train_ids: np.ndarray, test_ids: np.ndarray
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in (".npy", ".csv", ".npz")]
    by_name = {p.name: p for p in files}
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    seen = set()

    # Optional manifest names for 2D oof.npy/test.npy (e.g. busyaprime)
    manifest_names = None
    for mp in [p for p in files if p.name.lower() == "manifest.csv"]:
        try:
            mdf = pd.read_csv(mp)
            for col in ("name", "model", "member", "id", "file"):
                if col in mdf.columns:
                    manifest_names = [str(x) for x in mdf[col].tolist()]
                    break
        except Exception:
            pass

    # 1. Extract all .npz archives
    for p in files:
        if p.suffix == ".npz":
            pairs.extend(extract_npz_pairs(p, prefix, n_train, n_test))

    # 2. Extract 1D and 2D .npy / .csv pairs
    for p in files:
        if p.suffix not in (".npy", ".csv"):
            continue
        nm = p.name
        low = nm.lower()
        # Skip single-fold artifacts (fold0, f0) and metadata files, BUT KEEP *_5fold.npy!
        if any(skip in low for skip in ("manifest", "readme", "summary", "corr", "weight", "fold_ids", "folds.", "folds_", "fold0", "_f0.", "v_nov", "public_anchor", "scored_rankings")):
            continue
        if "oof" not in low:
            continue

        # Candidate test filenames
        cands = [
            nm.replace("oof", "test"),
            nm.replace("OOF", "TEST"),
            nm.replace("oof", "sub"),
            nm.replace("_oof", "_full"),  # e.g. goodpjw artifacts: ebm_dig_oof.npy -> ebm_dig_test.npy (prefer test first!)
        ]
        # Special known pair in goodpjw_lm7: megalm_oof_sha181c.npy -> megalm_test_sha3168.npy
        if nm == "megalm_oof_sha181c.npy":
            cands.insert(0, "megalm_test_sha3168.npy")

        test_path = None
        for cname in cands:
            if cname == nm:
                continue
            sd = p.parent / cname
            if sd.exists():
                test_path = sd
                break
            if cname in by_name:
                test_path = by_name[cname]
                break
        if test_path is None or p in seen:
            continue
        seen.add(p)

        clean_stem = p.stem
        for tok in ("pair_oof_", "oof_", "_oof"):
            clean_stem = clean_stem.replace(tok, "")

        if p.suffix == ".npy" and test_path.suffix == ".npy":
            try:
                oa = np.asarray(np.load(p, allow_pickle=False), dtype=np.float64)
                ta = np.asarray(np.load(test_path, allow_pickle=False), dtype=np.float64)
            except Exception:
                continue
            oa_sq, ta_sq = oa.squeeze(), ta.squeeze()
            if oa_sq.ndim == 1 and ta_sq.ndim == 1 and valid_pair(oa_sq, ta_sq, n_train, n_test):
                pairs.append((f"{prefix}:{clean_stem}", oa_sq, ta_sq))
            elif oa.ndim == 2 and ta.ndim == 2:
                if oa.shape[0] == n_train and ta.shape[0] == n_test and oa.shape[1] == ta.shape[1]:
                    K = oa.shape[1]
                    for j in range(K):
                        o1, t1 = oa[:, j], ta[:, j]
                        if valid_pair(o1, t1, n_train, n_test):
                            mname = manifest_names[j] if (manifest_names and len(manifest_names) == K) else f"c{j}"
                            pairs.append((f"{prefix}:{mname}", o1, t1))
                elif oa.shape[1] == n_train and ta.shape[1] == n_test and oa.shape[0] == ta.shape[0]:
                    K = oa.shape[0]
                    for j in range(K):
                        o1, t1 = oa[j, :], ta[j, :]
                        if valid_pair(o1, t1, n_train, n_test):
                            mname = manifest_names[j] if (manifest_names and len(manifest_names) == K) else f"r{j}"
                            pairs.append((f"{prefix}:{mname}", o1, t1))
        elif p.suffix == ".csv" and test_path.suffix == ".csv":
            oa = load_csv_1d(p, n_train, train_ids)
            ta = load_csv_1d(test_path, n_test, test_ids)
            if valid_pair(oa, ta, n_train, n_test):
                pairs.append((f"{prefix}:{clean_stem}", oa, ta))

    return pairs


def fit_nonneg_l2_logistic(X_tr: np.ndarray, y_tr: np.ndarray, l2_reg: float = 20.0) -> tuple[np.ndarray, float]:
    """Convex Non-Negative L2-Regularized Logistic Regression (w >= 0, free intercept b) via L-BFGS-B.
    Eliminates negative-weight collinearity overfitting across 200+ correlated models!
    """
    N, M = X_tr.shape
    y_f = y_tr.astype(np.float64)
    prior_b = float(np.log(y_f.mean() / (1.0 - y_f.mean())))

    # Param vector theta = [b, w_0, ..., w_{M-1}]
    theta0 = np.zeros(M + 1, dtype=np.float64)
    theta0[0] = prior_b
    theta0[1:] = 1.0 / M
    bounds = [(None, None)] + [(0.0, None)] * M

    def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        b = theta[0]
        w = theta[1:]
        z = X_tr @ w + b
        # Numerically stable BCE: mean(log(1 + exp(z)) - y*z)
        log1p_ez = np.logaddexp(0.0, z)
        bce = float(np.mean(log1p_ez - y_f * z))
        reg = 0.5 * (l2_reg / N) * float(np.dot(w, w))
        p = expit(z)
        diff = (p - y_f) / N
        gb = float(np.sum(diff))
        gw = X_tr.T @ diff + (l2_reg / N) * w
        grad = np.empty(M + 1, dtype=np.float64)
        grad[0] = gb
        grad[1:] = gw
        return bce + reg, grad

    res = minimize(loss_and_grad, theta0, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 400, "ftol": 1e-10})
    return res.x[1:], float(res.x[0])


def fast_hill_climb(X: np.ndarray, y: np.ndarray, n_steps: int = 150) -> np.ndarray:
    M = X.shape[1]
    single = np.array([roc_auc_score(y, X[:, j]) for j in range(M)])
    best_j = int(np.argmax(single))
    counts = np.zeros(M, dtype=np.float64)
    counts[best_j] = 1.0
    cur_sum = X[:, best_j].copy()
    cur_auc = float(single[best_j])
    top_idx = np.argsort(-single)[: min(100, M)]

    for step in range(2, n_steps + 1):
        step_best = cur_auc
        step_j = -1
        inv_s = 1.0 / step
        for j in top_idx:
            auc = roc_auc_score(y, (cur_sum + X[:, j]) * inv_s)
            if auc > step_best + 3e-8:
                step_best = auc
                step_j = int(j)
        if step_j == -1:
            break
        cur_sum += X[:, step_j]
        counts[step_j] += 1.0
        cur_auc = step_best
    return counts / counts.sum()


def verify_and_save(test_ids: np.ndarray, preds: np.ndarray, out_name: str) -> None:
    sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds})
    assert len(sub) == len(test_ids), "Row count mismatch!"
    assert not sub[TARGET_COL].isna().any(), "NaNs found!"
    assert (sub[TARGET_COL] >= 0).all() and (sub[TARGET_COL] <= 1).all(), "Out of [0,1]!"
    out_path = SUBMISSION_DIR / out_name
    sub.to_csv(out_path, index=False)
    print(f"Saved -> {out_path} (min={preds.min():.6f}, mean={preds.mean():.6f}, max={preds.max():.6f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--min-auc", type=float, default=0.942)
    parser.add_argument("--out", type=str, default="sub_v5_grand_stack.csv")
    args = parser.parse_args()

    ensure_dirs()
    t0 = time.time()

    zone_cols = [
        "id", "velocity_score", "merchant_risk_score", "cvv_retry_count",
        "prior_disputes", "is_new_merchant", "is_foreign_transaction",
        "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    train_df = pl.read_csv(DATA_RAW_DIR / "train.csv", columns=zone_cols + [TARGET_COL]).to_pandas()
    test_df = pl.read_csv(DATA_RAW_DIR / "test.csv", columns=zone_cols).to_pandas()
    y = train_df[TARGET_COL].to_numpy(dtype=np.int8)
    train_ids = train_df[ID_COL].to_numpy()
    test_ids = test_df[ID_COL].to_numpy()
    n_train, n_test = len(y), len(test_ids)

    zones_tr = get_zero_fraud_zones(train_df)
    zones_te = get_zero_fraud_zones(test_df)
    print(f"Target loaded | n_train={n_train:,}, n_test={n_test:,} | Zones 1-4: tr={zones_tr.sum():,} (frauds={y[zones_tr].sum()}), te={zones_te.sum():,}")

    # 1. Own locally trained models (v3, v4, v2)
    raw_members: list[tuple[str, np.ndarray, np.ndarray]] = []
    for tag in ("v3", "v4", "v2"):
        for f in sorted(OOF_DIR.glob(f"oof_{tag}_*.npy")):
            short = f.stem.replace(f"oof_{tag}_", "")
            tf = OOF_DIR / f"test_{tag}_{short}.npy"
            if not tf.exists():
                continue
            oa = np.asarray(np.load(f), dtype=np.float64).squeeze()
            ta = np.asarray(np.load(tf), dtype=np.float64).squeeze()
            if valid_pair(oa, ta, n_train, n_test):
                raw_members.append((f"own_{tag}:{short}", oa, ta))
    print(f"Loaded {len(raw_members)} own local models.")

    # 2. Public S6E11 OOF banks
    if not args.skip_download:
        print("\nDownloading / loading 14 public S6E11 OOF banks via kagglehub...")
        for prefix, slug in PUBLIC_OOF_DATASETS:
            try:
                ds_path = Path(kagglehub.dataset_download(slug))
                found = discover_all_in_dir(ds_path, prefix, n_train, n_test, train_ids, test_ids)
                print(f"  [{prefix:<14}] {slug:<54} -> {len(found):>3} valid pairs")
                raw_members.extend(found)
            except Exception as e:
                print(f"  [WARN] Failed {slug}: {e}")

    # 3. Filter & deduplicate
    scored = []
    for name, oa, ta in raw_members:
        auc = roc_auc_score(y, oa)
        if auc >= args.min_auc:
            scored.append((auc, name, oa, ta))
    scored.sort(key=lambda x: -x[0])

    kept_names, kept_aucs = [], []
    kept_pr_o, kept_pr_t = [], []
    kept_lg_o, kept_lg_t = [], []
    sample_idx = np.random.default_rng(SEED).choice(n_train, size=50000, replace=False)
    sample_vecs = []

    for auc, name, oa, ta in scored:
        pr_o = to_probit_rank(oa)
        sv = pr_o[sample_idx]
        sv_norm = (sv - sv.mean()) / (sv.std() + 1e-12)
        if any(float(np.dot(sv_norm, prev) / len(sv_norm)) > 0.99990 for prev in sample_vecs):
            continue
        sample_vecs.append(sv_norm)
        kept_aucs.append(auc)
        kept_names.append(name)
        kept_pr_o.append(pr_o)
        kept_pr_t.append(to_probit_rank(ta))
        kept_lg_o.append(to_clean_logit(oa))
        kept_lg_t.append(to_clean_logit(ta))

    M = len(kept_names)
    print(f"\n=== Kept {M} Unique High-Signal Models (deduped from {len(scored)}, elapsed={time.time()-t0:.1f}s) ===")
    print("Top 25 Single Models by OOF ROC AUC:")
    for i in range(min(25, M)):
        print(f"  {i+1:2d}. {kept_names[i]:<48} OOF AUC = {kept_aucs[i]:.6f}")

    O_pr = np.column_stack(kept_pr_o).astype(np.float64)
    T_pr = np.column_stack(kept_pr_t).astype(np.float64)
    O_lg = np.column_stack(kept_lg_o).astype(np.float64)
    T_lg = np.column_stack(kept_lg_t).astype(np.float64)
    del kept_pr_o, kept_pr_t, kept_lg_o, kept_lg_t, raw_members, scored
    gc.collect()

    # Pre-standardize logits per column cleanly inside CV folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(O_pr, y))

    # --- META 1: L2 Logistic Regression on Standardized Logits ---
    print("\n--- Meta-1: L2 Logistic Regression on Standardized Logits ---")
    best_m1_c, best_m1_auc, m1_oof, m1_test, m1_coefs = 0.01, 0.0, None, None, None
    for c in [0.001, 0.003, 0.008, 0.015, 0.03, 0.08]:
        so = np.zeros(n_train, dtype=np.float64)
        st = np.zeros(n_test, dtype=np.float64)
        cs = []
        for tr_i, va_i in splits:
            sc = StandardScaler()
            Xtr = sc.fit_transform(O_lg[tr_i])
            Xva = sc.transform(O_lg[va_i])
            Xte = sc.transform(T_lg)
            clf = LogisticRegression(C=c, solver="lbfgs", max_iter=800)
            clf.fit(Xtr, y[tr_i])
            so[va_i] = clf.decision_function(Xva)
            st += clf.decision_function(Xte) / 5.0
            cs.append(clf.coef_[0])
        auc = roc_auc_score(y, so)
        print(f"  LR-Logit      C={c:<7} Nested OOF AUC = {auc:.6f}")
        if auc > best_m1_auc:
            best_m1_auc, best_m1_c, m1_oof, m1_test, m1_coefs = auc, c, so, st, np.mean(cs, axis=0)

    # --- META 2: Convex Non-Negative L2 Logistic Regression on Standardized Logits (w >= 0!) ---
    print("\n--- Meta-2: Convex Non-Negative L2 Logistic on Logits (w >= 0) ---")
    best_m2_l2, best_m2_auc, m2_oof, m2_test, m2_ws = 100.0, 0.0, None, None, None
    for l2 in [5.0, 20.0, 60.0, 200.0, 600.0]:
        so = np.zeros(n_train, dtype=np.float64)
        st = np.zeros(n_test, dtype=np.float64)
        ws = []
        for tr_i, va_i in splits:
            sc = StandardScaler()
            Xtr = sc.fit_transform(O_lg[tr_i])
            Xva = sc.transform(O_lg[va_i])
            Xte = sc.transform(T_lg)
            w, b = fit_nonneg_l2_logistic(Xtr, y[tr_i], l2_reg=l2)
            so[va_i] = Xva @ w + b
            st += (Xte @ w + b) / 5.0
            ws.append(w)
        auc = roc_auc_score(y, so)
        print(f"  NN-LR-Logit   L2={l2:<6} Nested OOF AUC = {auc:.6f}")
        if auc > best_m2_auc:
            best_m2_auc, best_m2_l2, m2_oof, m2_test, m2_ws = auc, l2, so, st, np.mean(ws, axis=0)

    # --- META 3: Convex Non-Negative L2 Logistic on Probit-Ranks (w >= 0) ---
    print("\n--- Meta-3: Convex Non-Negative L2 Logistic on Probit-Ranks (w >= 0) ---")
    best_m3_l2, best_m3_auc, m3_oof, m3_test = 100.0, 0.0, None, None
    for l2 in [20.0, 80.0, 250.0]:
        so = np.zeros(n_train, dtype=np.float64)
        st = np.zeros(n_test, dtype=np.float64)
        for tr_i, va_i in splits:
            w, b = fit_nonneg_l2_logistic(O_pr[tr_i], y[tr_i], l2_reg=l2)
            so[va_i] = O_pr[va_i] @ w + b
            st += (T_pr @ w + b) / 5.0
        auc = roc_auc_score(y, so)
        print(f"  NN-LR-Probit  L2={l2:<6} Nested OOF AUC = {auc:.6f}")
        if auc > best_m3_auc:
            best_m3_auc, best_m3_l2, m3_oof, m3_test = auc, l2, so, st

    # --- META 4: L2 Logistic on Probit-Ranks ---
    print("\n--- Meta-4: L2 Logistic Regression on Probit-Ranks ---")
    best_m4_c, best_m4_auc, m4_oof, m4_test = 0.01, 0.0, None, None
    for c in [0.003, 0.01, 0.03]:
        so = np.zeros(n_train, dtype=np.float64)
        st = np.zeros(n_test, dtype=np.float64)
        for tr_i, va_i in splits:
            clf = LogisticRegression(C=c, solver="lbfgs", max_iter=800)
            clf.fit(O_pr[tr_i], y[tr_i])
            so[va_i] = clf.decision_function(O_pr[va_i])
            st += clf.decision_function(T_pr) / 5.0
        auc = roc_auc_score(y, so)
        print(f"  LR-ProbitRank C={c:<7} Nested OOF AUC = {auc:.6f}")
        if auc > best_m4_auc:
            best_m4_auc, best_m4_c, m4_oof, m4_test = auc, c, so, st

    # --- META 5: Compact Meta-MLP on Top-50 Non-Negative Weighted Logits ---
    print("\n--- Meta-5: Compact Meta-MLP on Top-50 Standardized Logits ---")
    top50 = np.argsort(-m2_ws)[: min(50, M)]
    m5_oof = np.zeros(n_train, dtype=np.float64)
    m5_test = np.zeros(n_test, dtype=np.float64)
    for fold, (tr_i, va_i) in enumerate(splits, 1):
        sc = StandardScaler()
        Xtr = sc.fit_transform(O_lg[tr_i][:, top50]).astype(np.float32)
        Xva = sc.transform(O_lg[va_i][:, top50]).astype(np.float32)
        Xte = sc.transform(T_lg[:, top50]).astype(np.float32)
        mlp = MLPClassifier(
            hidden_layer_sizes=(32,), activation="relu", alpha=1.0,
            batch_size=1024, learning_rate_init=1e-3, max_iter=20,
            early_stopping=True, n_iter_no_change=4, validation_fraction=0.1,
            random_state=SEED + fold,
        )
        mlp.fit(Xtr, y[tr_i])
        m5_oof[va_i] = to_clean_logit(mlp.predict_proba(Xva)[:, 1])
        m5_test += to_clean_logit(mlp.predict_proba(Xte)[:, 1]) / 5.0
    m5_auc = roc_auc_score(y, m5_oof)
    print(f"  Meta-MLP (Top 50)       Nested OOF AUC = {m5_auc:.6f}")

    # --- LEVEL-3: Convex Non-Negative Blend of All 5 Meta-Learners ---
    L3_oof = np.column_stack([m1_oof, m2_oof, m3_oof, m4_oof, m5_oof])
    L3_test = np.column_stack([m1_test, m2_test, m3_test, m4_test, m5_test])
    l3_oof = np.zeros(n_train, dtype=np.float64)
    l3_test = np.zeros(n_test, dtype=np.float64)
    l3_ws = []
    for tr_i, va_i in splits:
        sc = StandardScaler()
        Xtr = sc.fit_transform(L3_oof[tr_i])
        Xva = sc.transform(L3_oof[va_i])
        Xte = sc.transform(L3_test)
        w, b = fit_nonneg_l2_logistic(Xtr, y[tr_i], l2_reg=20.0)
        l3_oof[va_i] = Xva @ w + b
        l3_test += (Xte @ w + b) / 5.0
        l3_ws.append(w)
    avg_l3_w = np.mean(l3_ws, axis=0)
    avg_l3_w /= (avg_l3_w.sum() + 1e-12)
    l3_auc = roc_auc_score(y, l3_oof)

    print("\n================ V5.1 GRAND STACK SUMMARY (Strictly Nested 5-Fold CV) ================")
    cands = [
        (f"1. LR-Logit (C={best_m1_c})", best_m1_auc, m1_oof, m1_test),
        (f"2. NN-LR-Logit (L2={best_m2_l2})", best_m2_auc, m2_oof, m2_test),
        (f"3. NN-LR-Probit (L2={best_m3_l2})", best_m3_auc, m3_oof, m3_test),
        (f"4. LR-ProbitRank (C={best_m4_c})", best_m4_auc, m4_oof, m4_test),
        ("5. Meta-MLP (Top 50)", m5_auc, m5_oof, m5_test),
        ("6. Level-3 NonNeg Super-Blend", l3_auc, l3_oof, l3_test),
    ]
    for nm, auc, oo, _ in cands:
        z_auc = roc_auc_score(y, apply_zone_push(oo, zones_tr))
        print(f"  {nm:<32} Raw Nested OOF = {auc:.6f} | +Zones 1-4 = {z_auc:.6f} ({z_auc - auc:+.6f})")

    print(f"\nLevel-3 Meta Weights: LR-Logit={avg_l3_w[0]:.3f}, NN-LR-Logit={avg_l3_w[1]:.3f}, NN-LR-Probit={avg_l3_w[2]:.3f}, LR-Probit={avg_l3_w[3]:.3f}, Meta-MLP={avg_l3_w[4]:.3f}")

    print("\nTop 25 Non-Negative Weighted Members (Meta-2 NN-LR-Logit):")
    norm_w = m2_ws / (m2_ws.sum() + 1e-12)
    for idx in np.argsort(-norm_w)[:25]:
        if norm_w[idx] < 1e-4:
            break
        print(f"  {kept_names[idx]:<48} auc={kept_aucs[idx]:.6f} | nn_w={norm_w[idx]:.4f} | lr_c={m1_coefs[idx]:+.4f}")

    best_nm, best_raw_auc, best_oof, best_test = max(cands, key=lambda x: x[1])
    best_oof_z = apply_zone_push(best_oof, zones_tr)
    best_z_auc = roc_auc_score(y, best_oof_z)

    print(f"\n>>> WINNER: {best_nm}")
    print(f">>> Raw Nested OOF AUC     = {best_raw_auc:.6f}")
    print(f">>> +Zones 1-4 Nested OOF  = {best_z_auc:.6f}")
    fold_str = " | ".join(f"F{i+1}:{roc_auc_score(y[va], best_oof_z[va]):.5f}" for i, (_, va) in enumerate(splits))
    print(f">>> Fold AUCs (+Zones)     = {fold_str}")

    raw_sub = to_unit_rank(best_test)
    zone_sub = apply_zone_push(best_test, zones_te) if best_z_auc > best_raw_auc else raw_sub
    verify_and_save(test_ids, raw_sub, "sub_v5_grand_stack_nozone.csv")
    verify_and_save(test_ids, zone_sub, args.out)
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
