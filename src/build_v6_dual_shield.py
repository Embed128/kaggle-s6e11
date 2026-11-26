"""V6 Dual-Submission Champion Builder for Kaggle S6E11:
  1. SHIELD (Private LB Champion): Strictly Nested 5-Fold CV on elite heterogeneous streams
     (wide_lm_stack_nested + RealMLP PyTabKit + RealMLP PyTorch + 4 LMs + TabPFN-3.5 + RealMLP-BPE
     + TabM-BPE + sameer_glm10f + gb_bag5 + our own local models) + verified Zero-Fraud Zones 1-4.
     -> submissions/sub_v6_shield_private_cv.csv (ZERO Public LB probing!)
     -> submissions/sub_v6_hybrid_cv_champion.csv (includes anchor_095370)
  2. ATTACK (Public Top-200 Spear): Frontier Logit-Rank Consensus across latest Nov 25 Top-4
     public frontier streams (mzishran + dennis6886 + goodpjw_AG + monya1) with p99-clipped
     stabilization + 3% Private Shield variance hedge + verified Zero-Fraud Zones 1-4.
     -> submissions/sub_v6_attack_top200.csv
"""

import gc
import time
import warnings
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import polars as pl
from kagglesdk import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import DATA_RAW_DIR, ID_COL, OOF_DIR, PROJECT_ROOT, SEED, SUBMISSION_DIR, TARGET_COL, ensure_dirs
from train_grand_stack_v5 import (
    PUBLIC_OOF_DATASETS,
    apply_zone_push,
    discover_all_in_dir,
    fit_nonneg_l2_logistic,
    get_zero_fraud_zones,
    load_csv_1d,
    to_clean_logit,
    to_probit_rank,
    to_unit_rank,
    valid_pair,
    verify_and_save,
)
from train_v5_2_auc_stack import fast_auc

warnings.filterwarnings("ignore")

KERNEL_CACHE_DIR = PROJECT_ROOT / "data" / "kernel_outputs"


def download_kernel_file(client: KaggleClient, owner: str, slug: str, file_path: str, local_name: str) -> Path | None:
    KERNEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dst = KERNEL_CACHE_DIR / local_name
    if dst.exists() and dst.stat().st_size > 10000:
        print(f"  [cache] {local_name} ({dst.stat().st_size / 1e6:.2f} MB)")
        return dst
    try:
        req = ApiDownloadKernelOutputRequest()
        req.owner_slug = owner
        req.kernel_slug = slug
        req.file_path = file_path
        resp = client.kernels.kernels_api_client.download_kernel_output(req)
        with open(dst, "wb") as f:
            if hasattr(resp, "iter_content"):
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            elif hasattr(resp, "content"):
                f.write(resp.content)
            else:
                f.write(bytes(resp))
        print(f"  [fetch] {owner}/{slug}/{file_path} -> {local_name} ({dst.stat().st_size / 1e6:.2f} MB)")
        return dst
    except Exception as e:
        print(f"  [WARN] Failed {owner}/{slug}/{file_path}: {e}")
        if dst.exists():
            dst.unlink(missing_ok=True)
        return None


def optimize_simplex_direct_auc(X: np.ndarray, y: np.ndarray, w_init: np.ndarray) -> np.ndarray:
    """Fast, strictly monotonic direct-AUC simplex refinement starting from the strongest stream."""
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    rng = np.random.default_rng(SEED)
    idx = np.concatenate([pos, rng.choice(neg, size=min(80000, len(neg)), replace=False)])
    Xs = X[idx].astype(np.float32)
    ys = y[idx]

    # Start from the single highest-AUC stream (guarantees strictly >= best single stream!)
    single_aucs = np.array([fast_auc(ys, Xs[:, j]) for j in range(Xs.shape[1])])
    best_j = int(np.argmax(single_aucs))
    best_w = np.zeros(Xs.shape[1], dtype=np.float64)
    best_w[best_j] = 1.0
    best_val = float(single_aucs[best_j])

    # Also check w_init in case it's already higher
    w0 = np.maximum(w_init, 0.0)
    if w0.sum() > 1e-12:
        w0 = w0 / w0.sum()
        v0 = fast_auc(ys, Xs @ w0)
        if v0 > best_val:
            best_val = v0
            best_w = w0

    # Bounded coordinate ascent over fine step sizes (runs in ~2s per fold!)
    for step_size in (0.12, 0.05, 0.02, 0.008):
        for _ in range(2):
            improved = False
            for j in range(len(best_w)):
                for delta in (step_size, -step_size):
                    if delta < 0 and best_w[j] <= 1e-9:
                        continue
                    cand = best_w.copy()
                    cand[j] = max(0.0, cand[j] + delta)
                    if cand.sum() <= 1e-12:
                        continue
                    cand /= cand.sum()
                    v = fast_auc(ys, Xs @ cand)
                    if v > best_val + 3e-7:
                        best_val = v
                        best_w = cand
                        improved = True
            if not improved:
                break
    return best_w


def run_elite_nested_shield(
    O_lg: np.ndarray,
    T_lg: np.ndarray,
    O_pr: np.ndarray,
    T_pr: np.ndarray,
    y: np.ndarray,
    names: list[str],
    aucs: list[float],
    splits: list,
    zones_tr: np.ndarray,
    title: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    n_train, M = O_lg.shape
    n_test = T_lg.shape[0]
    print(f"\n================ {title} (M={M} elite streams) ================", flush=True)

    O_lg_std = (O_lg - O_lg.mean(axis=0)) / (O_lg.std(axis=0) + 1e-9)
    T_lg_std = (T_lg - O_lg.mean(axis=0)) / (O_lg.std(axis=0) + 1e-9)

    # Method 1: Full-Row Convex Non-Negative L2 Logistic on Standardized Logits (sweep L2)
    best_nn_l2, best_nn_auc, nn_oof, nn_test, nn_ws = 20.0, 0.0, None, None, None
    for l2 in (2.0, 10.0, 40.0, 120.0):
        so = np.zeros(n_train, dtype=np.float64)
        st = np.zeros(n_test, dtype=np.float64)
        ws = []
        for tr_i, va_i in splits:
            w, b = fit_nonneg_l2_logistic(O_lg_std[tr_i], y[tr_i], l2_reg=l2)
            so[va_i] = O_lg_std[va_i] @ w + b
            st += (T_lg_std @ w + b) / len(splits)
            ws.append(w)
        auc = roc_auc_score(y, so)
        print(f"  NonNeg-L2-Logit (L2={l2:<5})   Nested OOF = {auc:.6f}", flush=True)
        if auc > best_nn_auc:
            best_nn_auc, best_nn_l2, nn_oof, nn_test, nn_ws = auc, l2, so, st, ws

    # Method 2: Full-Row Free L2 Logistic on Standardized Logits
    best_lr_c, best_lr_auc, lr_oof, lr_test = 0.02, 0.0, None, None
    for c in (0.005, 0.02, 0.08, 0.25):
        so = np.zeros(n_train, dtype=np.float64)
        st = np.zeros(n_test, dtype=np.float64)
        for tr_i, va_i in splits:
            clf = LogisticRegression(C=c, solver="lbfgs", max_iter=800)
            clf.fit(O_lg_std[tr_i], y[tr_i])
            so[va_i] = clf.decision_function(O_lg_std[va_i])
            st += clf.decision_function(T_lg_std) / len(splits)
        auc = roc_auc_score(y, so)
        print(f"  Free-L2-Logit   (C={c:<6})   Nested OOF = {auc:.6f}", flush=True)
        if auc > best_lr_auc:
            best_lr_auc, best_lr_c, lr_oof, lr_test = auc, c, so, st

    # Method 3: Direct-AUC Simplex Refinement starting from NonNeg-L2 weights (Logits & ProbitRanks!)
    da_lg_oof = np.zeros(n_train, dtype=np.float64)
    da_pr_oof = np.zeros(n_train, dtype=np.float64)
    da_lg_ws, da_pr_ws = [], []
    for f_i, (tr_i, va_i) in enumerate(splits):
        w_init = nn_ws[f_i]
        w_lg = optimize_simplex_direct_auc(O_lg_std[tr_i], y[tr_i], w_init)
        w_pr = optimize_simplex_direct_auc(O_pr[tr_i], y[tr_i], w_init)
        da_lg_ws.append(w_lg)
        da_pr_ws.append(w_pr)
        da_lg_oof[va_i] = O_lg_std[va_i] @ w_lg
        da_pr_oof[va_i] = O_pr[va_i] @ w_pr
    avg_da_lg = np.mean(da_lg_ws, axis=0)
    avg_da_lg /= avg_da_lg.sum()
    da_lg_test = T_lg_std @ avg_da_lg

    avg_da_pr = np.mean(da_pr_ws, axis=0)
    avg_da_pr /= avg_da_pr.sum()
    da_pr_test = T_pr @ avg_da_pr

    # Method 4: Clean 50/50 Blend of Direct-AUC Logit + Free-L2-Logit
    hyb_oof = 0.5 * to_probit_rank(da_lg_oof) + 0.5 * to_probit_rank(lr_oof)
    hyb_test = 0.5 * to_probit_rank(da_lg_test) + 0.5 * to_probit_rank(lr_test)

    cands = [
        (f"1. NonNeg-L2-Logit (L2={best_nn_l2})", best_nn_auc, nn_oof, nn_test),
        (f"2. Free-L2-Logit (C={best_lr_c})", best_lr_auc, lr_oof, lr_test),
        ("3. Direct-AUC Simplex (Logit)", roc_auc_score(y, da_lg_oof), da_lg_oof, da_lg_test),
        ("4. Direct-AUC Simplex (Probit)", roc_auc_score(y, da_pr_oof), da_pr_oof, da_pr_test),
        ("5. 50/50 Blend (DirectAUC+LR)", roc_auc_score(y, hyb_oof), hyb_oof, hyb_test),
    ]
    print(f"\n--- Results [{title}] ---")
    for nm, auc, oo, _ in cands:
        za = roc_auc_score(y, apply_zone_push(oo, zones_tr))
        print(f"  {nm:<32} Raw Nested OOF = {auc:.6f} | +Zones 1-4 = {za:.6f} ({za - auc:+.6f})")

    print("  Top Elite Weights (Direct-AUC Simplex Logit & NonNeg-L2):")
    avg_nn = np.mean(nn_ws, axis=0)
    avg_nn /= (avg_nn.sum() + 1e-12)
    for idx in np.argsort(-(avg_da_lg + avg_nn))[:15]:
        if avg_da_lg[idx] + avg_nn[idx] < 1e-4:
            break
        print(f"    {names[idx]:<48} auc={aucs[idx]:.6f} | da_w={avg_da_lg[idx]:.4f} | nn_w={avg_nn[idx]:.4f}")

    best_nm, best_raw, best_oof, best_test = max(cands, key=lambda x: x[1])
    best_z = roc_auc_score(y, apply_zone_push(best_oof, zones_tr))
    fold_str = " | ".join(f"F{i+1}:{roc_auc_score(y[va], apply_zone_push(best_oof, zones_tr)[va]):.5f}" for i, (_, va) in enumerate(splits))
    print(f"  >>> WINNER [{title}]: {best_nm} | Raw={best_raw:.6f} | +Zones={best_z:.6f}")
    print(f"  >>> Folds (+Zones): {fold_str}", flush=True)
    return best_oof, best_test, best_raw, best_z


def main() -> None:
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

    # =========================================================================
    # STEP 1: Fetch RealMLP OOFs + Latest Top-Score Public Frontier Outputs
    # =========================================================================
    print("=== STEP 1: Fetching Kernel Outputs via Kaggle API ===", flush=True)
    k_files = {}
    with KaggleClient() as client:
        downloads = [
            ("realmlp_ptk_oof", "yekenot", "ps-s6-e11-realmlp-pytabkit", "oof_preds.csv"),
            ("realmlp_ptk_sub", "yekenot", "ps-s6-e11-realmlp-pytabkit", "submission.csv"),
            ("realmlp_torch_oof", "yekenot", "ps-s6-e11-realmlp-pytorch", "oof_preds.csv"),
            ("realmlp_torch_sub", "yekenot", "ps-s6-e11-realmlp-pytorch", "submission.csv"),
            ("frontier_goodpjw_ag", "goodpjw2008", "s6e11-board-model-realmlp-lb-0-95687", "submission.csv"),
            ("frontier_goodpjw_cvled", "goodpjw2008", "s6e11-board-model-realmlp-lb-0-95687", "submission_cv_led.csv"),
            ("frontier_dennis", "dennis6886", "s6e11-fraud-signal-rank-fusion-four-refinements", "submission.csv"),
            ("frontier_mzishran", "mzishran", "credit-card-fraud-clipped-rank-extrapolation", "submission.csv"),
            ("frontier_monya1", "monya1", "s6e11-zero-fraud-zones-and-rank-extrapolation", "submission.csv"),
            ("frontier_daoviet", "daoviet", "s6e11-on-the-journey-to-your-first-top-100", "submission.csv"),
            ("frontier_lm_ladder_cv", "goodpjw2008", "s6e11-lm-ladder-ladder-meets-stack-lb-0-95432", "submission_cv.csv"),
        ]
        for key, owner, slug, fp in downloads:
            k_files[key] = download_kernel_file(client, owner, slug, fp, f"{key}.csv")

    # =========================================================================
    # STEP 2: Assemble Elite CV Pool for V6 PRIVATE SHIELD (0.9527+ target!)
    # =========================================================================
    print("\n=== STEP 2: Assembling Elite CV Pool (Wide LM Stack + RealMLP + 4 LMs + TabPFN + Own) ===", flush=True)
    raw_members: list[tuple[str, np.ndarray, np.ndarray]] = []

    # 2A. Cached wide_lm_stack_nested & wide_stack_nested from goodpjw2008 artifacts (strictly nested 201-member stack!)
    art_root = Path(kagglehub.dataset_download("goodpjw2008/s6e11-fraud-series-artifacts"))
    for stem_o, stem_t, label in [
        ("wide_lm_stack_nested.npy", "wide_lm_stack_test.npy", "goodpjw_wide:wide_lm_stack_201"),
        ("wide_stack_nested.npy", "wide_stack_test.npy", "goodpjw_wide:wide_stack_200"),
    ]:
        po, pt = art_root / stem_o, art_root / stem_t
        if po.exists() and pt.exists():
            oa = np.asarray(np.load(po), dtype=np.float64).squeeze()
            ta = np.asarray(np.load(pt), dtype=np.float64).squeeze()
            if valid_pair(oa, ta, n_train, n_test):
                raw_members.append((label, oa, ta))
                print(f"  [ADDED] {label:<44} OOF AUC = {roc_auc_score(y, oa):.6f}")

    # 2B. yekenot RealMLP PyTabKit & RealMLP PyTorch (brand-new neural diversity!)
    for ok, tk, label in [
        ("realmlp_ptk_oof", "realmlp_ptk_sub", "yekenot:realmlp_pytabkit"),
        ("realmlp_torch_oof", "realmlp_torch_sub", "yekenot:realmlp_pytorch"),
    ]:
        if k_files.get(ok) and k_files.get(tk):
            oa = load_csv_1d(k_files[ok], n_train, train_ids)
            ta = load_csv_1d(k_files[tk], n_test, test_ids)
            if valid_pair(oa, ta, n_train, n_test):
                raw_members.append((label, oa, ta))
                print(f"  [ADDED] {label:<44} OOF AUC = {roc_auc_score(y, oa):.6f}")

    # 2C. Own locally trained models (v3, v4, v2)
    for tag in ("v3", "v4", "v2"):
        for f in sorted(OOF_DIR.glob(f"oof_{tag}_*.npy")):
            short = f.stem.replace(f"oof_{tag}_", "")
            tf = OOF_DIR / f"test_{tag}_{short}.npy"
            if tf.exists():
                oa = np.asarray(np.load(f), dtype=np.float64).squeeze()
                ta = np.asarray(np.load(tf), dtype=np.float64).squeeze()
                if valid_pair(oa, ta, n_train, n_test):
                    raw_members.append((f"own_{tag}:{short}", oa, ta))

    # 2D. Top public OOF banks (already cached!)
    for prefix, slug in PUBLIC_OOF_DATASETS:
        try:
            ds_path = Path(kagglehub.dataset_download(slug))
            raw_members.extend(discover_all_in_dir(ds_path, prefix, n_train, n_test, train_ids, test_ids))
        except Exception as e:
            print(f"  [WARN] {slug}: {e}")

    # Score, sort, and deduplicate
    scored = sorted(
        [(fast_auc(y, oa), nm, oa, ta) for nm, oa, ta in raw_members if fast_auc(y, oa) >= 0.945],
        key=lambda x: -x[0],
    )
    names, aucs, pr_o, pr_t, lg_o, lg_t = [], [], [], [], [], []
    sample_idx = np.random.default_rng(SEED).choice(n_train, size=50000, replace=False)
    sample_vecs = []
    for auc, nm, oa, ta in scored:
        po = to_probit_rank(oa)
        sv = po[sample_idx]
        sv_n = (sv - sv.mean()) / (sv.std() + 1e-12)
        if any(float(np.dot(sv_n, prev) / len(sv_n)) > 0.99990 for prev in sample_vecs):
            continue
        sample_vecs.append(sv_n)
        names.append(nm)
        aucs.append(auc)
        pr_o.append(po)
        pr_t.append(to_probit_rank(ta))
        lg_o.append(to_clean_logit(oa))
        lg_t.append(to_clean_logit(ta))

    O_pr = np.column_stack(pr_o)
    T_pr = np.column_stack(pr_t)
    O_lg = np.column_stack(lg_o)
    T_lg = np.column_stack(lg_t)
    del pr_o, pr_t, lg_o, lg_t, raw_members, scored
    gc.collect()

    print(f"\nTop 15 Single/Stack Streams in V6 Pool (Total unique={len(names)}):")
    for i in range(min(15, len(names))):
        print(f"  {i+1:2d}. {names[i]:<48} OOF AUC = {aucs[i]:.6f}")

    # Select focused Elite Pool (~18-22 genuinely diverse top streams: zero collinear weak-tree clutter!)
    elite_patterns = (
        "wide_lm_stack_201", "anchor_095370", "realmlp_pytabkit", "realmlp_pytorch",
        "mega_lm:lm", "masaya_gpt2", "gpt2_seed2026", "lm7_5fold",
        "sameer_glm10f", "weni_realmlp_bpe_v2", "weni_tabm_bpe",
        "tabpfn_100k", "v6_gb_xgb", "v6_gb_lgb", "gb_bag5", "gb_s4",
        "megayak_shallow_tree", "own_v3:", "own_v2:super_gam", "own_v2:lr_gam",
    )
    elite_idx = [i for i, nm in enumerate(names) if any(p in nm for p in elite_patterns)]
    if len(elite_idx) < 10:
        elite_idx = list(range(min(20, len(names))))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(O_pr, y))

    # SHIELD A: Pure CV Shield (EXCLUDES public-replayed anchor_095370 -> 100% pure Private LB armor!)
    pure_cv_idx = [i for i in elite_idx if "anchor" not in names[i].lower()]
    shield_oof, shield_test, shield_raw, shield_z = run_elite_nested_shield(
        O_lg[:, pure_cv_idx], T_lg[:, pure_cv_idx], O_pr[:, pure_cv_idx], T_pr[:, pure_cv_idx],
        y, [names[i] for i in pure_cv_idx], [aucs[i] for i in pure_cv_idx],
        splits, zones_tr, "V6 SHIELD: 100% PURE NESTED-CV PRIVATE CHAMPION",
    )

    # SHIELD B: Full Elite Hybrid CV (includes wide_lm_stack + RealMLP PyTabKit/PyTorch + anchor_095370)
    hyb_oof, hyb_test, hyb_raw, hyb_z = run_elite_nested_shield(
        O_lg[:, elite_idx], T_lg[:, elite_idx], O_pr[:, elite_idx], T_pr[:, elite_idx],
        y, [names[i] for i in elite_idx], [aucs[i] for i in elite_idx],
        splits, zones_tr, "V6 HYBRID CV CHAMPION (All Elite Streams)",
    )

    shield_final = apply_zone_push(shield_test, zones_te) if shield_z > shield_raw else to_unit_rank(shield_test)
    hyb_final = apply_zone_push(hyb_test, zones_te) if hyb_z > hyb_raw else to_unit_rank(hyb_test)

    verify_and_save(test_ids, shield_final, "sub_v6_shield_private_cv.csv")
    verify_and_save(test_ids, hyb_final, "sub_v6_hybrid_cv_champion.csv")

    # =========================================================================
    # STEP 3: Build V6 PUBLIC TOP-200 ATTACK (Frontier Consensus + Zones 1-4)
    # =========================================================================
    print("\n================ STEP 3: V6 PUBLIC TOP-200 ATTACK (Frontier Audit & Consensus) ================", flush=True)
    frontier_map = {}
    for k, label in [
        ("frontier_mzishran", "mzishran (0.95690+ clipped extrap)"),
        ("frontier_dennis", "dennis6886 (0.9569+ 4-refinements)"),
        ("frontier_monya1", "monya1 (zones + extrap)"),
        ("frontier_goodpjw_ag", "goodpjw2008 (0.95687 Step AG)"),
        ("frontier_daoviet", "daoviet (0.95603 Step AF)"),
        ("frontier_lm_ladder_cv", "goodpjw_cv (0.95379 CV-backed)"),
    ]:
        p = k_files.get(k)
        if p is not None and p.exists():
            arr = load_csv_1d(p, n_test, test_ids)
            if arr is not None and np.all(np.isfinite(arr)):
                u = to_unit_rank(arr)
                # Audit Zero-Fraud Zones 1-4 compliance (max rank among the 16,730 zero-fraud test rows!)
                zone_max_u = float(u[zones_te].max())
                zone_frac = float(zones_te.mean())  # ~0.0558 -> strictly pushed zones have max_u <= 0.0558!
                zone_leaks = int((u[zones_te] > zone_frac + 1e-4).sum())
                print(f"  [LOADED] {label:<38} | zone_max_rank={zone_max_u:.4f} | zone_leaks={zone_leaks:,}/{zones_te.sum():,}")
                frontier_map[k] = u

    # Print pairwise rank correlations among loaded frontier streams + our V6 Pure CV Shield
    keys_all = list(frontier_map.keys())
    print("\nPairwise Spearman Rank Correlations (vs V6 Pure CV Shield & each other):")
    u_shield = to_unit_rank(shield_final)
    for k in keys_all:
        c_sh = float(np.corrcoef(frontier_map[k], u_shield)[0, 1])
        c_ag = float(np.corrcoef(frontier_map[k], frontier_map["frontier_goodpjw_ag"])[0, 1]) if "frontier_goodpjw_ag" in frontier_map else 0.0
        print(f"  {k:<24} corr(V6_Shield)={c_sh:.6f} | corr(goodpjw_AG)={c_ag:.6f}")

    # Build disciplined Logit-Rank Consensus of Top-4 Frontier Streams
    # Weights emphasize the top independent frontier streams (mzishran + dennis + goodpjw_AG + monya1)
    def L(u: np.ndarray) -> np.ndarray:
        return logit(np.clip(u, 1e-6, 1.0 - 1e-6))

    attack_parts = []
    attack_weights = []
    for k, w in [
        ("frontier_mzishran", 0.34),
        ("frontier_dennis", 0.30),
        ("frontier_goodpjw_ag", 0.24),
        ("frontier_monya1", 0.12),
    ]:
        if k in frontier_map:
            attack_parts.append(L(frontier_map[k]))
            attack_weights.append(w)

    if attack_parts:
        aw = np.array(attack_weights)
        aw /= aw.sum()
        consensus_logit = sum(w * part for w, part in zip(aw, attack_parts))

        # Apply p99-clipped gentle 3% stabilization toward our V6 Hybrid CV Champion (reduces single-file probe variance!)
        gap_cv = L(to_unit_rank(hyb_final)) - consensus_logit
        clip_lim = float(np.quantile(np.abs(gap_cv), 0.99))
        stabilized_logit = consensus_logit + 0.03 * np.clip(gap_cv, -clip_lim, clip_lim)

        # Pure frontier consensus + Zones 1-4 AND stabilized Top-200 Attack + Zones 1-4
        attack_top200 = apply_zone_push(stabilized_logit, zones_te)
        attack_pure_frontier = apply_zone_push(consensus_logit, zones_te)

        verify_and_save(test_ids, attack_top200, "sub_v6_attack_top200.csv")
        verify_and_save(test_ids, attack_pure_frontier, "sub_v6_attack_pure_frontier.csv")

        print(f"\nAttack Consensus corr vs mzishran   : {np.corrcoef(attack_top200, frontier_map.get('frontier_mzishran', attack_top200))[0, 1]:.6f}")
        print(f"Attack Consensus corr vs dennis6886 : {np.corrcoef(attack_top200, frontier_map.get('frontier_dennis', attack_top200))[0, 1]:.6f}")
        print(f"Attack Consensus corr vs goodpjw_AG : {np.corrcoef(attack_top200, frontier_map.get('frontier_goodpjw_ag', attack_top200))[0, 1]:.6f}")
        print(f"Attack Consensus corr vs V6_Shield  : {np.corrcoef(attack_top200, u_shield)[0, 1]:.6f}")
    else:
        print("[WARN] No frontier kernel outputs downloaded; skipping Attack submission.")

    print(f"\nAll V6 Dual-Shield builds completed in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
