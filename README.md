# Kaggle Playground Series S6E11 — Predicting Credit Card Fraud

模块化表格竞赛工程，兼容 **MacBook Air (CPU)** 与 **RTX 5060 Ti 台式机 (CUDA GPU / 16GB RAM)**。

## 历史迭代战绩

| 版本 | 方案摘要 | 本地 Nested OOF AUC | Public LB | 排名 |
| :--- | :--- | :---: | :---: | :---: |
| **V1 Baseline** | Raw-24 + LightGBM (`leaves=63`, 5-Fold) | `0.945816` | `0.94637` | `#2354 / 3075` |
| **V2 Ensemble** | FE v3 (146 维：多尺度数字位 + 严格折内 OOF TE + 14 组交互) + 5-Fold Rank Blend | `0.950693` | `0.95033` | `#1421` |
| **V3 GPU Suite** | 10-Fold 10 模型异构全家桶（`super_gam`/`lr_gam`/`xgb_d2_gpu`/`lgb_d2`/`tabm_nn`/`cb_gpu`）+ Hill-Climb Log-Odds | `0.950932` | **`0.95051`** | **`#1403`** |
| **V4 Champion (冲刺 0.9525+)** | **3 变体 Target-First 数字 Token 因果语言模型（`tf_lm_rev`/`fwd`/`disc`，贝叶斯对数似然比）+ RealMLP-PBLD（高频周期嵌入 `sigma=5`）+ 100万行 LOO 无监督簇画像模型 + V3/V4 共 17 模型融合 + Zero-Fraud Zones 1-4** | 待 5060 Ti 跑出 | 待提交 | — |

---

## RTX 5060 Ti Windows 台式机一键开跑 V4

### 1. 拉取最新代码
```powershell
cd kaggle-s6e11
git pull
```
*(若尚未安装 CUDA 12.8 PyTorch：`uv pip install torch --index-url https://download.pytorch.org/whl/cu128`)*

### 2. 一键运行 V4 全部 GPU 模型 + 自动汇总 V3 & V4 共 17 模型融合
```powershell
uv run python src/run_v4_gpu.py
```

#### 运行亮点与贴心机制
- **按收益从高到低排序**：先跑收益最大的 `tf_lm_rev`（第 1 个 Target-First 因果语言模型），跑完立刻自动生成一份中间融合提交 `submissions/sub_v4_after_lm1.csv`，方便先交一次看大跃进！
- **逐折断点续跑**：LM 和 RealMLP 每跑完 1 折都会立刻把该折 OOF/Test 存入 `oof/ckpt_v4_*/`，万一中断重跑无缝接续。
- **自动复用你已跑完的 10 个 V3 OOF**：`src/blend_v4.py` 会把 `oof/oof_v3_*.npy` 和 `oof/oof_v4_*.npy` 全部 17 个模型一起做 10-Fold 严格 Nested-CV 融合，并自动校验 **Zero-Fraud Zones 1–4** 后处理增益。
- **最终主力提交**：`submissions/sub_v4_champion.csv`（另附不加后处理的 `submissions/sub_v4_raw_best.csv`）。
