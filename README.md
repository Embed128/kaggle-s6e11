# Kaggle Playground Series S6E11 — Predicting Credit Card Fraud

模块化表格竞赛工程，兼容 **MacBook Air (CPU)** 与 **RTX 5060 Ti 台式机 (CUDA GPU / 16GB RAM)**。

## 当前进度记录

| 版本 | 方案摘要 | 本地 OOF AUC | Public LB | 排名 |
| :--- | :--- | :---: | :---: | :---: |
| **V1 Baseline** | Raw-24 + LightGBM (`leaves=63`, 5-Fold) | `0.945816` | `0.94637` | `#2354 / 3075` |
| **V2 Ensemble** | FE v3 (146 维：多尺度数字/小数位 + 严格折内 OOF TE + 14 组交互 TE) + 5-Fold Rank Blend | `0.950693` | **`0.95033`** | **`#1421`** |
| **V3 GPU Suite** | 10-Fold 10 模型真异构全家桶（`lr_gam` + `super_gam` + `lgb_d1/d2/d3_et` + **CUDA XGBoost d2/d3** + **CUDA CatBoost d3/d4** + **PyTorch CUDA TabM BatchEnsemble NN**）+ 严格 Nested-CV 融合 | 待 5060 Ti 跑出 | 待提交 | — |

---

## RTX 5060 Ti Windows 台式机一键开跑 V3

### 1. 拉取代码 & 安装基础依赖
```powershell
git clone https://github.com/Embed128/kaggle-s6e11.git
cd kaggle-s6e11
# 如果已有仓库，改用：git pull

uv sync
```

### 2. （推荐）安装 PyTorch CUDA 12.8+ 以启用 5060 Ti TabM 深度网络
RTX 5060 Ti 为 Blackwell (`sm_120`) 架构，安装 `cu128` 版 PyTorch：
```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```
*(如果暂不装 PyTorch，`run_v3_gpu.py` 也会自动优雅跳过 NN 并跑完其余 9 个 GPU/CPU 模型。)*

### 3. 配置 Kaggle API Token 并下载数据（首次执行一次）
```powershell
mkdir $HOME\.kaggle -Force
Set-Content -Path $HOME\.kaggle\access_token -Value "你的KGAT_token" -NoNewline
uv run python src/download_data.py
```

### 4. 一键运行 V3 全部 10-Fold GPU 训练 + 自动融合
```powershell
uv run python src/run_v3_gpu.py
```
- **断点续跑**：每个模型跑完都会把 OOF 和测试预测存入 `oof/oof_v3_<model>.npy`，万一中途中断，再次运行 `uv run python src/run_v3_gpu.py` 会自动跳过已完成的模型！
- **16GB 内存保护**：全程 `Polars + float32` 并在每折后主动 `gc.collect()` 与清理显存。
- **产出文件**（在 `submissions/` 下）：
  - `sub_v3_gpu_ensemble.csv`（Nested-CV 最优融合主力提交）
  - `sub_v3_logodds.csv`（概率校准版 Log-Odds 融合）
  - `sub_v3_rank.csv`（排序融合版）

跑完后把终端最后 `=== Single Model OOF ROC AUC ===` 到 `>>> Selected Best Ensemble` 之间的输出贴回给架构端，即可继续迭代 V4！
