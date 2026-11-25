# Kaggle Playground Series S6E11 — Predicting Credit Card Fraud

模块化表格竞赛工程，兼容 **MacBook Air (CPU)** 与 **RTX 5060 Ti 台式机 (CUDA GPU / 16GB RAM)**。

## 当前进度记录

| 版本 | 方案摘要 | 本地 5-Fold OOF AUC | Public LB | 排名 |
| :--- | :--- | :---: | :---: | :---: |
| **V1 Baseline** | Raw-24 + LightGBM (`leaves=63`) | `0.945816` | `0.94637` | `#2354 / 3075` |
| **V2 Ensemble** | FE v3 (146 维：多尺度数字/小数位 + 严格折内贝叶斯平滑 OOF TE + 14 组强交互 TE) + 8 模型评估 + Rank Blend (`lr_gam` + `super_gam` + `xgb_d2` + `lgb_d2`) | `0.950693` | **`0.95033`** | **`#1421`** |

## 目录结构

```text
kaggle-s6e11/
├── pyproject.toml          # uv 依赖配置
├── uv.lock                 # 精确依赖锁文件
├── data/raw/               # 原始赛题 CSV（git ignored）
├── oof/                    # OOF 与测试集预测 .npy（git ignored）
├── submissions/            # 生成的提交文件 .csv（git ignored）
└── src/
    ├── config.py           # 路径与全局超参
    ├── download_data.py    # 通过 kagglehub 下载并整理数据
    ├── eda.py              # Polars 高速探索性数据分析
    ├── features.py         # FE v3（数字/小数位 + 严格防泄露折内 OOF TE）
    ├── train_lgb.py        # V1 LightGBM Baseline
    ├── train_v2.py         # 多模型统一训练器（支持 --device cpu|cuda 与 --n-folds）
    ├── train_super_gam.py  # 多尺度平滑 Super GAM + 2 层神经网络 MLP
    ├── train_sparse_gam.py # 62k 维稀疏 One-Hot GAM
    └── blend.py            # 严格 Nested-CV 融合优化（Log-Odds / Stacking / Rank）
```

## 在 5060 Ti Windows 台式机上快速开跑

### 1. 环境准备
```powershell
# 克隆仓库后进入目录
uv sync
```
配置 Kaggle API Token（写入 `%USERPROFILE%\.kaggle\access_token` 或设置环境变量 `KAGGLE_API_TOKEN`），然后下载数据：
```powershell
uv run python src/download_data.py
```

### 2. 开启 CUDA GPU 跑全量模型（推荐 10-Fold）
```powershell
# 跑全部 6 个模型（XGBoost & CatBoost 自动走 5060 Ti CUDA GPU）
uv run python src/train_v2.py --models lr_gam,lgb_d1,lgb_d2,xgb_d1,xgb_d2,cb_d2 --device cuda --n-folds 10

# 跑 Super GAM + MLP（5-Fold，如需 10-Fold 可在 config.py 将 N_FOLDS 改为 10）
uv run python src/train_super_gam.py

# 自动融合并生成提交文件
uv run python src/blend.py --tag f10 --out sub_v3_f10_ensemble.csv
```
*(若只跑 5-Fold，把 `--n-folds 10` 去掉、最后 `--tag` 改为 `v2` 即可。)*
