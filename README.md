# Kaggle Playground Series S6E11 — Predicting Credit Card Fraud

模块化表格竞赛工程，兼容 **MacBook Air (CPU)** 与 **RTX 5060 Ti 台式机 (CUDA GPU / 16GB RAM)**。

## 历史迭代战绩

| 版本 | 方案摘要 | 本地 Nested OOF AUC | Public LB | 排名 |
| :--- | :--- | :---: | :---: | :---: |
| **V1 Baseline** | Raw-24 + LightGBM (`leaves=63`, 5-Fold) | `0.945816` | `0.94637` | `#2354 / 3075` |
| **V2 Ensemble** | FE v3 (146 维：多尺度数字位 + 严格折内 OOF TE + 14 组交互) + 5-Fold Rank Blend | `0.950693` | `0.95033` | `#1421` |
| **V3 GPU Suite** | 10-Fold 10 模型自训全家桶（`super_gam`/`lr_gam`/`xgb_d2_gpu`/`lgb_d2`/`tabm_nn`/`cb_gpu`）+ Hill-Climb Log-Odds | `0.950932` | `0.95051` | `#1403` |
| **V5 Grand Stack (209 异构流)** | 自训模型 + 14 大公开同折 OOF 库（4× 独立 Target-First LM + TabPFN-3.5 + RealMLP-BPE + TabM + EBM + Golem + GBDT/GLM，去重 209 模型）+ 严格 Nested 5-Fold Direct-AUC 剪枝元堆叠 + Zero-Fraud Zones 1–4 | **`0.952593`** *(V5.2)* / **`0.952547`** *(V5.3 精细融合)* | 待提交 | — |

---

## 快速复现 V5（含 5060 Ti 10-Fold V3 模型合体）

```powershell
cd kaggle-s6e11
git pull

# 自动下载/复用 14 个公共同折 OOF 库 + 本机全部 V3/V4 自训 OOF，2 分钟生成最终冲榜提交：
uv run python src/train_v5_3_fine_auc.py
uv run python src/train_v5_2_auc_stack.py
```

### 产出核心提交文件（位于 `submissions/`）
1. **`sub_v5_2_grand_champion.csv`**：全量 209 流严格 Nested-CV 最高分（OOF **`0.952593`**）
2. **`sub_v5_3_champion.csv`**：全量 209 流精细多模型混合（`anchor + sameer_glm10f + gb_s4 + lm7 + realmlp_bpe_v2 + gpt2 + tabm`，OOF **`0.952547`**）
3. **`sub_v5_2_pure_shield.csv`**：纯单模底座（不含任何预融合 anchor，OOF **`0.952104`**，Private LB 抗抖动备用盾牌）
