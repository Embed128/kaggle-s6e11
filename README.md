# Kaggle Playground Series S6E11 — Predicting Credit Card Fraud

模块化表格竞赛工程，兼容 **MacBook Air (CPU)** 与 **RTX 5060 Ti 台式机 (CUDA GPU / 16GB RAM)**。

## 历史迭代战绩

| 版本 | 方案摘要 | 本地 Nested OOF AUC | Public LB | 排名 |
| :--- | :--- | :---: | :---: | :---: |
| **V1 Baseline** | Raw-24 + LightGBM (`leaves=63`, 5-Fold) | `0.945816` | `0.94637` | `#2354 / 3075` |
| **V2 Ensemble** | FE v3 (146 维：多尺度数字位 + 严格折内 OOF TE + 14 组交互) + 5-Fold Rank Blend | `0.950693` | `0.95033` | `#1421` |
| **V3 GPU Suite** | 10-Fold 10 模型自训全家桶（`super_gam`/`lr_gam`/`xgb_d2_gpu`/`lgb_d2`/`tabm_nn`/`cb_gpu`）+ Hill-Climb Log-Odds | `0.950932` | `0.95051` | `#1403` |
| **V5 Grand Stack (209 异构流)** | 自训模型 + 14 大公开同折 OOF 库（4× 独立 Target-First LM + TabPFN-3.5 + RealMLP-BPE + TabM + EBM + Golem + GBDT/GLM，去重 209 模型）+ 严格 Nested 5-Fold Direct-AUC 剪枝元堆叠 + Zero-Fraud Zones 1–4 | **`0.952593`** *(V5.2)* / **`0.952547`** *(V5.3 精细融合)* | 待提交 | — |
| **V6 Dual-Submission（一攻一守双版本）** | **【守】** 100% 纯 Nested-CV 盾牌（`wide_lm_stack_201` + `sameer_glm10f` + `yekenot RealMLP PyTabKit` + `gb_s4` + 3× LM + `realmlp_bpe_v2`，零 Public LB 探针）<br>**【攻】** Nov 25 Top-4 前沿共识（`mzishran` / `dennis6886` / `goodpjw_AG` / `monya1`）Logit-Rank 聚合 + p99 截断 3% V6 Hybrid CV 稳健对冲 + Zero-Fraud Zones 1–4（0/16,730 泄露） | **`0.952574`** *(V6 纯 CV 盾牌)*<br>**`0.952570`** *(V6 Hybrid CV)* | 待提交 *(攻目标 `0.9568+`)* | — |

---

## 快速复现 V6 一攻一守双版本（约 2 分钟）

```powershell
cd kaggle-s6e11
git pull

# 自动拉取最新 Kernel 产出 + 复用本地同折 OOF，约 2 分钟生成 V6 四份提交：
uv run python src/build_v6_dual_shield.py
```

### 产出核心提交文件（位于 `submissions/`）
1. **`sub_v6_attack_top200.csv`（主推【攻】— 冲 Public Top 200）**：Top-4 前沿共识加权 Logit-Rank + p99 截断 3% CV 稳健对冲 + Zero-Fraud Zones 1–4（0 泄露）
2. **`sub_v6_shield_private_cv.csv`（主推【守】— 保 Private 终榜盾牌）**：100% 严格 Nested 5-Fold 纯 CV 集成（OOF **`0.952574`**，不含任何 Public Board 回放锚点，抗 Private LB 抖动）
3. **`sub_v6_attack_pure_frontier.csv`（备选纯攻）**：Top-4 前沿纯共识 + Zero-Fraud Zones 1–4（0% CV 对冲，极致刷 Public LB）
4. **`sub_v6_hybrid_cv_champion.csv`（备选混合 CV）**：融合 `wide_lm_stack_201` + `anchor_095370` + `RealMLP PyTabKit` + 3× LM（OOF **`0.952570`**）
5. **`sub_v5_2_grand_champion.csv`（V5 历史最高本地 OOF）**：全量 209 流严格 Nested-CV 最高分（OOF **`0.952593`**）

> **最终提交勾选建议**：Kaggle 允许勾选 **2 个 Final Submissions**，推荐直接勾选 **`sub_v6_attack_top200.csv`（攻）+ `sub_v6_shield_private_cv.csv`（守）**。
