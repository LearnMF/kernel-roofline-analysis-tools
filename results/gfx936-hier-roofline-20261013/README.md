# gfx936 分层 roofline（kra 标定 + KDA G2 kernel 落点）

标定：`machines/gfx936-bw1000.json`（bw7，2026-10-10T02:48:21Z）。
带宽与算力都取**实测可达值**（最佳配置），理论值单列。

## 屋顶线

| 层级 | 可达带宽 | 理论带宽 |
|---|---|---|
| LDS | 15.24 TB/s | — |
| L2 | 6.10 TB/s | — |
| HBM | 1.34 TB/s | 1.84 TB/s |

| 计算 | 可达算力 |
|---|---|
| bf16 MMAC | 476.0 TFLOPS |
| tf32 MMAC | 237.7 TFLOPS |
| fp32 MMAC (16x16x8) | 61.0 TFLOPS |
| fp32 VALU FMA | 29.7 TFLOPS |

## 拐点（FLOP/Byte = 可达算力 ÷ 可达带宽）

| 计算 \ 存储层 | LDS | L2 | HBM | HBM（理论带宽） |
|---|---|---|---|---|
| bf16 MMAC | 31.2 | 78.0 | 355.4 | 258.2 |
| tf32 MMAC | 15.6 | 39.0 | 177.5 | 129.0 |
| fp32 MMAC (16x16x8) | 4.0 | 10.0 | 45.5 | 33.1 |
| fp32 VALU FMA | 1.9 | 4.9 | 22.2 | 16.1 |

## KDA 算子（fwd+bwd）在 HBM roofline 上的位置

| 形态 | 模型 FLOPs | HBM 流量 | kernel 时间 | 算术强度（模型 FLOPs） | 实际算力 | 实际 HBM 带宽 | 执行 MMAC 估计（上界） |
|---|---|---|---|---|---|---|---|
| 8K/H48 | 215.8 GFLOP | 11.22 GB | 15.42 ms | 19.2 FLOP/B | 14.0 TFLOPS | 728 GB/s | AI 93，67.8 TFLOPS |
| 8K/H12 | 54.0 GFLOP | 2.72 GB | 4.16 ms | 19.9 FLOP/B | 13.0 TFLOPS | 653 GB/s | AI 96，62.8 TFLOPS |

来源：Kimi K3 KDA fwd+bwd, hip_kda G2 (R5 tree), K=V=128, chunk 64, bf16；FLOPs 为分块 KDA 的模型 FLOPs（反向 = 2 × 前向）。

## 每个 kernel 的实测利用率（8K/H48，按时间排序）

| kernel | 时间 µs | HBM 流量 MB | HBM 带宽占可达 | VMEM 发射占饱和 | VALU 发射槽 | kra 分类 |
|---|---|---|---|---|---|---|
| p3_wy_s4b | 3094 | 1948 | 47% | 77% | 32% | B.memory.vmem_issue |
| wu | 1992 | 1690 | 63% | 41% | 47% | B.memory.hbm |
| p3_wy_s4a | 1705 | 1042 | 46% | 68% | 18% | B.memory.vmem_issue |
| fwd_prep_a | 1403 | 781 | 42% | 42% | 89% | B.compute.valu |
| bwd_tail | 1390 | 1261 | 68% | 71% | 56% | B.memory.hbm |
| dhu_v8b4 | 1304 | 928 | 53% | 79% | 31% | B.memory.vmem_issue |
| dav | 1089 | 684 | 47% | 101% | 18% | B.memory.vmem_issue |
| fwd_h_v8o4 | 1041 | 772 | 55% | 82% | 26% | B.memory.vmem_issue |
| fwd_h_v8o4 | 1005 | 688 | 51% | 80% | 33% | B.memory.vmem_issue |
| fwd_prep_b | 700 | 705 | 75% | 97% | 25% | B.memory.vmem_issue |
| gate | 339 | 302 | 66% | 35% | 97% | B.compute.valu |
| l2n_apply | 312 | 406 | 97% | 100% | 30% | B.memory.hbm |

## 复现

`python examples/kda/hier_roofline.py`（需要 matplotlib；读取 `machines/gfx936-bw1000.json` 与 `results/kda-gfx936-20261010-ceiling/g2_8192_{48,12}_prod.waterlevel.json`）。

说明：kra 的“执行 MMAC FLOPs”由 VALU 发射槽估计（slots − insts，按 8192 FLOP/条），超越函数（4 槽）和 fp32 MMAC 会使其偏大，只作为上界画出；不画逐 kernel 的 FLOP 落点，逐 kernel 改看右图的实测利用率。
