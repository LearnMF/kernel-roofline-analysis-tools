# Megatron K3 端到端：A800 + FLA，对比 BW + hip_kda G2（2026-10-10）

## 配置（两边完全相同）

**代码**：两边是同一份代码，都在共享的 `/public` 上：
- Megatron-LM-das，core 0.18.2；
- Megatron-Bridge 的 K3 模型；
- 同样的 `pretrain_gpt.py` 参数；
- mock data，micro batch 1，12 次迭代；
- 步长取第 4–9 次迭代的中位数；
- rank0 用 profiler 采集第 10–11 步，KDA 的 kernel 时间按 kernel 名分类求和。

**规模**：K3 的 L4 切片，即 3 层 KDA 加 1 层 MLA，16 个 expert，用 4 张卡、EP4。
- A800 只有 4 张空闲卡；
- 原报告用的是 L8 切片 8 张卡，在 4 张卡上两边都 OOM。

| 项 | A800（a800-141，A800 80GB PCIe） | BW（bw52，BW1000 64GB） |
|---|---|---|
| 镜像 | 官方 NGC `nvcr.io/nvidia/pytorch:26.06-py3`（torch 2.13a、triton 3.7.0、TE 2.16） | `megatron:0.18.2-latest`（torch 2.10 DAS、triton 3.6.0、TE 2.13 dtk2604） |
| KDA | FLA 0.5.2 `chunk_kda`（默认，以及 `disable_recompute` 版） | **hip_kda G2，`KDA_G2_KEEP_STATE=0`**：不保留状态，不用显存换速度。对照组为 FLA 0.5.2，带 DCU 的 autotune 剪枝 |
| 卡间互连 | PCIe，没有 NVLink | — |

## 结果：每步 3 层 KDA 的 kernel 时间合计（ms，rank0）

| 配置（每卡 24 个 head） | A800 FLA 默认 | A800 FLA 最快配置 | BW FLA 默认 | **BW ours** | **A800 FLA 默认 / ours** | BW FLA / ours |
|---|---:|---:|---:|---:|---:|---:|
| 8K，TP4，全层重算 | 13.13 | 12.31 | 21.58 | **14.28** | **0.92** | 1.51 |
| 8K，TP4，不重算 | 10.90 | 10.04 | 18.18 | **11.87** | **0.92** | 1.53 |
| 16K，TP4，全层重算 | OOM（见下） | OOM | 43.53 | **26.33** | — | 1.65 |
| 8K，TP2 / 32K，TP4 | 能跑（TP2：26.38） | 能跑 | OOM（64 GB） | OOM | — | — |

**结论**：
1. **在实际训练中，A800 + FLA 的 KDA kernel 时间比 BW + G2 少约 8%**，重算开或关都一样；FLA 最快配置少约 14%。这与算子级的设备时间对比一致：8K、24 个 head 时，A800 是 6.44 ms，BW G2 是 7.80 ms，比值 0.83。
2. **G2 在 BW 上比 FLA 快 1.51–1.65 倍**，与原报告的 1.49–1.54 倍一致，说明这次在 bw52 上重建的环境是可信的。
3. **整步时间不能用来比较 KDA**：A800 每步约 3.6 s，BW 约 1.0 s。但 A800 这台机器是 PCIe 互连，每步的通信 kernel 就占了 1.46 s，BW 只有 76 ms。整步时间反映的是机器互连的差异，不是 KDA 的差异。
4. **A800 在 16K 时 OOM**，原因在 MLA 层而不是 KDA：NGC 镜像里 TE 的 attention 对 MLA（q/k 192 维、v 128 维）退回到非融合实现，要一次申请 24 GiB 的 score 矩阵。BW 上 TE（dtk）会走 flash-attn。BW 在 TP2 和 32K 时是 64 GB 显存不够。

## 说明

- **loss**：不同平台之间不能直接比较，因为 mock data 和参数初始化用的是两边不同的 RNG 实现。同一平台内 FLA 和 G2 的 loss 走势一致。
- **算子级的 16 个形态**（TP1/2/4/8 × 8K–64K）正在两边测量，结果会写在 `op16.md`。
