# flash-train（GitHub fork `LearnMF/flash-train` 的 `feat/kda`，`8424b05`）对比 hip_kda P-88（2026-10-20，bw52）

## 测的是什么
- **flash-train**：`feat/kda` 分支基于上游 develop `548e0e9`，共 29 个提交：原样保留的 23 个 KDA 提交、Apache-2.0 许可证提交，以及本轮新增的 5 个提交。
  - `c77f5c4`：移植 P-77b 和 P-82/87。前向保存 qn/kn 和 h，C API 增加 qn、kn、h 三个端口。
  - `78fae93`：新增 `flash_train.torch.chunk_kda`，签名与 FLA 的 `chunk_kda` 相同。
  - `c07d8bd`：反向的 gate 临时空间放到 dq|dk 里；no_grad 下前向不再保存任何东西。
  - `c4b21f4`：workspace 内各条数据流按 HBM 通道错开。
  - `8424b05`：只有一条序列的 cu_seqlens 改走 dense kernel。
- **构建产物**：`libflash_train.so` 的 md5 为 `15f4152f69f3`；wheel 在 `bw52:/public/home/tanbo/xplat/flash_train-0.1.0-cp310-cp310-linux_x86_64.whl`。
- **hip_kda**：`g2_opt` `6bbdc76`（P-88），走 `hip_kda.fla.chunk_kda`，与 Megatron 的调用入口相同。
- **flash-train 侧的入口**：`flash_train.torch.chunk_kda`，参数与 Megatron-Bridge `kimi_k3_ops.kda` 的调用完全一样。

## 正确性
- **逐位门禁**（`scripts/ftgh_check.sh`）：flash-train 分别与 hip_kda P-88 和上一版 flash-train（`a1e0e35`）比对。
  - 覆盖 8 个形态：8192:12/48/96、32768:12/48，以及 3 种变长布局。
  - 每个形态测 3 种路线：默认（保存 qn/kn 和 h）、完整重算、keep-state。
  - 结果：24/24 全部 GATE PASS。
- **测试**：pytest 84 项通过，其中包括每种保存路线都与完整重算逐位一致、`chunk_kda` 按 K3 方式调用时与 `kimi_delta_attention` 逐位一致；ctest 181 项通过。

## 性能：算子级设备时间（`scripts/round_ftgh.sh`）
- **测法**：每个形态 5 个进程，加地址抖动（R20），两边逐进程交替测量。设备时间是 hipprof 统计的一次前向+反向 kernel 时间之和。
- **结果**：flash-train / hip_kda 的几何平均为 **1.013**，范围 0.993–1.041。完整表格见 `xplat_table.md`。
- **单进程 kern_cmp**：16K/24 时 flash-train 快 4%，8K/48 时快 3%。
- **结论**：剩下 ±4% 的差别来自张量地址的随机落位，两边都会受影响，不是 flash-train 的系统性损失。

| 改动 | 修复前 | 修复后 |
|---|---|---|
| 反向 dq\|dk\|dg 三条 fp32 写流相距正好是 32 MiB 的整数倍，挤在同一组 HBM 通道上 | flash-train 的 s4b 比 hip_kda 慢 1.2–1.3 倍（两边 ISA、kernel descriptor、launch 完全相同；TCP→TA 数据等待是 hip_kda 的 1.9 倍，指令数和请求数相同） | s4b 与 hip_kda 持平（8K/12：0.606 对 0.626 ms；16K/96：10.31 对 10.30 ms） |
| cu_seqlens=[0,T] 时走的是变长 kernel 实例 | 64K/12：递推慢 1.30 倍，prep 慢 1.11 倍，s4a 慢 1.16 倍 | 改走 dense 实例 |
| 保存 qn/kn 之后，dq/dk 在反向期间闲置 | 峰值显存比 hip_kda 多 7.4% | 峰值是 hip_kda 的 1.003–1.014 倍 |

`raw/round_dev_*_packed_cu.json` 是加 dense 快捷路径之前的一轮：几何平均 1.029。

## A800 FLA 复测（2026-10-21，a800-141）
- **环境**：容器 `tanbo_kda_nv210`（torch 2.10.0+cu128、triton 3.6.0、FLA 0.5.2），与 10-17 相同；用空闲的 GPU 0。
- **测法**：`op16_nv.sh`，每个形态顺序跑 3 个进程，与 10-17 的流程相同。数据在 `raw/a800_fla052_20261021.jsonl`，表格在 `a800_table_20261021.md`。
- **与 10-17 对比**：FLA 设备时间 10-21 / 10-17 的几何平均为 0.999（0.995–1.004）；3 个进程之间的差别不超过 0.6%。所以 A800 的基准是稳定的。

| 16 个形态的几何平均（A800 FLA 10-21 / 我们） | flash-train | hip_kda（同一轮） |
|---|---:|---:|
| 设备时间，对 FLA 默认 | **1.223**（1.14–1.40） | 1.239 |
| 设备时间，对 FLA 最快配置 | **1.102**（1.03–1.24） | 1.117 |
| 墙钟时间，对 FLA 默认 | **1.261**（1.13–1.70） | — |
| 墙钟时间，对 FLA 最快配置 | **1.140**（1.03–1.50） | — |

- **两列都是 16 个形态全赢**：设备时间和墙钟时间，flash-train 在 16 个形态上都快于 A800 的 FLA 默认配置和最快配置。
- **墙钟时间的测法**：`raw/bw_wall_ft_vs_hip_20261021.jsonl`，bw52 单卡，在同一进程里交替跑 g2、ft、ftc 三路，每个形态 3 个进程。flash-train 的墙钟时间是 hip_kda 的 0.920 倍，差别主要来自 hip_kda 的 FLA 适配层在 host 侧的开销。

## 性能：与 H20 上 FLA 的对比（flash-train 路线）
H20 的 FLA 数据沿用 `../kda-xplat-3way-20261017` 的结果（10-12 测）：h20-7 的 8 张卡仍被 sglang 占满，h20-2 连不上。下表 A800 一列也是 10-17 的旧数据，复测结果见上一节。

| 16 个形态的几何平均 | flash-train | hip_kda（10-17 表） |
|---|---:|---:|
| A800 FLA 默认 / 我们 | **1.224**（1.14–1.40），16 个形态全部更快 | 1.250 |
| A800 FLA 最快配置 / 我们 | **1.104**（1.03–1.23），16 个形态全部更快 | 1.128 |
| H20 FLA 默认 / 我们 | 0.857（0.78–0.97） | 0.875 |
| H20 FLA 最快配置 / 我们 | 0.764（0.71–0.85） | 0.781 |

flash-train 列与 10-17 表相差约 2%，原因是本轮 hip_kda 自己也比 10-17 慢约 1%（不同时段测量），外加上面 1.3% 的落位差。

## Megatron K3 训练切片（`scripts/k3_ft_bw52.sh`，`raw/k3_slice_ft_vs_hip.json`）
- **配置**：4 卡，EP4，L4 切片（3 层 KDA + 1 层 MLA），每次 12 步。
- **路由切换**：在 Megatron-Bridge `kimi_k3_ops.kda` 里加了 `KDA_K3_BACKEND=ft`，只把 import 换成 `flash_train.torch.chunk_kda`，其他参数不变。日志里 `[flash_train] K3 KDA call ...` 一行证明确实走了 flash-train。
- **KDA 时间**：取 rank0 第 11 步 profiler 中 KDA kernel 时间之和。

| 配置 | step 时间 ft/hip | 每步 KDA 时间 ft/hip | 峰值显存（rank0） |
|---|---:|---:|---|
| 8K TP4，不重算 | 1.003 | 1.003 | 相同 |
| 8K TP4，全层重算 | 0.895* | 0.978 | 相同 |
| 16K TP4，全层重算 | 1.006 | 0.948 | 相同 |
| 8K TP2，全层重算 | 0.931* | 0.964 | 相同 |

\* hip 那一次运行的非 KDA 部分本身偏慢，step 时间以 KDA 列为准。

- **重算形式下 flash-train 更快**：Megatron 重算的第一次前向在 no_grad 下运行，flash-train 不再写 h（递推只用 MODE 1）；hip_kda 仍然写（MODE 5 调用 6 次）。
- **loss**：同一条路线两次运行之间的差别本来就大（MoE 等部分不确定），flash-train 与 hip_kda 的 loss 曲线在这个范围内。
- **32K TP4 全层重算**：两边都 OOM，切片配置本身放不下。
