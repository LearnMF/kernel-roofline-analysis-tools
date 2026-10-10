# kernel-roofline-analysis-tools (kra)

这个工具做两件事：判断 kernel 的瓶颈类型，判断它离性能上限还有多远。

它回答的问题是：一个算子（可能由多个 kernel 组成）在某个具体 shape 下，
- 时间花在哪里？
- 每个 kernel 属于哪类瓶颈，受哪一种硬件资源限制？
- 离上限还有多远？指的是哪一个上限：物理下限、现有拆法的下限，还是现有指令流的下限？
- 下一步该做什么：局部调优、减少工作量，还是重构？如果已经到顶，依据是什么？

文档：
- **操作流程** [docs/playbook.md](docs/playbook.md)：每轮怎样判断上限、怎样在局部和全局之间选方向、规则库、预测记分；
- 测量口径 [docs/methodology.md](docs/methodology.md)；
- 设计 [docs/design.md](docs/design.md)；
- 分类法 [kra/taxonomy.json](kra/taxonomy.json)；
- 阈值及其来源 [kra/thresholds.json](kra/thresholds.json)。

## 实践平台与适用范围

**本仓库的所有数值、阈值和结论，都是在下面这台机器上实测得到的**：

| 项 | 值 |
|---|---|
| GPU | 海光 HCU **gfx936**（`gfx936:sramecc+:xnack-`），驱动报告的名称为 **BW200 / UBB BW1000** |
| 节点 | bw7（经 bw-zz 跳板访问），容器 `tanbo_mega_k3` |
| 规格 | 80 CU，wave64，主频 1.5 GHz，HBM 4096 bit @ 1.8 GHz，L2 8 MiB，每 CU 64 KiB LDS |
| 软件 | DTK（`/opt/dtk`：hipcc、hipprof）；SQTT 采集用 XProf（ROCm 6.3.3 容器 `glm_bw7_xprof2`），分析用 XCompute 4.6.3 CLI |
| 标定结果 | [`machines/gfx936-bw1000.json`](machines/gfx936-bw1000.json)，原始输出在 `.raw.jsonl` |
| 实践样本 | Kimi K3 KDA 算子（hip_kda G2 实现），fwd+bwd，8K×12/48 heads 等 shape，见 `results/` |

**换机器、换工具链或换算子时**：
- 峰值和延迟必须重新跑 `kra calibrate`；
- 计数器语义（例如 `TA_TA_BUSY` 的归一化）要用已知工作量的 kernel 重新标定；
- `thresholds.json` 中标为 `hcu_calibrated: true` 的阈值，只在 gfx936 + KDA 上验证过。

## 判断上限要看哪些指标

每个 kernel 都要逐项检查下面每一种资源，取占用最高的那一项作为**主要受限因素**（`binding angle`）。每一项都给出：公式、在 gfx936 上实测的饱和值、可借鉴的公开方法或工具，以及我们实践下来的结论。

| # | 角度 | kra 的计算方式（按正在工作的 CU 计算） | gfx936 实测的饱和值 / 峰值 | 借鉴的公开方法与工具 | KDA 上的实践结论 |
|---|---|---|---|---|---|
| 1 | **HBM 带宽** | 实测读写字节（`TCC_EA_*` 计数器，公式已标定）÷ 可达带宽；小 grid 按 `hbm_read_vs_ctas` 曲线修正 | 读 1338 GB/s，**是理论 1843 GB/s 的 72.6%**；每个 CTA 约 42 GB/s | Roofline [1]、分层 Roofline [2]；带宽基准 BabelStream [5]；Nsight Compute 的 Speed Of Light / Memory Workload [8]；rocprofiler-compute（原 Omniperf）的 SOL 与 Memory Chart [9] | **分母必须用实测可达峰值**：“带宽到 85–90% 算到顶”如果按理论值算，是达不到的目标。l2n_apply 达到 96%，被判为到顶，对照实验 P-0 也证实了这一点 |
| 2 | **计算流水线（VALU 发射）** | `SQ_ACTIVE_INST_VALU` ÷ (周期 × CU 数)；发射槽计法：普通 VALU 占 1 个槽，MMAC 2 个，超越函数 4 个 | 在 MMAC、打包 FMA、exp2 的峰值下，实测都达到 0.97–1.00 | Nsight Compute 的 Compute Workload / Pipe Utilization [8]；Instruction Roofline [3] | prep_a 81%、head_gate 90%，判为流水线已满，只能减少 VALU 工作量。P-2 去掉 exp2 的保护分支后快了 1.20–1.26 倍 |
| 3 | **MFU / HFU（算力利用率）** | MFU = 算法的模型 FLOPs（op_spec 中的 `flops`；反向 = 2 × 前向）÷ (时间 × 理论峰值)；HFU = MMAC 单元实际执行的 FLOPs（含重算和 padding，来自 PMC）÷ (时间 × 峰值) | bf16 MMAC 理论峰值 491.5 TFLOPS，实测可达 476 TFLOPS（96.8%）；fp32 VALU 29.7 TFLOPS | MFU 和 HFU 的定义来自 PaLM 论文附录 B [6]；Roofline 中的算术强度 [1]；NVIDIA 的 GPU Performance Background 指南 [10] | KDA 在 8K/H12 下 **MFU 只有 2.6%，HFU 12.8%**，计算下限 113 µs，仅占 4160 µs 的 3%。说明它**不是计算受限的算子**，MFU 只用于汇报，不是优化目标。HFU 是 MFU 的 5 倍，差额来自分块重算和 padding |
| 4 | **向量访存发射（TA）** | `Σ TA_TA_BUSY`（16 个实例）÷ (GRBM 周期 × CU 数) | **0.92**（`vmem_issue` 微基准达到饱和时；与 hipprof 给出的“L1 cache unit is active”一致）。每个 CU 发射一条 `dwordx2` 读要 16 周期，`dwordx4` 要 18–23 周期；存储的成本约为每 512 B 27 周期 | Nsight Compute 的 L1TEX / LSU 利用率 [8]；rocprofiler-compute 的 vL1D / TA 指标 [9]；Instruction Roofline [3] | **这是这次实践中新增的、最关键的一项**：KDA 的递推、dav、s4 在 HBM 和计算都只有约 30% 时，被它卡在了 63–88%。按它的指引改用 16 B 读（L16），dhu 快了 1.23 倍。只看带宽和算力，会把这类 kernel 误判为“延迟受限” |
| 5 | **LDS** | bank 冲突周期 ÷ (周期 × CU 数)；用 `SQ_WAIT_INST_LDS` 所占比例判断冲突是否在关键路径上 | LDS 带宽 15.2 TB/s（理论值的 99%），延迟 64 周期 | Nsight Compute 的 Shared Memory 冲突统计 [8]；rocprofiler-compute 的 LDS 指标 [9] | **冲突率高不等于浪费了时间**：wu 的冲突占 19.6%，但 LDS 等待只有 3.6%，消除冲突后只快了 1.4%（P-1）。所以要同时看 LDS 等待占比 |
| 6 | **依赖链（延迟）** | 递推步数 × 每一步依赖链上的延迟之和（L0 实测：MMAC 依赖 44 周期、LDS 64 周期、barrier 52–76 周期）；按 op_spec 的 `serial` 字段计算；逐指令的暴露等待用 L3 `pcmap` 找 | — | Volkov 的延迟隐藏模型与 Little 定律 [7]；Nsight Compute 的 Warp State 停顿原因 / Source Counters [8]；Top-Down 分层归因的思路 [4] | 递推只算依赖的话只要 13–15 µs，实测却要 360–440 µs，依赖链只占 15–18%，**不是算法本身串行**。真正的停顿是保守的 waitcnt 和读扎堆，P-9/P-11 改掉之后快了 5–7% |
| 7 | **CU 占用（并行度）** | 正在工作的 CU ÷ 总 CU 数；占用率受 VGPR、LDS、wave 数限制 | 每 CU 最多 2560 线程，64 KiB LDS | CUDA / ROCm 的 occupancy 计算器，以及 Nsight 的 Achieved Occupancy [8]、rocprofiler-compute 的 Wavefront Occupancy [9] | H=12 时递推 kernel 只有 48 个 CTA，只用了 60% 的 CU。要得到更多并行度，必须改递推的结构（例如两级递推），这属于全局层面的改动 |
| 8 | **系统层（launch / 空泡）** | hipprof trace：空泡占比、gap 成因、藏在 kernel 时长里的派发开销 | 每次 launch 约 2 µs，HIP graph 约 1.67 µs | Nsight Systems、rocprofiler-systems（原 Omnitrace）的时间线 [9] | G2 的空泡占比为 0%；FLA 在 8K/H12 下有 11.8%，由 Python 侧提交慢导致 |

**算子级的四层分解**（`kra analyze` 中的 Ceiling verdict 一节）：物理下限 → 拆分代价 → 多余工作 → 执行效率损失。每层对应一类手段：硬件极限、重构、减少工作量、局部调优。最大的那一层决定下一步方向。思路借鉴了 Top-Down 的分层归因 [4]。用法见 [playbook §2](docs/playbook.md)。

### 参考与可借鉴的开源项目

| # | 名称 | 用途 / 我们借鉴的部分 |
|---|---|---|
| [1] | S. Williams, A. Waterman, D. Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, CACM 2009 | 算术强度、“受带宽还是受计算限制”的基本判断 |
| [2] | C. Yang, T. Kurth, S. Williams, *Hierarchical Roofline Analysis for GPUs*, CCPE 2020 | 对 L1、L2、HBM 各层分别画 roofline；kra 的 L0 也分层标定 |
| [3] | N. Ding, S. Williams, *An Instruction Roofline Model for GPUs*, PMBS 2019 | 按指令吞吐算的 roofline，对应 kra 的“VALU 发射”和“向量访存发射” |
| [4] | A. Yasin, *A Top-Down Method for Performance Analysis and Counters Architecture*, ISPASS 2014 | 分层归因的思路，对应 kra 的 A/B/C/D 分类和四层分解 |
| [5] | BabelStream（github.com/UoB-HPC/BabelStream）；Empirical Roofline Toolkit（LBNL）；mixbench（github.com/ekondis/mixbench） | 实测可达峰值的做法，对应 kra 的 L0 `calibrate` |
| [6] | A. Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, arXiv:2204.02311, 附录 B | MFU 与 HFU 的定义 |
| [7] | V. Volkov, *Understanding Latency Hiding on GPUs*, UC Berkeley PhD thesis, 2016 | 用 Little 定律分析延迟隐藏（在途请求数 = 吞吐 × 延迟），对应依赖链和并行度两个角度 |
| [8] | NVIDIA Nsight Compute（Speed Of Light、Memory / Compute Workload、Warp State、Source Counters、Occupancy） | SOL 的概念和停顿原因分类；判定阈值的初始值参考了 NV 的 triage 做法，来源见 `thresholds.json` |
| [9] | AMD rocprofiler-compute（原 Omniperf，github.com/ROCm/rocprofiler-compute）、rocprofiler-systems（原 Omnitrace） | 与 HCU 同源架构的 SOL、Roofline、Memory Chart、wavefront 统计。kra 的 TA 指标与其中 vL1D/TA 的思路一致 |
| [10] | NVIDIA, *GPU Performance Background User's Guide* | 每字节运算量、launch 与尾波（tail effect） |
| [11] | KernelBench（Ouyang et al., arXiv:2502.10517） | 以加速比为指标，评估 LLM 生成的 kernel；我们采用的报告口径是“对方耗时 / ours” |

**说明**：表中的公开方法大多是为 NVIDIA 或 AMD 设计的。kra 在 gfx936 上做了这几件事：
- 所有峰值都**实测**，不使用规格表数值；
- 计数器语义用已知工作量的 kernel **标定**，例如 `TA_TA_BUSY` 的归一化、`SQ_ACTIVE_INST_VALU` 的发射槽计法；
- 只有在 KDA 实验中**证实或证伪**过的规则，才写进规则库，见 [playbook §4](docs/playbook.md)。

## 状态

| 层 | 功能 | 状态 |
|---|---|---|
| L0 | `kra calibrate`：实测可达峰值（HBM/L2/LDS、MMAC 各 dtype、VALU、SFU、**向量访存读写的发射成本**）、延迟、launch 代价，并核对 ISA | ✅ |
| L1 | `kra timeline`：hipprof trace → 空泡占比、gap 成因、launch 下限、可信性检查 | ✅ |
| L2 | `kra pmc`：PMC 采集（计数器语义已标定，**包括 TA 利用率**）→ 各资源利用率 → A/B/C/D 分类 | ✅ |
| L3 | `python -m kra.sqtt.pcmap`：逐指令归因（关键 wave、按源码行的停顿）；`kra/isa/scan.py`：三条静态规则（带保护的数学函数展开、hot loop 中暴露的等待、可选分支导致的保守 waitcnt） | ✅ |
| L4 | `kra analyze`：先验下限 / 实现下限 / 融合下限、**MFU/HFU**、**四层分解与每个 launcher 的判定**；`python -m kra.model.fusion`：融合候选 | ✅ |
| L5 | 规则引擎（规则库已经写入 playbook）、第二个算子上的验证 | 进行中 |

## 用法

只依赖 Python 标准库（≥3.8）。L0 需要在目标机上运行，并且能调用 `hipcc`；分析部分可以在任何机器上运行。

```bash
# L0：在 HCU 节点（容器内）标定
HIP_VISIBLE_DEVICES=0 python -m kra calibrate --id gfx936-bw1000 --out machines/gfx936-bw1000.json

# L1：采集 trace 并分析（示例：KDA 算子，见 examples/kda）
hipprof --hip-trace --output-type 0 -o /tmp/tl/run  python3 examples/kda/kda_iter.py --T 8192 --H 12 --iters 5
python -m kra timeline /tmp/tl/run.json --marker spin_kernel --machine machines/gfx936-bw1000.json --reference-ms 4.84

# L2 + L4：launcher 接口捕获 + PMC 采集，然后生成水位表和上限评估
python3 examples/kda/kda_iter.py --T 8192 --H 12 --iters 2 --optrace /tmp/wl/g2.optrace.json
python -m kra pmc --out /tmp/wl/g2 --marker spin_kernel -- python3 examples/kda/kda_iter.py --T 8192 --H 12 --iters 1
python -m kra analyze --machine machines/gfx936-bw1000.json --pmc /tmp/wl/g2.pmc.json \
       --opspec examples/kda/op_spec.json --optrace /tmp/wl/g2.optrace.json --shape T=8192,H=12 --out /tmp/wl/g2
python -m kra.model.fusion --optrace /tmp/wl/g2.optrace.json --machine machines/gfx936-bw1000.json

# L3：逐指令分析（.perf 由 examples/kda/sqtt_capture.sh 采集；ISA 用 -gline-tables-only 编译）
python -m kra.sqtt.pcmap --perf X.perf --dispatch 91 --asm kernel_dbg.s --symbol <mangled kernel>

# 测试
python -m unittest discover -s tests
```

## 目录

```
kra/calibrate/   L0：microbench.hip（HIP 微基准）+ run.py（构建、运行、汇总、ISA 核对）
kra/timeline/    L1：hipprof trace 解析、窗口、空泡、gap 成因
kra/pmc/         L2：hipprof PMC 采集与解析（计数器语义见 docs/methodology.md §3、§4.1）
kra/sqtt/        L3：xcompute.py（停顿汇总）、pcmap.py（逐指令归因）
kra/isa/         L3：静态 ISA 规则
kra/opspec/      L4 输入：launcher 接口捕获（PyTorch 包装的任意算子通用）
kra/model/       L2 分类 + L4 下限模型、MFU/HFU、四层分解、融合候选、报告
machines/        各机器的标定结果（machine.json + 原始输出）
examples/kda/    KDA 算子的采集驱动、门禁、A/B 和实验补丁脚本
results/         分析结果、预先登记的预测与验证记录、上限评估
docs/            操作流程、测量口径、设计
tests/           单元测试（不需要 GPU）
```
