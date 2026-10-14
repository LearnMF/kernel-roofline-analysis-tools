# 任务说明：gfx936 构件成本库（KDA 第二阶段：构造式设计）

> 给执行 agent 的完整说明，读完再动手。本任务**只做微基准测量和建模**，不改任何生产代码，也不改主 agent 维护的文件（`examples/kda/decomp_eval.py` 等）。产出是一份机器可读的构件成本库，供主 agent 搭建“构造式成本模型”使用。

---

## 0. 背景：为什么需要这份库

KDA 算子（Kimi Delta Attention 训练前向 + 反向）在 Hygon gfx936（BW1000）上的第一阶段优化已经结束：
- **诊断**：体检卡 `results/kda-ceiling-cards-20261013/`，16 个形态 × 每个 kernel 的分层 roofline。
- **局部优化**：P-52、P-53、P-54a 已合入。
- **扰动式结构评估**：`examples/kda/decomp_eval.py`，从实测 kernel 出发做加减法。

体检卡把实测时间到算子第一性原理下限的差距拆成三层：

| 层 | 倍数（16 个形态） | 含义 |
|---|---:|---|
| 实测 / Σ现指令流下限 | 1.21–1.30 | kernel 没跑满自己的指令流 |
| Σ现指令流下限 / Σ接口下限 | 1.3–1.65 | 访问方式浪费（TA 放大 4–5 倍） |
| Σ接口下限 / 第一性原理下限 | 4.5–5.7 | 拆分方式带来的中间张量交接 |

**实测 / Σ接口下限 = 1.6–2.1**：理论上，即使不改 kernel 边界，也还有大约一倍的空间。但第一阶段只拿到了 1–3%。原因是方法上缺了一层：
- 现在的评估器是扰动模型，只能评估对**现有 kernel** 的小改动；
- **无法估算一个还不存在的 kernel**，比如换一种 tiling、布局或算法写法。

**第二阶段的做法**：搭一个构造式成本模型。给定一个设计（哪些运算放进一个 kernel、怎么分块和映射、中间张量用什么布局、chunk 多大、算法用哪种写法），就能从**基本构件的实测成本**组合出这个 kernel 的时间和资源占用，再在设计空间里搜索。
- 主 agent 负责：数据流 IR、构造式模型、搜索；
- **你负责：构件成本库**，也就是每种基本操作在 gfx936 上的实测成本，以及它们组合时的规律。

**模型的验收标准**：只用构件库，不用这些 kernel 自己的实测数据，“从零”推算出现有 11 个生产 kernel 的时间，误差在 ±20–30% 以内。所以构件库必须覆盖这些 kernel 实际用到的全部操作形式，并且自身经得起交叉验证（第 4 节）。

---

## 1. 边界与环境

- **机器**：只用 bw7，`ssh -J bw-zz tanbo@10.16.1.47`，容器 `tanbo_mega_k3`。主 agent 在 bw52 上工作，不要占用 bw52。
- **GPU**：
  - 先 `hy-smi` 确认空闲，持有 flock 锁 `/public/home/tanbo/codex_validation/locks/bw7/hcu_<N>.lock` 后再测，测完释放；
  - GPU 0–3、5 常驻别人的服务（显存占满、利用率 0），**不要用**；通常 4、6、7 可用；
  - 测量前后各记录一次 `hy-smi`，发现有别人的进程就整组重测。
- **生产树只读**：`/opt/kda_env/g2_r5`（P-54a，设备代码指纹 `90a91fdd5fdd`）、`/public/home/tanbo/g2_opt`、`ft_*` 都不能改。需要生产 helper 时：
  - 微基准用 `-I/opt/kda_env/g2_r5/csrc/g2` 引用头文件；
  - 或者拷贝一份到自己的目录。
- **Python 环境**：PYTHONPATH 必须以 `/opt/kda_env` 开头（否则 `import fla` 会段错误），并设 `KDA_SKIP_FLA_PRUNE=1`。
- **编译**：`/opt/dtk/bin/hipcc -O3 -std=c++17 --offload-arch=gfx936`。生产 kernel 的 ISA 用 `--cuda-device-only -S` 生成 `.s`，再用 `examples/kda/isa_loop_mix.py <file.s> --kernel <符号片段> [--whole]` 统计指令构成。
- **文件归属**：
  - 新代码放 `examples/kda/blocks/`，结果放 `results/kda-blocks-20261014/`，可以新增 `examples/kda/blocks_model.py`；
  - 不要修改仓库里其他已有文件。现有微基准可以**拷贝**到 `examples/kda/blocks/` 后再扩展。
- **git**：只 add 自己新增的路径（不要 `git add -A`）；**不要 amend 任何已有提交**；不要 push。完成后向用户汇报。
- **远程命令**：复杂命令写成脚本文件，scp 过去再执行，不要用多层嵌套引号（之前多次因此出错）。

---

## 2. 硬件事实（已标定，作为核对基准）

`machines/gfx936-bw1000.json` 和 `results/gfx936-hier-roofline-20261013/README.md` 中已有：
- 80 个 CU，每个 CU 4 个 SIMD，wave64，时钟约 1.5 GHz；
- 每 CU 64 KB LDS；每 wave 最多 256 个 VGPR，**没有 AGPR**；
- 实测：VGPR 在 200–256 时，每个 SIMD 仍能同时驻留 3 个 wave（R38，编译器的 occupancy 注释偏低）；
- 可达峰值：HBM 读 1339 GB/s、写 1100 GB/s；L2 读 6.1 TB/s；LDS 读 15.2 TB/s；BF16 MMAC 476 TFLOPS；FP32 VALU FMA 29.7 TFLOPS；
- MMAC 指令：`__builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c)`（v4bh × v4bh → f4）；s4b 还用到 fp32 的 16x16x8 形式（`mm8`），见 `g2_wy.cuh`；
- **MMAC 结果依赖 K 的顺序**（R44），但两个操作数对调后的结果与转置逐位一致（`mmac_swap_probe.hip`）；
- gfx936 **没有**硬件 bf16 转换指令（`v_cvt_pk_bf16_f32` 不支持），RNE 舍入需要 2 条 VALU。

---

## 3. 要测的构件（按优先级）

每一项都要给出：参数网格、测量方法、原始数据、拟合出的成本公式（或查表规则），以及适用范围。**每个微基准都要用 `isa_loop_mix.py` 核对**：编译出的循环体里，被测指令的条数与设计一致，其余指令没有被编译器删掉或挪走（R43）。必要时用 `asm volatile` 固定指令和依赖。

### P0-1 递推单步成本模型（最重要）

递推在 TP8 下决定了算子的第一性下限（16K/12 时串行链约 0.83 ms），所有候选设计（chunk 大小、切片宽度、不显式构造 W 的写法、是否保存状态）都要靠它定价。

- **已有数据**：`bound_microbench.hip` 中的 `k_step`、`k_step128`、`k_step32`，结论见 `results/kda-bound-20261012/README.md` 第 72 行以及 chunk 128 / chunk 32 两节：
  - 前向（含 o）每步 1.23 µs，只更新状态 1.00 µs（每 CU 1 个 CTA）；
  - 生产版每步 2.1–2.66 µs，是微基准的 1.7–2.2 倍，差距来自循环内的显存访问。
- **要做的**：写一个**参数化的合成单步 kernel**，每次循环迭代 = 一步，参数如下：

  | 参数 | 取值 |
  |---|---|
  | 每个 CTA 的 wave 数 W | 4、8 |
  | 每 CU 的 CTA 数 | 1、2、3（容量允许时） |
  | 状态块 | K=128 × BV，BV ∈ {32, 64}，fp32 存在寄存器中，按生产方式分布在各 wave 上 |
  | chunk 长度 C | 32、64、128 |
  | 每步的 GEMM 列表 | 每个 GEMM 指定 (M, N, K)、A/B 操作数来源（寄存器 / LDS / 全局）、输出去向（寄存器 / LDS），以及 GEMM 之间的依赖关系 |
  | 每步的 LDS 交换 | 次数、每次字节数、访问模式（无冲突 / 2 路冲突 / ds_read_u16 转置收集） |
  | 每步的 barrier 数 | — |
  | 每步的全局读 | 字节数、访问模式（取第 P0-2 节的模式）、预取距离（0 / 1 / 2 步） |
  | 每步的全局写 | 字节数、访问模式 |
  | 每步的逐元素运算 | VALU 条数、超越函数条数、bf16 RNE 转换次数 |

  - **4 个复刻点必须包含**：照生产源码（`g2_fwd_v8.cuh`、`g2_engine.cuh`）还原每步的 GEMM、LDS 交换、barrier、全局读写，分别是：
    - 前向 MODE 1（含 o）；
    - 重算 MODE 4；
    - dhu 8-wave（TP8 用）；
    - dhu 4-wave（TP2 用，生产中有 P-24 限流）。

    用 `isa_loop_mix.py` 核对复刻点与生产主循环的指令构成（MMAC、VMEM、LDS、waitcnt、barrier、VALU）是否接近。
  - **其余点**用部分析因设计覆盖参数空间，重点是对“改变 chunk 大小”“每步多一个 GEMM”“每步多一次 LDS 交换”“改变 BV”这几类变化的敏感度。
- **拟合**：给出单步时间的函数形式，并说明各项的物理含义。例如：
  ```
  t_step = 固定开销 + 依赖链上各段延迟之和 + max(0, 访存时间 − 可重叠部分)
  ```
  具体形式由你根据数据确定。分别给出每 CU 1 个 CTA（延迟受限）和多个 CTA（吞吐受限）两种区间的参数。

### P0-2 全局访存（VMEM）

- **已有数据**：`ta_microbench.hip` 测了 8 种读取模式（数据在 L2 / 显存中）的每条 wave 指令 TA 周期，结果表在 `results/kda-bound-20261012/README.md` 第 95 行起。
- **要补全**：

  | 维度 | 取值 |
  |---|---|
  | 每个 lane 的字节数 | 2、4、8、16 |
  | 一条指令覆盖的行数 × 每行连续字节数 | 1×(64·b)、2×、4×、8×、16×、32×、64× |
  | 数据来源 | 显存（工作集 ≥ 1 GB）、L2（4 MB）、L1/TCP（重复读同一块） |
  | 读 / 写 | **写要单独测**，包括部分缓存行写和 8 B 分散写（R10：写的代价与覆盖的地址范围成正比） |

- **每个点测三个量**：
  1. 吞吐：饱和时每条 wave 指令占用的 CU 周期数（TA 代价），以及每 CU 每周期的字节数；
  2. 延迟：低负载下从发出到数据可用的周期数，用 `s_memtime` 打点；
  3. 每条访存附带的地址计算条数：生产 kernel 里 `valu_int_addr → waitcnt` 是 SQTT 停顿的头号来源，要量化 64 位地址运算的成本，以及用 SGPR 基址或 buffer 指令能省掉多少。

### P0-3 LDS 与 barrier

- **单条 LDS 指令**：ds_read/ds_write 的 b16、b32、b64、b128，read2/write2，ds_read_u16 收集，ds_bpermute、DPP、swizzle。分别测 bank 冲突程度为 1、2、4、8、16 路时的吞吐（每 CU 每周期字节数）和延迟。已有 `lds_tr_microbench.hip`、`probe_ds_read_m32x16.hip` 可以参考。
- **复合操作“LDS 交换一步”**：W 个 wave 各写 X 字节 → `s_barrier` → 各读 X 字节。测时间随 X（1–32 KB）、W（4/8）、每 CU 的 CTA 数（1–3）的变化。这是递推每一步和各 kernel 内转置的基本单元。
- **barrier 本身**：4/8/16 个 wave 时，从最后一个 wave 到达到全部放行的延迟；以及 wave 到达时间不齐时的代价。

### P1-1 MMAC 依赖与交错

- 依赖链：同一个累加器连续做 k 次 MMAC（k = 1..16）的延迟；
- 每个 wave 交错 1/2/4/8 条独立链时的发射率，每 SIMD 1–4 个 wave；
- MMAC 与 VALU 交错时的互相影响；
- 操作数刚从 LDS / 全局读入时的等待开销；
- bf16 16x16x16 和 fp32 16x16x8 两种形式都要测。已有 `bound_microbench.hip` 中的 `k_mmac_tile` 可以参考。

### P1-2 VALU、超越函数与转换

在实际的指令混合环境里（不是孤立的单指令循环）测以下指令的吞吐和延迟：
- v_exp_f32、v_rcp_f32、v_log_f32；
- packed fp32（v_pk_fma_f32、v_pk_mul_f32）；
- v_add3_u32、v_bfe_u32、v_perm_b32、v_cvt_*；
- bf16 RNE 转换（2 条 VALU）；
- shuffle / readlane。

### P2 kernel 级开销

- 同一 stream 上前后相继、有依赖的 kernel 之间的 launch 间隙：可以直接从体检卡已有的生产 trace（`results/kda-ceiling-cards-20261013/raw/`）里读出 kernel 间隔；
- 尾部 wave 量化的实际代价：CTA 数不是 80 × 驻留数的整数倍时；
- 最小 kernel 的时间下限：空 kernel、只做一次读写的 kernel。

驻留数对时间的影响（`results/kda-decomp-calib-20261013/` 的 P1）和寄存器溢出曲线（同一目录的 P2）**已经有了，不用重测**，在库里引用即可。

---

## 4. 交叉验证（验收的核心）

构件库必须经得起下面两项验证。**验证时只能用构件库和 kernel 的结构信息**，也就是源码和 ISA 里的指令构成、访问模式、每步结构，不能用这些 kernel 的实测时间去拟合任何常数。

1. **递推单步模型 → 4 个生产递推实例**：
   - 根据前向 MODE 1、重算 MODE 4、dhu 8-wave、dhu 4-wave 每步的结构（GEMM、LDS 交换、barrier、全局读写，从源码和 ISA 得出），用模型预测每步时间，再乘以步数；
   - 与体检卡实测比较：16K/12（每 CU 1 个 CTA）和 16K/48（每 CU 多个 CTA）两个形态都要比；
   - **误差 ≤ ±20%**。
2. **VMEM / LDS 模型 → 3 个按 chunk 并行的 kernel**：
   - 选 dav、bwd_tail、wu，按源码和 ISA 统计每种访问模式的指令条数，用构件库预测每个 kernel 的 TA 忙碌周期和 LDS 周期；
   - 与体检卡的 PMC 计数（`ta_busy`、`lds_insts` 等字段，在 `cards.json` 里）比较；
   - **误差 ≤ ±20%**。
3. **健全性核对**：构件库里的饱和值与第 2 节的可达峰值一致，误差 ±5%。包括 HBM 读写带宽、L2 读带宽、LDS 带宽、MMAC 峰值。

哪一项达不到，就在 README 里写明差多少、可能的原因、还缺哪个构件。**不要为了凑数调参数**：每个常数都要能追溯到某次微基准实测。

---

## 5. 交付物

1. **代码**：`examples/kda/blocks/`，包括所有微基准 `.hip`、采集脚本、分析脚本，以及一条命令重跑全部的说明。
2. **构件库**：`results/kda-blocks-20261014/blocks.json`，格式如下（可以增加字段，不要删减）：
   ```json
   {
     "meta": {"machine": "gfx936-bw1000", "date": "...", "clock_ghz": 1.5, "gpu": "bw7 dev N", "fingerprint_ref": "90a91fdd5fdd"},
     "vmem_load":  [{"bytes_per_lane": 16, "rows": 16, "row_bytes": 64, "source": "hbm|l2|l1",
                     "ta_cycles_per_inst": 0.0, "latency_cycles": 0.0, "sat_bytes_per_cycle_per_cu": 0.0,
                     "addr_valu_per_inst": 0.0, "n": 0, "spread": 0.0}],
     "vmem_store": [{"...": "同上，外加 partial_line、scatter 字段"}],
     "lds": [{"op": "ds_read_b64", "conflict_way": 1, "cycles_per_inst": 0.0, "latency_cycles": 0.0, "bytes_per_cycle_per_cu": 0.0}],
     "lds_exchange": [{"waves": 4, "ctas_per_cu": 1, "bytes": 8192, "pattern": "conflict_free", "us": 0.0}],
     "barrier": [{"waves": 4, "skew_cycles": 0, "cycles": 0.0}],
     "mmac": [{"form": "bf16_16x16x16", "chain_len": 8, "indep_chains": 1, "waves_per_simd": 1, "cycles_per_mmac": 0.0}],
     "valu": [{"op": "v_exp_f32", "context": "...", "cycles_per_inst": 0.0, "latency_cycles": 0.0}],
     "recurrence_step": {
       "model": "函数形式的文字说明",
       "params": {"每个系数的名字": "值，含单位"},
       "regimes": {"1cta": {}, "multi_cta": {}},
       "raw": [{"config": {"W": 4, "BV": 32, "C": 64, "gemms": [], "lds_x": [], "barriers": 2, "gload": [], "gstore": [], "valu": 0}, "ctas_per_cu": 1, "us_per_step": 0.0, "spread": 0.0}],
       "replicas": [{"name": "fwd_mode1", "config": {}, "isa_match": {}, "us_per_step": 0.0}]
     },
     "kernel_overhead": {"launch_gap_us": 0.0, "min_kernel_us": 0.0, "tail_wave_rule": "..."},
     "references": {"occupancy_curves": "results/kda-decomp-calib-20261013/calib.json#p1",
                    "register_spill": "results/kda-decomp-calib-20261013/calib.json#p2"}
   }
   ```
3. **查询接口**：`examples/kda/blocks_model.py`，纯 Python，只依赖标准库和 json。提供：
   ```python
   vmem_cost(bytes_per_lane, rows, row_bytes, source, store=False) -> dict(ta_cycles, latency_cycles, addr_valu)
   lds_cost(op, conflict_way) -> dict(cycles, latency)
   lds_exchange_us(bytes, waves, ctas_per_cu, pattern) -> float
   barrier_cycles(waves, skew=0) -> float
   mmac_cycles(form, chain_len, indep_chains, waves_per_simd) -> float
   valu_cycles(op, context) -> float
   step_us(config, ctas_per_cu) -> float      # 递推单步模型
   ```
   不在测量范围内的查询要明确报错或返回带“外推”标记的结果，不能静默外推。
4. **交叉验证脚本和结果**：`examples/kda/blocks/validate.py`，输出第 4 节三项验证的表格（预测 / 实测 / 误差）。
5. **README**：`results/kda-blocks-20261014/README.md`，写清每个构件的方法、参数网格、主要结论、适用范围、交叉验证结果，以及不确定或测不准的地方。图放 `fig/`，要清楚可读：坐标轴有单位，图例完整。

---

## 6. 已知陷阱（来自 kra 的 playbook）

- **R43：微基准必须保持指令构成可控**。编译器会删掉结果没被使用的计算，或把读提前、合并。每个微基准都要用 ISA 核对，结果要写回内存（用不可预测的条件写，避免被删）。
- **R31/R32：用 inline asm 发访存时**，等待用 `__builtin_amdgcn_s_waitcnt`（写在 asm 里编译器看不见）；超过 64 位的 store 要处理 EXP_CNT 冒险。
- **R38：occupancy 要实测**，不要信编译器的注释。
- **R42：计数高不等于时间代价**。构件库给的是成本，组合时的重叠规律（哪些可以被掩盖）要通过复合实验（LDS 交换一步、递推单步）测出来，不能简单相加。
- **R45/R47：访问代价取决于模式**（每条指令覆盖几行、每行多少字节），不取决于指令条数；模式要从代码推出来。
- **R20/R41：显存位置和代码位置都会引起 2–8% 的波动**。每个点至少重复 5 次，报告离散度；离散度 > 3% 的点要重测或标注。
- **共享 GPU 有干扰**：测前测后都记录 `hy-smi`。
- **时钟**：gfx936 的时钟可能随功耗浮动。用 `s_memrealtime`（固定频率）和 `s_memtime`（shader 时钟）同时打点，确认换算关系；以 µs 为单位报告时注明用的是哪个时钟。

---

## 附录 A（2026-10-13 补充）：布局模型需要的访问模式，以及 bwd_tail 的核对重点

主 agent 的布局模型（`examples/kda/kda_layout.py`，结果在 `results/kda-layout-20261013/`）盘点了 s4b 和 bwd_tail 的全部访问。**P0-2 的参数网格至少要覆盖下面这些模式**，读写分开，数据来源用显存：

| 模式（每条 wave 指令） | 用在哪里 | 旧 TA 表 |
|---|---|---|
| 16 B/lane，1 行 × 1 KB | 原生布局整块读 | 有（21） |
| 8 B/lane，1 行 × 512 B | 原生片段读；tail 读一整行 fp32 | 有（22） |
| 4 B/lane，1 行 × 256 B | tail 读一整行 bf16 | 有（10） |
| **4 B/lane，1 行 × 256 B 的写** | tail 写 dg_raw、dq、dk（short2） | **缺** |
| 16 B/lane，4 行 × 256 B | 按行分块的读；dwT | 有（43） |
| 16 B/lane，16 行 × 64 B（读 / 写） | s4b 的 g、dA、dAqk 行读，dq/dk/dg 写 | 有（169 / 67） |
| 8 B/lane，16 行 × 32 B | s4b 读 kn、qn、do | 有（85） |
| **4 B/lane，4 行 × 64 B** | s4b 的 aqB/akB 转置读取 | **缺** |
| **4 B/lane，16 行 × 4 B（读 / 写）** | rstd、chunk 末行的 g、tail 的部分和 | **缺** |
| 4 B/lane，64 行 × 4 B | beta、db | 有（229） |

**特别注意“行距”**：生产张量是 [T, H, X] 行主序，相邻 token 的行之间相隔 H×X×dtype 字节（H=12 到 96，即 3 KB 到 24 KB 以上）。“1 行 × 256–512 B”这类指令，在这种行距下的实际代价，旧表没有测过。tail 实测平均每条 VMEM 指令约 30–40 个 TA 周期，旧表给的只有 10–22。请按 H=12、48、96 对应的行距分别测量。

**bwd_tail 的指令条数核对**：按源码和 ISA 推算，生产路径（门控 + l2norm）每 wave 约 148 条 VMEM 指令，PMC 是 194.8 条，相差 24%。第 4 节交叉验证 tail 时，请先用 PMC 把读、写分开计数（例如 SQ_INSTS_VMEM_RD、SQ_INSTS_VMEM_WR，按 gfx936 上可用的计数器），查清这个差距，再比较 TA。
