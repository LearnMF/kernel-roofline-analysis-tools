# 任务说明：KDA 算子“逐 kernel × 逐形态”见顶体检卡（gfx936 / BW1000）

> 给执行 agent 的完整说明。读完再动手。本任务**只测量和分析，不改任何生产代码**。
> 主 agent 同时在 bw52 上做优化实验；你在 bw7 上工作，两边互不干扰。

---

## 0. 目标与边界

**目标**：对 Kimi K3 的 KDA 训练算子（前向 + 反向，hip_kda G2 实现），在 16 个形态上，给每个 kernel 出一张“体检卡”，回答三个问题：

1. 它被哪一类资源限制（带宽、发射还是延迟链），利用率多少，离对应的屋顶还有多远；
2. 在这类资源上，实际执行的工作量（字节、指令）是必需量的几倍；
3. 结论：已见顶，还是有缺口；如果有缺口，属于哪一类（见第 5 节），潜在可省多少 µs。

最后汇总到算子级：各 kernel 下限之和、实测时间、第一性原理算子下限，三者对比，得出“实现缺口”和“结构缺口”各占多少。

**边界（必须遵守）**：

- **只读**：不得修改 `/opt/kda_env/g2_r5`（bw7 生产树）、`/public/home/tanbo/g2_opt`（bw52 生产树）、`/public/home/tanbo/ft_*`（flash-train）中的任何文件，也不得在其中重新构建。
- **冻结测量对象**：只测冻结副本（见 2.2），以免主 agent 合入新改动后测量对象中途改变。
- **只用 bw7**：不要使用 bw52 的 GPU，bw52 留给主 agent 做实验。
- **不动别人的进程**：bw7 上有其他用户的容器和进程，不得 kill。每次测量前用 `hy-smi` 确认目标 GPU 空闲，并把当时的状态记入原始数据。
- **只新增文件**，产出写到 kra 仓库的两个目录（见第 7 节）：
  - `examples/kda/ceiling/`
  - `results/kda-ceiling-cards-20261013/`
- **只在本地提交**，不要 push。
- 不确定或测不准的指标，要明确标注“不可靠”，不要硬给结论。

---

## 1. 背景（你需要知道的最少信息）

- **硬件**：Hygon BW1000（gfx936），80 个 CU，每 CU 4 个 SIMD，wave64，主频约 1.5 GHz。
  - 每个 CU 64 KB LDS；每个 wave 最多 256 个 VGPR。
  - L2 8 MB；64 GB HBM。
- **算子**：KDA 的分块 delta rule 训练算子，K=V=128，chunk=64，bf16，门控和 l2norm 都在 kernel 内完成。
  - 前向：prep_a → prep_b → 递推（前向，输出 o）；
  - 反向：l2n_apply、gate、wu、to_gn + exp2、递推（反向重算，输出 bN 和 Vn）、dav、dhu、s4a、s4b、bwd_tail、bwd_tail_reduce。
- **已有结论**（供参考，需要你用数据验证或推翻）：
  - 没有 kernel 碰到带宽屋顶；多数 kernel 受 VMEM/TA 发射、VALU 发射或递推延迟链限制；
  - TP8 时三个递推 kernel 受延迟链限制，TP2 时更多受 TA 和带宽限制；
  - 按现有拆分方式，实际执行的字节约是算法必需量的 6.5 倍（16K/TP2），见 `results/kda-excess-work-20261013/README.md`。
- **方法论文档**（必读）：
  - `docs/methodology.md`：标定、计数器语义、分类规则；
  - `docs/playbook.md`：规则 R1–R41，尤其是 R20、R36、R38、R41；
  - `results/kda-bound-20261012/README.md`：第一性原理下限和 TA 代价表；
  - `results/gfx936-hier-roofline-20261013/README.md`：上一版分层 roofline，8K 两个形态，**其中 MMAC 拆分有误，见 4.3**。

---

## 2. 环境与访问

### 2.1 登录与容器

```bash
# bw7 的主机名已变成 bw47；~/.ssh/config 里的 bw7 别名在跳板机上解析不了，必须用 IP
ssh -J bw-zz tanbo@10.16.1.47
# 容器（已在运行）
docker exec -it tanbo_mega_k3 bash
```

在容器里运行 KDA 的标准环境变量：

```bash
export KDA_G2=1 KDA_SKIP_FLA_PRUNE=1 HIP_VISIBLE_DEVICES=<gpu>
# /opt/kda_env 必须放在最前面，见下方说明
export PYTHONPATH=/opt/kda_env:/public/home/tanbo/xplat/bwenv:<TREE>:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
export PATH=/opt/dtk/bin:$PATH
```

**已知坑：tanbo_mega_k3 里 `import fla` 会导致 g2 段错误**（上一个 agent 实测）。
- **原因**：site-packages 里装的那份 fla（2025-09-29 安装），在 torch 之后导入时会崩在 `flash_attn.utils.sparse_utils`；即使绕过这一处，导入 `fla.ops.kda` 之后 g2 的前向也会段错误。`kda_op_bench_tp.py` 开头就有 `from fla.ops.kda import ...`，所以不加处理时 `kda_iter.py` 会直接崩。
- **修法**：PYTHONPATH 以 `/opt/kda_env` 开头，让 `fla` 解析到源码树那一份，同时设 `KDA_SKIP_FLA_PRUNE=1`。参考 `examples/kda/hier_pmc/run_kda.sh` 的写法。
- **自检**：先跑一次 `kda_iter.py --iters 1`，确认不崩再开始批量采集。
- 远程命令的引号容易被多层转义打散。复杂的命令写成脚本文件，scp 过去再执行，不要用多层嵌套引号。

- `/public/home/tanbo` 是共享 NFS，bw52 和 bw7 都能看到。但新写的文件在另一台机器上可能有几秒的缓存延迟。
- kra 仓库有两份：
  - **权威副本**在用户 Mac 上：`/Users/tanbo/Desktop/WorkSpace/kernel-roofline-analysis-tools`；
  - 集群上的 `/public/home/tanbo/kra` 是用 scp 同步的工作副本。你写的脚本要放进 Mac 上的仓库，再用 scp 拷到集群上运行。

### 2.2 冻结测量对象（第一步就做）

```bash
# 容器内
cp -a /opt/kda_env/g2_r5 /opt/kda_env/g2_card_p52
TREE=/opt/kda_env/g2_card_p52
# 记录设备代码指纹（应为 35b7bd3a8862，即 hip_kda 生产版 9cc164f，含 P-52）
cd /tmp && so=$(ls $TREE/hip_kda/_G2*.so | head -1)
objcopy -O binary --only-section=.hip_fatbin $so f.bin
tgt=$(/opt/dtk/llvm/bin/clang-offload-bundler --list --type=o --input=f.bin | grep gfx936 | head -1)
/opt/dtk/llvm/bin/clang-offload-bundler --unbundle --type=o --input=f.bin --targets=$tgt --output=g.co
md5sum < g.co | cut -c1-12
```

之后所有测量都用 `TREE=/opt/kda_env/g2_card_p52`，并把这个指纹写进每份原始数据。

### 2.3 GPU 使用规则

- 每张卡都有 flock 锁，用法参考 `examples/kda/hier_pmc/run_kda_m.sh`：同时持有 `/public/home/tanbo/codex_validation/locks/bw7/hcu_<N>.lock` 和 `.../locks/hcu_<N>.lock`。你写的采集脚本也必须先拿到锁再运行。
- **PMC 并不是每张卡都能用**：历史上 GPU1 出现过 PMC abort。每张卡第一次使用前，先用 `examples/kda/hier_pmc/pmc_probe.sh` 的方式探测一次，能正常出数据才使用。GPU0 已确认可用。
- DTK 的默认 `--pmc` 预设会 abort。可用的方式有两种：
  - `--pmc-read` / `--pmc-write` 预设（`python -m kra pmc` 会调用它们）；
  - 用 `HIPPROF_INPUT_FILE` 指定自定义计数器列表，配合 `--pmc --pmc-type 3`，见 `hier_pmc/run_kda_m.sh`。
- 两次 profiler 运行之间要间隔几秒（`kra pmc` 已内置间隔和重试）。

---

## 3. 测试对象

### 3.1 形态矩阵（16 个）

| | TP8（H=12） | TP4（H=24） | TP2（H=48） | TP1（H=96） |
|---|---|---|---|---|
| T=8K（8192） | ✓ | ✓ | ✓ | ✓ |
| T=16K | ✓ | ✓ | ✓ | ✓ |
| T=32K | ✓ | ✓ | ✓ | ✓ |
| T=64K | ✓ | ✓ | ✓ | ✓ |

- 输入构造与 `tests/megatron/kda_op_bench_tp.py` 相同：dense，cu_seqlens=[0,T]，门控在 kernel 内，lower_bound=-5，beta 已过 sigmoid。
- 驱动脚本：`examples/kda/kda_iter.py --tree $TREE --arm g2 --T <T> --H <H> --iters <n>`。每次迭代之前有一个 marker kernel（名字含 `spin_kernel`），可以用它切出每次迭代的窗口。
- **执行顺序**：先做 16K 的 4 个 H，做完先出一版初稿（见 7.3）；再做 8K、32K、64K。64K/TP1 显存占用最大，如果 OOM，记录下来后跳过。

### 3.2 kernel 清单（生产版，按执行顺序）

| kernel 名（符号里的关键字） | 阶段 | 说明 / 实例区分 |
|---|---|---|
| `g2_fwd_prep_a_kernel` | 前向 | 门控、l2norm、块内矩阵，写 Gn 时已取 exp2f |
| `g2_fwd_prep_b_kernel` | 前向 | 块内 T 求逆的后半段，写出 W/U 等原生布局 |
| `g2_fwd_h_v8o4_kernel` 第 1 次（或 `_v8o4w_`） | 前向 | 递推 + 输出 o（MODE 1）；H≥40 且 NT≥512 时走 64 列宽版 `v8o4w` |
| `g2_l2n_apply_kernel` | 反向 | 重算 qn/kn |
| `g2_gate_kernel` | 反向 | 重算门控 cumsum g |
| `g2_wu_kernel` | 反向 | 重算 W/U，写出多种布局 |
| `at::native::*elementwise*`（2 个）+ `g2_exp2_kernel` | 反向 | to_gn 的 gather/permute，以及 Gn→exp2f |
| `g2_fwd_h_v8o4_kernel` 第 2 次（或 `_v8o4w_`） | 反向 | 重算递推，输出 bN/Vn（MODE 4）；**窄版在 ≥192 个 CTA 时走 GE=false 实例**（内部计算 exp2f） |
| `g2_dav_kernel` | 反向 | dA/dv |
| `g2_dhu_v8b_kernel` / `g2_dhu_v8b4_kernel` | 反向 | dh 递推；CTA 数在 96–256 之间用 4-wave 版，其余用 8-wave 版 |
| `p3_wy_s4a_kernel` | 反向 | 块内梯度（第一段） |
| `p3_wy_s4b_kernel` | 反向 | 块内梯度（第二段，dq/dk/dg） |
| `g2_bwd_tail_kernel` + `g2_bwd_tail_reduce_kernel` | 反向 | 门控、l2norm、beta 的反向收尾 |

注意：两次 `g2_fwd_h_v8o4_kernel` 同名，**按窗口内的出现顺序区分**（第 1 次为前向，第 2 次为反向重算）。

### 3.3 每个 kernel 的静态信息（每个形态各记一份）

grid、block、静态与动态 LDS（含 P-24 的 LDS 补齐）、arch VGPR、SGPR、scratch。PMC 的 CSV 里有 `grd`、`wgr`、`lds`、`arch_vgpr`、`scr` 等列。

---

## 4. 指标与测法

所有屋顶值都用**本机实测的可达值**，不用规格书数字。来源是 `machines/gfx936-bw1000.json`（2026-10-10 在 bw7 上标定）：

| 资源 | 可达值 | 理论值 / 备注 |
|---|---:|---|
| HBM 读 | 1339 GB/s | 1843 |
| HBM 写 | 1100 GB/s | |
| HBM 拷贝（读+写） | 1146 GB/s | |
| L2 读（数据常驻 L2） | 6101 GB/s | |
| LDS 读 | 15237 GB/s | |
| bf16 MMAC（16×16×16） | 476 TFLOPS | 491.5 |
| fp32 MMAC（16×16×8） | 61 TFLOPS | |
| fp32 VALU FMA | 29.7 TFLOPS | |
| exp2（SFU） | 1916 Gop/s | |
| TA 饱和利用率 | 0.92 | `TA_TA_BUSY` 归一化后的值 |

**标定复核（必做，写进交付物）**：在你实际使用的那张卡上重跑以下三项，与上表的偏差都应在 ±5% 以内。超出时以实测值为准，并在报告中注明。
- 流式读：`kra calibrate` 的 `hbm_read`，同时确认 TA 利用率约为 0.92；
- bf16 MMAC；
- LDS 读。

### 4.1 时间（所有比值的分母）

- **测法**：`hipprof --hip-trace --output-type 0`，采集 `kda_iter.py --iters 6` 的窗口，取每个 kernel 的设备时间（不用墙钟时间）。
- **重复次数**：每个形态至少跑 5 个独立进程，并打开显存位置抖动（`KRA_JITTER_SEED=<进程序号>`，用法见 `examples/kda/ab_kernels.sh`）。
  - 报告中位数和离散度（(max-min)/中位数）；
  - 离散度 > 5% 的 kernel，标注为“测量不稳定”。
- **PMC 运行时的时长不能用作时间**：PMC 会 replay，时长可能偏大约 12%。所有比值一律用 hip-trace 的中位时间作分母。

### 4.2 带宽屋顶

| 层 | 字节数的算法 | 利用率 |
|---|---|---|
| HBM | 读 = 64·(TCC_EA_RDREQ − TCC_EA_RDREQ_32B) + 32·TCC_EA_RDREQ_32B；写 = 64·TCC_EA_WRREQ_64B + 32·(TCC_EA_WRREQ − TCC_EA_WRREQ_64B)（已标定，见 `kra/pmc/hipprof_csv.py`） | `(rd/1339 + wr/1100 GB/s) / t`，即“按可达带宽搬完这些字节需要的时间 ÷ 实测时间” |
| L2 | TCC_REQ × 64 B；命中率 = TCC_HIT / TCC_REQ | 字节 / t ÷ 6101 GB/s |
| LDS | SQ_LDS_IDX_ACTIVE × 128 B（**含 bank 冲突重放周期**，不能当作真实数据量） | 字节 / t ÷ 15237 GB/s，只作为 LDS 通道占用的参考 |

**交叉核对**：至少对 wu、bwd_tail、dav 三个 kernel，用解析的接口字节核对 HBM 字节，偏差应在 ±15% 以内。
- 解析字节等于输入输出张量大小之和，参考 `results/kda-excess-work-20261013/README.md` 的账目和 `examples/kda/kda_bound.py`；
- 偏差超出时要找出原因（L2 命中、重复读取等），并写进报告。

### 4.3 发射屋顶

| 资源 | 计数与公式 | 饱和值 | 说明 |
|---|---|---|---|
| **VMEM/TA 发射** | `TA_TA_BUSY` 对 16 个实例求和 ÷ (GRBM_GUI_ACTIVE × 80) | 0.92 | 利用率 = 上式 ÷ 0.92。同时报告 VMEM 指令数（读和写分开），以及**每条 VMEM 指令搬运的字节数**（HBM+L2 字节 ÷ 指令数），用来衡量访问的分片程度。再按 `results/kda-bound-20261012/README.md` 的 TA 代价表，估算“如果全部改成 16 B/lane 连续访问，TA 能降到多少” |
| **VALU 流水线** | SQ_ACTIVE_INST_VALU ÷ (GRBM × 80) | 约 0.97 | 普通 VALU 计 1 槽，MMAC 16×16×16 计 2 槽，超越函数计 4 槽。要单独报告超越函数占的槽位比例，数据来自 ISA 静态计数，见 4.6 |
| **MMAC** | FLOPs = n_bf16 × 8192 + n_f32x8 × 4096（每条 wave 级指令） | 476 TFLOPS（bf16） | **必须用 ISA 核对**：上一版用 SQ_INSTS_MMOP 和 SQ_ACTIVE_INST_MMOP 推出 bf16/fp32 拆分（n_bf=(8I−A)/6），把前向递推的一部分 MMAC 算成了 fp32，得出“接近 fp32 MMAC 屋顶”的错误结论；实际上该 kernel 只有 `v_mmac_f32_16x16x16_bf16`。正确做法是用 ISA 里每种 MMAC 指令的静态条数 × 循环次数（递推为每个 CTA NT 步，其他 kernel 为每个 CTA 一轮）算 FLOPs，再与 SQ_INSTS_MMOP 核对条数 |
| **LDS 发射** | SQ_INSTS_LDS，冲突周期 SQ_LDS_BANK_CONFLICT | — | 报告每条 LDS 指令平均冲突几个周期，以及冲突周期 ÷ (GRBM × 80)。已知：dav 4.00、wu 2.51、前向递推 1.52、s4b 1.18（生产版，16K） |

### 4.4 延迟链屋顶（只针对三个递推 kernel：前向递推、反向重算、dhu）

- **链下限** = NT × t_step_min × CTA 轮数。
  - t_step_min 取 `examples/kda/bound_microbench.hip` 的 `k_step` 微基准实测值：48 个 CTA 时，只更新状态 1.015 µs，含 o 1.242 µs；192 个 CTA 时分别为 1.729 / 2.556 µs。
  - 注意：P-52 之后每步的指令变少了，**需要重新跑一次 `k_step`**（只更新状态和含 o 两种，48 和 192 个 CTA）。
- **实测每步时间** = kernel 时间 ÷ (NT × CTA 轮数)；它与 t_step_min 之比，就是“每步还剩多少可压缩空间”。
- **CTA 轮数** = ceil(CTA 数 ÷ (80 × 每 CU 驻留的 CTA 数))。每 CU 驻留数取以下三者的最小值：
  - LDS 限制：floor(64 KB ÷ 每个 CTA 的 LDS，含动态补齐)；
  - VGPR 限制：**按实测的同时驻留数计算，不要用编译器的 `; Occupancy:` 注释**（R38：VGPR 为 200–256 时每个 SIMD 仍能驻留 3 个 wave，见 `examples/kda/occupancy_probe.hip`。拿不准时用这个探针实测）；
  - P-24 的限流：
    - 前向/反向递推（v8o4）在 CTA 数 ≥224 时，用 LDS 补齐把每个 CU 限制为 2 个 CTA（每个 CTA 24 KB）；
    - dhu 在 CTA 数 ≥192 时，每个 CU 限制为 1 个 CTA（每个 CTA 33 KB）。
- TP8 时只有 48 个 CTA，不到 80 个 CU，所以还要报告“只按活跃 CU 计算”的各项利用率。

### 4.5 并行度与占用率（所有 kernel）

grid、block、每 CU 驻留的 CTA 数和 wave 数、CTA 轮数、最后一轮的空闲比例（tail）。

### 4.6 工作量下限（实际执行量 ÷ 必需量）

在主约束资源上给出“实际执行 ÷ 必需”的比值：

- **HBM 字节**：必需量取接口张量的字节数（该 kernel 必须读入和写出的张量，每个只算一次），来自 `kda-excess-work` 的账目。
- **VMEM 指令数**：必需量 = 实测字节 ÷ (64 lane × 16 B)，即最理想的 16 B/lane 连续访问需要的指令数。
- **VALU**：从 ISA 统计主循环里的指令构成（MMAC、超越函数、bf16 舍入 `v_bfe_u32`/`v_add3_u32`、`ds_write_b16`、64 位地址运算 `v_add_co`/`v_addc_co`、`s_waitcnt` 等）。统计方法可以参考主 agent 的做法：
  1. 在冻结树的 `csrc/g2` 下运行 `hipcc -O3 --offload-arch=gfx936 --cuda-device-only -S -x hip g2_ops.hip -o x.s`；
  2. 按 kernel 符号截取代码，找到主循环（最后一条回跳的 `s_cbranch` 到它的目标标签之间）；
  3. 按操作码计数。

  同时给出每个输出元素平均用多少条指令。
- **LDS 指令数**：报告每条 LDS 指令搬运的字节数（b16、b64、b128 的构成）。

### 4.7 停顿归因（SQTT，只在代表形态上做）

- **形态**：16K/12（TP8）和 16K/48（TP2）。
- **对象**：每个形态取时间占比前 6 的 kernel。
- **采集**：

```bash
hipprof --sqtt --sqtt-type 1 --kernel-name <kernel 关键字> --output-type 0 -o <dir>/x \
  python3 kda_iter.py --tree $TREE --arm g2 --T <T> --H <H> --iters 1 --warmup 1
```

- **分析**：`python3 examples/kda/sqtt_hotspots.py <dir> --top 12`。得到 issue、waitcnt、barrier、issue_delay 各占多少，以及等待最多的位置和被等待的访存指令。
- 判断时要注意：waitcnt 里等的是 vmcnt（显存）还是 lgkmcnt（LDS/SMEM）；issue_delay 高，通常说明某条流水线（VMEM/LDS/VALU）已经饱和。

---

## 5. 判定规则

对每个 (kernel, 形态)：

1. **找主约束**：在 {HBM, L2, LDS 通道, TA, VALU 流水线, MMAC, LDS 发射（含冲突）} 中取利用率最高的一项；递推 kernel 再加上“链下限 ÷ 实测时间”一项。再给出第二高的一项。
2. **判定是否见顶**：主约束利用率 ≥ 0.85，**并且**该资源上的执行量不超过必需量的 1.2 倍（递推 kernel 则要求实测每步时间不超过 t_step_min 的 1.15 倍）。
3. **没见顶时，判定缺口类型**：
   - **U（利用率缺口）**：主约束利用率 < 0.6，即延迟没有被隐藏。原因要写明：占用率、CTA 轮数、tail、等待链，还是依赖；
   - **W（工作量缺口）**：主约束资源上的执行量超过必需量的 1.2 倍。要写出多余的部分在哪：分片式访问、重复读、多写的布局副本、多余的 VALU（如舍入、地址计算）；
   - **C（链缺口）**：递推 kernel 的每步时间明显高于 t_step_min，或者 CTA 数少于 CU 数；
   - **S（结构缺口）**：这个 kernel 的必需工作量本身是拆分方式决定的，即 kernel 之间传递的中间张量。在 kernel 内部优化消不掉；
   - 可以同时属于多类，按占比排序。
4. **潜在收益** = t − max(各项下限)。其中“各项下限”指：按可达带宽搬运必需字节的时间、TA 下限、VALU 下限、MMAC 下限、链下限。

**算子级**（每个形态）：

| 量 | 来源 |
|---|---|
| 实测总时间 | Σ kernel 设备时间 |
| 现有拆分方式的下限 | Σ 每个 kernel 的 max(各项下限) |
| 第一性原理下限 | `python3 examples/kda/kda_bound.py`，参数与用法见文件头部 |

由此得出：实现缺口 = 实测 − 拆分下限；结构缺口 = 拆分下限 − 第一性原理下限。

---

## 6. 分析角度（交付物必须覆盖）

1. **按形态**：每个形态一张表。列包括：kernel、时间及占比、主约束及利用率、第二约束、执行量 ÷ 必需量、链比值（仅递推）、结论 / 缺口类型、潜在收益 µs。
2. **按 kernel 看形态变化**：每个 kernel 一行，16 个形态各一列，格子里写主约束和利用率。重点回答：主约束如何随 TP（H）和序列长度变化，例如递推 kernel 从 TP8 到 TP1 是否由延迟链转为 TA 限制。
3. **机会排序**：按 Σ 潜在收益（16K 的 4 个形态权重为 2，其他为 1）排出前 10。每一项标注置信度（计数器是否可靠、是否经过交叉核对），以及指向的优化类型：消除冲突、加宽访存、削减指令、提高并行度、改结构。
4. **专项清单**：
   - LDS 冲突：每条指令冲突周期 > 0.5 的 kernel；
   - 访存分片：每条 VMEM 指令搬运 < 512 B 的 kernel；
   - VALU 构成：舍入、地址计算、waitcnt 在主循环中各占多少。
5. **测量可靠性**：哪些指标经过了交叉核对、哪些没有；不同计数器之间、计数器与 ISA 或解析值之间的不一致之处。
6. **算子级**：16 个形态的“实测 / 拆分下限 / 第一性原理下限”三栏表，以及实现缺口、结构缺口各占的比例。

---

## 7. 交付物与验收

### 7.1 脚本（`examples/kda/ceiling/`）

- `collect.sh`：给定 TREE、GPU 和形态列表，完成全部采集。要求：
  - 运行前先拿锁；
  - 包含 hip-trace（5 个进程，带抖动）、PMC 预设（read/write）和自定义计数器（TCC_REQ/HIT、SQ_INSTS_MMOP、SQ_LDS_IDX_ACTIVE、SQ_LDS_BANK_CONFLICT、SQ_INSTS_LDS、TA_TA_BUSY、SQ_ACTIVE_INST_VALU 等）；
  - 可以重复运行，已有的原始数据跳过。
- `isa_mix.py`：从冻结树生成 ISA，统计每个 kernel 主循环的指令构成和 MMAC 类型，输出 JSON。
- `cards.py`：读取原始数据，加上 machine 文件、ISA 统计和解析字节，输出 `cards.json` 和 `cards.md`。
- `plot_cards.py`：生成 7.3 规定的全部图。在 `examples/kda/hier_roofline.py` 的基础上**复制一份再扩展**，不要修改原文件。
- `README.md`：一条命令重跑全部流程的说明。主 agent 以后每次合入新改动，都会用它重跑。

### 7.2 结果（`results/kda-ceiling-cards-20261013/`）

- `raw/<T>_<H>/`：原始 CSV 和 JSON，附带设备代码指纹，以及运行时的 `hy-smi` 状态。
- `calibration.md`：标定复核的结果。
- `cards.json`：机器可读，字段名要有说明。
- `cards.md`：中文报告，覆盖第 6 节全部 6 个角度；先给结论，再给表格；不确定的地方要明说。
- `fig/`：7.3 规定的全部图。`cards.md` 里用相对路径把图嵌进去，图紧挨着对应的表。

### 7.3 图和表（硬性要求：格式与参考图、参考表一致）

**参考图**：`results/gfx936-hier-roofline-20261013/gfx936_hier_roofline.png`，由 `examples/kda/hier_roofline.py` 生成。
**参考表**：`results/gfx936-hier-roofline-20261013/README.md` 里的“每个 kernel 的实测利用率”和“逐 kernel 分层落点”两张表。
用户最关心的就是这类图和表，必须**每个形态各出一份**，版式与参考图一致，只在下面规定的地方扩展或修正。

#### 图 A：每个形态一张双栏图 `fig/roofline_<T>_<H>.png`（共 16 张）

整体版式：matplotlib（Agg 后端），`figsize=(17, 7.5)`，左右两栏宽度比 1.35 : 1，`dpi=150`。

**左栏：分层 roofline**（与参考图相同）

- 坐标轴：双对数；x 轴为算术强度（FLOP/Byte），范围 0.1–2e4；y 轴为性能（TFLOPS），范围 0.5–900。
- 访存屋顶（斜线）：
  - LDS 15.24 TB/s，紫色实线；
  - L2 6.10 TB/s，蓝色实线；
  - HBM 1.34 TB/s，红色实线；
  - HBM 理论值 1.84 TB/s，红色点线。
- 算力屋顶（水平虚线）：
  - bf16 MMAC 476，黑色；
  - tf32 MMAC 237.7，灰色；
  - fp32 MMAC（16×16×8）61，绿色；
  - fp32 VALU FMA 29.7，橙色。

  每条线右端都标出名称和数值。
- 拐点：bf16 MMAC 和 fp32 MMAC 分别与 LDS / L2 / HBM 相交的点，画实心圆点并标出数值（参考图中为 31/78/355 和 4/10/46）。
- 算子级的点：本形态 KDA 前向+反向，x = 模型 FLOPs ÷ Σ HBM 字节，y = 模型 FLOPs ÷ Σ kernel 时间，画实心方块，旁边标注“AI xx, yy TFLOPS”。
- **每个有 MMAC 的 kernel 画三个点**，y 值相同，都是该 kernel 实际达到的 TFLOPS：
  - ● 圆点，x = FLOPs ÷ HBM 字节；
  - ▼ 倒三角，x = FLOPs ÷ L2 字节；
  - ◆ 菱形，x = FLOPs ÷ LDS 字节。

  三点之间用细横线相连（`lw=0.7, alpha=0.35`）。在圆点旁标注 `kernel名 (µs)`，字号 7。标签要手动错开，不能互相重叠。
- 图例中写明三种点型各代表哪一层。
- **相对参考图的两处修正**：
  - kernel 的 FLOPs 一律用 ISA 统计的 MMAC 条数 × 每条的 FLOP 数（bf16 16×16×16 为 8192，fp32 16×16×8 为 4096），**不再用 (8I−A)/6 的拆分**，也不画“executed-MMAC 估计（上界）”那个空心点；
  - 标题里写明形态、冻结树的设备代码指纹和数据日期。

**右栏：每个 kernel 卡在哪个屋顶上**（在参考图基础上扩展）

- 布局：每个 kernel 一组水平条形，按时间从大到小排列，纵轴标签写 `kernel (µs, 占比%)`。
- 每组的条形，按固定颜色和顺序排列：
  1. HBM 带宽 ÷ 可达值（红）
  2. L2 带宽 ÷ 可达值（棕）
  3. LDS 通道 ÷ 可达值（紫）
  4. VMEM/TA 发射 ÷ 0.92（橄榄绿）
  5. VALU 流水线槽位（橙）
  6. MMAC ÷ 476（黑）
  7. LDS 冲突周期占比（灰）
  8. 递推 kernel 额外加一根：链下限 ÷ 实测时间（青）
- 每组中主约束那根条形加粗边框，并在条形末端标注判定，如 `W: TA 0.78`。
- 竖直参考线两条：0.85（见顶阈值，实线）、0.6（U 类阈值，点线）；x 轴范围 0–1.05。
- 标题：`KDA G2 kernels, <T>/H<H>: which roof binds (measured PMC + ISA)`。

#### 图 B：主约束矩阵热力图 `fig/binding_matrix.png`

- 行为 kernel（含递推的两个实例），列为 16 个形态，按 TP8、TP4、TP2、TP1 分组，每组内按 8K→64K 排列。
- 格子颜色表示主约束利用率（0–1，同一个色阶）；格子里的文字写主约束缩写和百分比，如 `TA 78%`、`链 0.91`、`VALU 89%`。
- 已见顶的格子加粗边框。

#### 图 C：缺口类型矩阵 `fig/gap_matrix.png`

版式同图 B，格子内容改为缺口类型（U / W / C / S，可组合）和潜在收益 µs。颜色按缺口类型区分。

#### 图 D：算子级缺口 `fig/op_gap.png`

- 16 个形态并排，每个形态三根柱：实测总时间、现有拆分方式的下限（Σ 各 kernel 下限）、第一性原理下限。
- 实测那根按 kernel 分段堆叠，颜色在所有图中保持一致。
- 柱顶标注实现缺口和结构缺口各占的百分比。

#### 表（每个形态一组，放在 `cards.md` 里对应的图 A 下面）

**表 1：逐 kernel 分层落点**。列与参考表相同：

| kernel | µs | GFLOP（ISA 精确） | LDS GB/s（AI） | L2 GB/s（AI） | HBM GB/s（AI） | L2 命中率 | LDS 冲突周期/指令 |

**表 2：实测利用率与判定**。在参考表基础上扩展：

| kernel | 时间 µs（占比） | HBM 流量 MB | HBM 带宽 ÷ 可达 | L2 ÷ 可达 | LDS ÷ 可达 | VMEM 发射 ÷ 饱和 | VALU 槽位 | MMAC ÷ 峰值 | 链比值 | 主约束 / 第二约束 | 执行量 ÷ 必需量 | 结论（缺口类型） | 潜在收益 µs |

**表 3：主循环指令构成**（只列递推 kernel 和 s4a/s4b/dav/wu），每步或每轮的指令数：

| kernel | 总数 | MMAC | VMEM 读/写 | LDS 读/写（其中 b16 写） | bf16 舍入 | 64 位地址运算 | 超越函数 | s_waitcnt | barrier |

**汇总表**（放在 `cards.md` 开头）：
- 图 B、图 C 对应的矩阵表（kernel × 16 形态）；
- 机会前 10 名；
- 算子级三栏表（与图 D 对应）。

#### 通用要求

- 所有数字都由 `cards.py` 从原始数据生成，不要手工填写。图和表来自同一份 `cards.json`，必须一致。
- 数值保留到能区分差异即可：时间取整到 µs，利用率取两位小数，AI 取一位小数。
- 不可靠或缺失的指标：图里用斜线填充或灰色表示，表里写“—”并加注原因。

### 7.4 初稿

16K 的 4 个形态做完后，先提交一版初稿，包括 `cards.md`、这 4 个形态的图 A，以及图 B、图 C 的 16K 部分，并告知用户，方便主 agent 尽早参考。

### 7.5 验收标准

- 每个形态各 kernel 设备时间之和，与独立的 hip-trace 总时间偏差在 ±3% 以内；
- wu、bwd_tail、dav 的 HBM 字节与解析值偏差在 ±15% 以内，超出时要给出解释；
- 前向递推、反向重算、dhu、s4a、s4b 的 MMAC 条数，ISA 推算值与 SQ_INSTS_MMOP 偏差在 ±5% 以内；
- 16K/12 独立跑两次，各项利用率偏差在 ±3% 以内；
- 每个 (kernel, 形态) 都有结论，或明确标为“不可判定”并写明原因；
- 7.3 规定的图和表齐全：16 张图 A、图 B、图 C、图 D，每个形态的表 1–3，以及开头的汇总表。左栏版式与参考图一致，即屋顶线、拐点、三种点型、标签不重叠；
- 图中数值与 `cards.json` 抽查一致：随机抽 3 个形态 × 3 个 kernel 核对。

**提交时只 add 自己新增的文件**（`git add <路径>`），不要用 `git add -A` 或 `git add .`。工作区里可能有其他人尚未提交的改动。

完成后在本地 kra 仓库提交（不要 push），提交信息写清楚范围和主要结论，然后向用户汇报。

---

## 8. 已知陷阱

- **显存位置噪声**（R20/R41）：只要分配顺序或 code object 大小变了，机器码相同、受带宽限制的 kernel 也会波动 2–8%。所以必须多进程加抖动，并报告离散度。
- **occupancy**（R38）：DTK 编译器的 `; Occupancy:` 注释按每个 lane 256 个 VGPR 计算，结果偏低。要以实测为准。
- **PMC**：read 和 write 两次 replay 的时长可能相差约 12%，不要拿来当时间用。短于 10 µs 的 kernel，计数不可靠。
- **TA_TA_BUSY**：必须对 16 个实例求和，再除以 (GRBM × 80)。单个实例的数值无法直接归一化（见 `docs/methodology.md` 第 4 节，第 3 节里“不使用”的说法已经过时）。
- **MMAC 拆分**：不要用 (8I−A)/6 推算 bf16 和 fp32 的比例，一律以 ISA 为准（见 4.3）。
- **两个同名的递推 kernel**：按出现顺序区分。另外 H≥40 且 NT≥512 时走宽版 `v8o4w`，16K/48 和 8K/96 的反向重算走 GE=false 实例，这些都要在表里注明。
- **LDS 字节**：`SQ_LDS_IDX_ACTIVE×128B` 包含冲突重放，冲突越多，这个数虚高越多。
- **GPU 被别人占用**：测之前确认空闲；测量过程中发现被占用的，整组重测。
- **自定义计数器**：同一个 block 的计数器分多行写时，hipprof 会合并处理；一次能采的计数器数量有限，必要时分多次运行，参考 `hier_pmc/run_kda.sh` 的四次分组（SQ 一组、TCC 一组、EA 读一组、EA 写一组）。
- **计数高不等于时间代价**（R42）：dav 平均每条 LDS 指令冲突 4.0 个周期，但消融显示完全消除冲突只快 0.9–2.4%，冲突被 VMEM 和 HBM 的限制掩盖了。所以“潜在收益”不能只按计数推算；机会前 10 名里每一项都要注明，是否有消融上限（指令条数不变、只去掉被怀疑的开销）作为依据。已有的消融结果见 `results/kda-gfx936-20261013-dav-conflict/`。
- **`import fla` 段错误**：见 2.1，PYTHONPATH 必须以 `/opt/kda_env` 开头。
- **自写探针的输入形状**：上一个 agent 自己写的探针，把 g 的形状写错了，导致前向崩溃。输入一律从 `kda_op_bench_tp.py` 的构造代码里取，不要自己拼。

---

## 9. 与主 agent 的协作

- 主 agent 在 bw52 上做优化实验，生产版可能会更新（新的提交会写进 `results/` 下的记录和 `docs/playbook.md`）。你只测冻结的 `g2_card_p52`。
- 如果发现一个确定的优化点（例如某个 kernel 的冲突或分片特别严重），在报告里列出证据就行，**不要自己改 kernel**。
- 你写的脚本以后会被主 agent 用来测新版本，所以路径、TREE、GPU 都要能通过参数传入，不要写死。
