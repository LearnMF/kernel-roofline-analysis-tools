# 测量方法与口径

本页记录 kra 每个数字的来源、计算方式和已知的工具语义。对外引用任何结论时，应同时给出本页对应的口径。

## 1. L0 硬件标定（`kra calibrate`）

### 1.1 原则

- **峰值取“可达峰值”**：每项在多组配置（grid、block、unroll、独立累加链数）下测量，取最好的一组作为分母，同时记录该配置的中位数。不使用宣传值作分母。
- **ISA 核对**：保存编译出的设备汇编，逐个 kernel 检查确实使用了预期指令（例如 `v_mmac_f32_16x16x16_bf16`、`v_pk_fma_f32`、`global_load_dwordx4`）。检查不通过时 `isa_check.ok=false`，对应峰值不可引用。
- **可推导时给出理论值**：HBM 理论带宽 = 显存时钟 × 2（DDR 假设）× 位宽 / 8；算力项给出“每 CU 每时钟的运算数”，可直接和指令手册的发射率对照。
- **计时**：hipEvent 计时，每个配置 2 次预热 + 10 次计时；kernel 时长至少数百微秒到数毫秒，使 launch 和尾部效应低于 1%。
- **可复现**：同一节点两张卡结果差异 ≤0.4%（L2 为 3%）。

### 1.2 gfx936 / BW1000（bw7）实测结果

来源：[`machines/gfx936-bw1000.json`](../machines/gfx936-bw1000.json)（原始输出 `.raw.jsonl`）。
设备：80 CU，wave64，shader 时钟实测 1500 MHz，L2 8 MiB，LDS 64 KiB/CU，DTK 25.10。

| 项 | 可达峰值 | 理论/对照 | 说明 |
|---|---:|---|---|
| HBM 读 | **1338 GB/s** | 1843 GB/s（72.6%） | 非临时读 1309；写 1101；拷贝 1146（读+写字节） |
| L2 读（驻留） | 6.1 TB/s | 49 B/CU/clk | 工作集 ≤ L2/2；工作集 32 MiB 时回落到 HBM 水平 |
| LDS 读 | 15.2 TB/s | 127 B/CU/clk（理论 128，99%） | 无冲突 `ds_read_b128` |
| MMAC bf16 | **476 TFLOPS** | 3966 / 4096 flop/CU/clk（96.8%） | f16 480；tf32 238；fp32（16x16x8）61；int8 951 TOPS |
| VALU fp32 FMA | 29.7 TFLOPS | 248 / 256 flop/CU/clk | **需要 `v_pk_fma_f32`**；不打包的 `v_fma_f32` 只有一半（约 15） |
| 特殊函数 exp2 | 1.92 Top/s | 16 op/CU/clk | 1/4 速率 |
| 屋脊点（vs HBM 读） | bf16 MMAC 356 flop/B；VALU fp32 22 flop/B | | 算术强度低于屋脊点即先验访存受限 |

延迟（单 lane、指针追逐、cycle @1.5 GHz）：

| 项 | cycles | ns |
|---|---:|---:|
| 全局读，16 KiB 工作集（L1 驻留） | 152–156 | ~103 |
| 全局读，2 MiB（L2 驻留） | 369–383 | ~250 |
| 全局读，512 MiB（HBM + TLB miss） | 572–577 | ~382 |
| LDS 读 | 64 | 43 |
| 相互依赖的 MMAC bf16 | 44 | 29 |
| CTA barrier（256 / 512 / 1024 线程） | 52 / 56 / 68–76 | 35 / 37 / 45–51 |

launch：同一 stream 上连续空 kernel 1.96–2.18 µs/个；HIP graph 回放 1.67 µs/个；单次 launch + 同步往返 6–8 µs。PCIe（pinned）：H2D 28.4 GB/s，D2H 35.4 GB/s。

### 1.3 对判定标准的直接影响

- **带宽利用率的分母必须写清楚。** 本机纯流式读也只能达到理论带宽的 72.6%。按理论值 1843 GB/s 计算，“达到 85–90% 带宽利用率”根本不可能实现；按可达峰值 1338 GB/s 计算，85–90% 对应 1137–1204 GB/s。kra 默认使用可达峰值作分母，同时报告相对理论值的比例。
- **VALU 峰值依赖打包指令。** kernel 的 ISA 中没有 `v_pk_fma_f32` 时，其 VALU 上限应按一半计算。
- 上面的延迟数据就是延迟型 kernel 关键路径下限（`串行步数 × 单步最小延迟`）的基本单元。

## 2. L1 系统层时间线（`kra timeline`）

### 2.1 采集

```bash
hipprof --hip-trace --output-type 0 -o OUT  <app>      # 产物 OUT.json（chrome trace）
python -m kra timeline OUT.json --marker spin_kernel --machine machines/gfx936-bw1000.json \
       --reference-ms <不开 profiler 时每个窗口的耗时>
```

每次算子调用前插入一个标记 kernel（KDA 示例中用 `torch.cuda._sleep`），按标记把 trace 切成窗口，每个窗口统计一次，最后汇总中位数和最小/最大值。

### 2.2 指标

- `span`：窗口内第一个设备操作开始到最后一个结束；`busy`：所有设备操作区间（跨 stream）的并集。
- `bubble_ratio = (span − busy) / span`。
- gap 成因：`sync`（gap 内有同步类 API 返回）、`host_late`（下一个操作提交时上一个已经结束，即主机供不上）、`dispatch`（已在队列中，属于设备侧派发间隔）、`unknown`。
- `launch_floor`：操作数 × L0 测得的单次 launch 代价。
- `embedded_dispatch_upper`：见 2.3。

### 2.3 hipprof 时间戳语义（实测，gfx936 / DTK 25.10）

用 L0 的空 kernel 做对照实验：

| 场景 | hipprof 报告的单个 kernel 时长 | 报告的间隔 | 不开 profiler 的实测 |
|---|---:|---:|---:|
| 单个 kernel 独立 launch | 0.48 µs | 22–24 µs | 往返 6–8 µs |
| 2000 个连续 launch | **4.8 µs** | **0（97% 的间隔严格为 0）** | 1.96 µs/个 |

结论：
1. **已经在队列中的 kernel，其 BeginNs 等于前一个 kernel 的 EndNs。** 相互依赖的连续 kernel 之间的派发开销被并入 kernel 时长，trace 中看不到间隔。trace 里可见的间隔只代表“队列空了”（主机来不及提交，或发生同步）。
2. **开启 tracing 会放大每次派发的开销**（4.8 µs，未开 profiler 时为 1.96 µs）。
3. 因此 L1 分别报告：可见空泡（来自 trace）和隐藏的派发开销上限（排队的连续操作数 × L0 不开 profiler 的 launch 代价）。短 kernel（< 10 µs）的 hipprof 时长会被派发开销显著抬高，做 kernel 级分类时应使用不开 profiler 的计时或 PMC 的 `GRBM_GUI_ACTIVE` 口径。

### 2.4 可信性检查

`traced_over_reference = trace 中窗口 span 的中位数 / 不开 profiler 的每窗口耗时`，超过 2 时结论不可信（NV Triage Guide D0）。注意，KDA 示例的参考计时从“算子调用开始”算起，包含了窗口开头主机发出第一个 kernel 前的时间（因为每轮之前都做了同步，队列是空的）。trace 窗口则从第一个设备操作开始，所以这个比值可能略小于 1。

## 3. L2 实测分类（`kra pmc` + `kra analyze`）

### 3.1 采集

```bash
python -m kra pmc --out OUT --marker spin_kernel -- <app>   # 两次运行：--pmc-read、--pmc-write（csv）
```

在 gfx936 / DTK 25.10 上，可用的只有 `--pmc-read` 和 `--pmc-write` 两个预设：默认的 `--pmc` 预设在本机每次都会在队列创建时 abort（signal 6）；GPU1 上 PMC 也会 abort，GPU0 正常。两次 profiler 运行之间需要间隔几秒，`kra pmc` 已内置间隔和重试。

### 3.2 计数器语义（用工作量已知的 kernel 实测标定）

`microbench pmccal` 每个参考 kernel 只发射一次，工作量精确已知。原始 csv 在 [`results/pmc-semantics-gfx936-20261009/`](../results/pmc-semantics-gfx936-20261009/)，对应回归测试为 `tests/test_pmc_semantics.py`。

| 计数器 | 实测语义 | 证据 |
|---|---|---|
| `TCC_EA_RDREQ` / `_32B` | 每个请求 64 B（32B 请求 32 B） | 读 1 GiB → 2^24 个请求，误差 < 0.01% |
| `TCC_EA_WRREQ` / `_64B` | 64B 请求 64 B，其余 32 B | 写 1 GiB 完全吻合 |
| `SQ_INSTS_VALU` | wave 级 VALU 指令数，**包含 MMAC** | MMAC 6.71 亿条 → INSTS 6.71 亿 + 循环开销 |
| `SQ_ACTIVE_INST_VALU` | **VALU 流水线发射槽**：普通 VALU 计 1，MMAC 16x16x16 计 2，exp 计 4 | ACTIVE 与 INSTS 之比：pk_fma 1.00，MMAC 2.00，exp 4.00 |
| 流水线利用率 | `ACTIVE / (周期数 × CU 数)`，三种峰值下都达到 0.97–0.98 | 分母使用较快那次 replay 的时间 × 1.5 GHz |
| `ACTIVE − INSTS` | 多槽指令的额外槽位：没有超越函数时等于 MMAC 条数；有超越函数时为 MMAC 条数的上限 | |
| `TA_TA_BUSY` | 无法归一化（超过 GRBM 周期数），**不使用** | |

注意：
- 之前 `pmc_lds.py` 中的 “valu_busy = ACTIVE / CU 周期 / 4” 多除了一个 4，把 VALU 利用率低估了 4 倍。
- read 和 write 两次 replay 的 kernel 时长在长时间计算 kernel 上可能相差约 12%。kra 取两次中较小的作为 `t_us`。

### 3.3 分类规则

- 两个顶层 SOL：`sol_pipe` 为流水线利用率；`sol_mem` 为 HBM 实测字节数按可达读/写带宽折算的时间 ÷ 实测时间。
- 任一 SOL ≥ 60% 判为 B 类（算力或访存，取较高者；MMAC 发射槽占比 ≥ 50% 判为 `B.compute.mmac`）。两者都在 40–60% 判为均衡型。否则判为 C 类，再细分：
  - CTA 数少于 CU 数 → `C.parallelism`，同时给出“仅按活跃 CU 计算”的 SOL；
  - op_spec 声明为串行 → `C.serial`；
  - 其余为 `C.unresolved`，子类型需要 L3（SQTT）确认。
- D 层病因：LDS bank conflict 周期 ≥ 5%（`SQ_LDS_BANK_CONFLICT / (周期 × CU)`）、scratch > 0、kernel 短于 10 µs。

## 4. L4 下限模型

每个 launcher 调用（1–2 个 kernel）有两种下限：

| 下限 | 组成 | 回答的问题 |
|---|---|---|
| **先验下限** `T_lower_prior` | max(接口字节按该 grid 可达带宽折算的时间, 只算依赖的关键路径, launch) | 换任何实现、只要保持这个算法和 launcher 边界，最快能到多少 |
| **实现下限** `T_lower_impl` | max(实测字节按可达带宽, VALU 流水线发射槽满载, MMAC 工作量, 当前数据流的关键路径, launch) | 不改工作量、只把硬件用满，最快能到多少 |

- **接口字节**：由 `kra.opspec.capture` 在真实运行中逐个记录每个 launcher 的输入/输出张量（同一 storage 只计一次）。假设每个输入至少读一次、每个输出至少写一次。
- **小 grid 的带宽**：L0 的 `hbm_read_vs_ctas` 曲线，每个 CTA 约 42 GB/s，48 个 CTA 时 1248 GB/s，80 个 CTA 时 1335 GB/s。
- **关键路径**：`串行步数 × Σ(原语个数 × L0 测得的原语延迟)`。原语包括 barrier、LDS、相互依赖的 MMAC。`chain` 描述当前数据流；`chain_min` 只保留算法上不可避免的依赖（一次跨 lane 的状态交换，加两级 MMAC），用于先验下限。项数只会少算，因此仍是合法的下限。
- **结论档位**（阈值见 `thresholds.json`）：
  - 先验达成率 ≥ 80%：已到顶；
  - 实现达成率 ≥ 80%：硬件已忙，但有多余工作；
  - 先验达成率 < 50%：空间很大；
  - 其余：中等空间。
- **可回收时间** = 实测 − 先验下限，按此排序就是优化优先级（Amdahl 定律）。

## 5. 阈值

所有判定阈值在 [`kra/thresholds.json`](../kra/thresholds.json)，每项标注来源（NV Triage Guide、KernelPro、AutoKernel、团队约定）以及 `hcu_calibrated`。目前除噪声门槛外都尚未经 HCU 校准，KDA 是第一个校准样本。
