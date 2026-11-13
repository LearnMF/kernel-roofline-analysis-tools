# 机器文件能力等级与字段申请清单

两种用法共用同一个算子语义库，区别只在机器文件有多全：

* **A. 已有硬件**：先用 `kra.calibrate` 采一轮微基准，内化到机器文件，之后只输入算子结构就能分析。
* **B. 未落地硬件**：只有数据手册或 PV/SOC 仿真的参数。

工具对缺字段的处理一贯是「不猜、该项不计、在 notes 里写明」。好处是下限永远是下限，坏处是
**缺字段不报错，只是悄悄变松**，容易让人误以为结论一样可信。本文把「这张卡能回答到哪一层」
变成显式的等级，并给出向上游申请数据时的字段清单。

## 四个能力等级

| 等级 | 机器文件里有什么 | 能回答什么 | 不能回答什么 |
|---|---|---|---|
| **L0 参数** | `cu_count`、HBM 带宽、各 dtype 算力峰值 | `T_theory`、算术强度、拐点、瓶颈在算力侧还是访存侧、天花板 MFU | 任何与拆分/片上容量/并发有关的问题 |
| **L1 + 容量** | 加 `lds_per_cu`、`regs_per_block`、`max_threads_per_cu`、`warp_size`、`clock_khz`、`l2_bytes` | 容量可行性与 `T_split`：哪些 kernel 融得下、交接要不要落盘、VGPR 够不够 | 「铺不开」的代价；巨核不被罚，于是**融合永不要钱、kernel 越少越优** |
| **L2 + 曲线** | 加 `hbm_read_vs_ctas` 曲线、`latency.gmem_hbm_ws`、`l2_read`、`lds_read`、`launch` | 并发带宽、掩藏不住的延迟、多层屋顶、固定开销；**拆法推荐才有意义** | 算子特有的依赖链步长 |
| **L3 + 算子相关** | 加 `structural`：`mmac_tile_eff`、`vmem_cycles_per_kb`、`recurrence_step_us` | 可达算力、访存发射、串行递推链 | — |

当前状态：gfx936 = **L3**；gfx938 = **L2**（无 `structural`，其报告自动声明「串行链没有计入」）。

### 等级是实测过的，不是推测

只有「带宽 + 算力 + 容量」的 L1 机器文件（104 CU / 1.7 GHz / 3200 GB/s / 900 TF）跑 KDA
16K/H=12，按 kernel 数取最优 `T_split`：

| | 8 个 kernel | 9 个 | 10 个 | 最优 |
|---|---:|---:|---:|---|
| L1，占用率开 | **1122 us** | 1153 | 1185 | 8 个 |
| L1，占用率关 | **1122 us** | 1153 | 1185 | 8 个 |
| gfx936（L3），占用率开 | 3536 | **3121** | 3197 | 9 个 |
| gfx936（L3），占用率关 | **3028** | 3103 | 3179 | 8 个 |

L1 的两行完全相同：占用率项只剩「CU 铺不满」一条，而 KDA 每个 kernel 的 tile 数都远超 104，
所以恒不生效。**所以 L1 不该用来推荐拆法。** 两项真正起作用的占用率输入（CTA 曲线、访存延迟）
都在 L2。

L0 则连容量都解不出来，工具直接拒绝而不是估一个：

```
machine L0-min has no device.lds_per_cu: the capacity split cannot be solved on it
```

## 字段申请清单

「怎么拿」一栏的判断依据：带宽/延迟/发射是时序仿真的天然产物，PV/SOC 阶段就能出；
可达系数类需要真机调优才知道。

| 字段 | 等级 | 支撑的下限项 | 数据手册 | PV/SOC 仿真 | 真机微基准 |
|---|---|---|---|---|---|
| `device.cu_count` `warp_size` `clock_khz` | L0 | 每 CU 资源折算、发射项 | 有 | 有 | `hipDeviceProp` |
| `peaks.hbm_read` `hbm_write` | L0 | `memory:hbm` | 有（理论值） | **有（可达值）** | `hbm_read`/`hbm_write` |
| `peaks.mmac_*`（各 dtype） | L0 | `compute:*` | 有 | 有 | `mmac_bf16_full` 等 |
| `peaks.valu_fma_f32` `sfu_exp2` | L0 | `other:*` | 多半无 | 有 | `valu_pk_fma`/`sfu_exp2` |
| `device.lds_per_cu` `lds_per_block` | L1 | 容量可行性、落盘判定 | 有 | 有 | `hipDeviceProp` |
| `device.regs_per_block` | L1 | 寄存器占用率上限、预取深度 | 有 | 有 | `hipDeviceProp` |
| `device.max_threads_per_cu` | L1 | wave slot 上限 | 有 | 有 | `hipDeviceProp` |
| `device.max_vgpr_per_thread` | L1 | 在途窗口上限 | 有（ISA 文档） | 有 | **两张卡都缺，现按 256 兜底** |
| `device.l2_bytes` | L1 | 多层屋顶的驻留判断 | 有 | 有 | `hipDeviceProp` |
| `peaks.hbm_read_vs_ctas.curve` | **L2** | **并发带宽（巨核惩罚）** | 无 | **有，优先要这个** | `hbm_read_ctas` |
| `latency.gmem_hbm_ws` | **L2** | **`latency:hbm`** | 无 | **有** | `k_chase` 指针追逐 |
| `peaks.l2_read` `lds_read` | L2 | `memory:l2` `memory:lds` | 多半无 | 有 | `l2_read`/`lds_read` |
| `launch.launch_stream` | L2 | `fixed:launch` | 无 | 有（含驱动模型时） | 空 kernel `k_empty` |
| `latency.lds` `mmac_*_dependent` `barrier` | L2 | 依赖链项（**已测、predict 尚未接入**） | 无 | 有 | `k_lds_chase`/`k_mmac_latency`/`barrier` |
| `peaks.vmem_issue` `vmem_store` | L2 | 推导 `vmem_cycles_per_kb` | 无 | 有 | `vmem_issue`/`vmem_store` |
| `structural.mmac_tile_eff` | **L3** | 可达算力系数 | 无 | 勉强 | 需真机调优，**最难要** |
| `structural.vmem_cycles_per_kb` | L3 | `issue:vmem` | 无 | 有 | 从 `vmem_issue` 推 |
| `structural.recurrence_step_us` | L3 | `serial:chain` | 无 | 无 | **算子专属**，换算子必须重标 |

### 一句话版本（可直接发给上游）

> 必需（否则只能出理论天花板）：CU 数、频率、warp 宽度、HBM 读写带宽、各 dtype 算力峰值、
> 每 CU 的 LDS 容量、寄存器文件大小、每 CU 最大线程数、每线程最大 VGPR、L2 容量。
>
> 希望拿到（否则无法评估 kernel 拆分，只能给天花板）：**HBM 读带宽随驻留 CTA 数的曲线**
> （1/2/4/8/16/32/48/64/满 CU 各一个点）、**HBM 访存延迟**（大工作集指针追逐）、L2 与 LDS 带宽、
> 空 kernel 的 launch 开销、LDS/MMAC 依赖延迟与 barrier 延迟。
>
> 这两项中的曲线和延迟是时序仿真的天然产物，比流片后微基准更早可得。

## 外推（还没做，待定）

缺的曲线原则上可以从已有卡外推：CTA 曲线按「CU 数 × 单 CU 斜率」平移、延迟按频率缩放。
好处是 L1 参数也能做 what-if，坏处是**结果不再是严格下限**。若要做，规则是：

1. 每个外推值在台账里标成高泛化风险；
2. 报告里逐条列出哪个数是推的、从哪张卡推的；
3. 结论口径从「下限」降级为「估计」，并与 PV/SOC 仿真结果对照（见
   `docs/sim/validation-pv-soc-sim.md`）。
