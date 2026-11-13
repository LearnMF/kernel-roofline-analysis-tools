# 机器文件字段对照表

`machines/<id>.json`（`schema: kra.machine/1`）是求解器的硬件侧输入。本文逐字段说明：**谁生成的**、
**哪个下限项读它**、**缺了会怎样**。

生成方式分三类：

| 标记 | 含义 |
|---|---|
| 自动 | `python3 -m kra.calibrate run` 跑一轮微基准后写出，带 config / n_configs / theoretical 等现场记录 |
| 查询 | `hipDeviceProp` / `rocm-smi` 直接读出 |
| 手写 | 人工填入，且必须在同级加一个 `_<键名>` 说明出处 |

原则：**缺字段一律不猜。** 对应的下限项直接不计入，并在报告里写明「该项没有计入」。
这保证了下限永远是下限，代价是机器文件越不全、下限越松。

## 顶层

| 字段 | 来源 | 被谁用 | 缺了会怎样 |
|---|---|---|---|
| `schema` / `id` | 自动 | 报告头、结果文件名 | id 缺则退回 `device.name` |
| `provenance` | 自动 | 不进计算 | 无法追溯这组数测于何时何机 |
| `device` | 查询 | 见下 | — |
| `peaks` | 自动 | 见下 | — |
| `ridge_points_flop_per_byte` | 自动 | 报告里的拐点一栏（各算力峰值 ÷ `hbm_read`） | 报告现算，不影响下限 |
| `latency` | 自动 | 见下 | — |
| `launch` | 自动 | `fixed:launch` | 不计固定开销 |
| `pcie_GBps` | 自动 | **当前未用**（算子都在卡上） | — |
| `observed_shader_clock_mhz` | 自动 | **当前未用**；核对 `clock_khz` 是否被降频 | — |
| `warnings` | 自动 | 标定期间的异常，人工看 | — |
| `isa_check` | 自动 | 反汇编核对微基准真的发了目标指令 | 峰值可能是被优化掉的空循环 |
| `raw` | 自动 | 全部原始 benchmark 行，复算用 | — |
| `structural` | **手写** | 见下 | 对应项不计入 |

## `device`（全部来自 `hipDeviceProp`）

| 字段 | 被谁用 |
|---|---|
| `cu_count` | 每 CU 资源折算、`cu_frac`、`latency:hbm`、`issue:vmem` |
| `warp_size` | 占用率：每 CTA 波数、寄存器文件换算 |
| `clock_khz` | `issue:vmem`；`valu_lane` 峰值缺失时的兜底推导 |
| `lds_per_cu`（或 `lds_per_block`） | 占用率的 LDS 上限 |
| `regs_per_block` | 占用率的寄存器文件上限、在途字节窗口 |
| `max_threads_per_cu` | 占用率的 wave slot 上限 |
| `max_vgpr_per_thread` | 在途窗口的上限；**两张卡都没有这个字段，默认按 256** |
| `l2_bytes` / `memory_bus_width` / `memory_clock_khz` / `global_mem_bytes` / `wall_clock_khz` / `name` / `gcnArchName` | 报告与校验，不进下限 |

## `peaks`

每一项都是一次扫参的完整记录，不是一个数：

```json
"mmac_bf16": {"value": 475.955, "median_of_best_config": 475.751,
  "config": {"grid": 640, "block": 512, "chains": 8, "iters": 16384},
  "n_configs": 8, "unit": "TFLOPS", "theoretical": 491.52,
  "theoretical_note": "4096 FLOP/clk/CU x 80 CUs x 1.5 GHz", "pct_of_theoretical": 96.8}
```

因为 `value` 和 `theoretical` 都在，所以 `--peaks measured|theoretical` 能一键切换两套屋顶。

| 字段 | 被哪个下限项用 | 缺了会怎样 |
|---|---|---|
| `hbm_read` | `memory:hbm` 的读部分 | 四个 HBM 峰值全缺则直接报错退出 |
| `hbm_write` | `memory:hbm` 的写部分 | 退回「总字节 ÷ 最好的 HBM 带宽」 |
| `hbm_read_nt` / `hbm_copy` | 只参与「最好的 HBM 带宽」兜底 | — |
| `hbm_read_vs_ctas.curve` | **并发带宽**：按驻留 CTA 数插值，读按可达带宽计价 | 读按峰值计价（下限更松） |
| `l2_read` | `memory:l2`（多层屋顶） | 该层不计 |
| `lds_read` | `memory:lds`（按 `cu_frac` 折算） | 该层不计 |
| `mmac_bf16` / `mmac_f16` / `mmac_tf32` / `mmac_f32x8` / `mmac_i8` | `compute:*`，按 dtype 选 | 报错（算力下限无从算起） |
| `valu_fma_f32` → `valu_lane` | `other:valu_lane` | 由 `cu_count × warp_size × clock` 推导 |
| `sfu_exp2` | `other:sfu_exp2` | 该项不计 |
| `vmem_issue` / `vmem_store` | **当前未直接用**；`structural.vmem_cycles_per_kb` 就是从这两项推出来的手写值 | — |

CTA 曲线是这一轮占用率建模的关键输入，两张卡都有：

| 驻留 CTA | 1 | 8 | 16 | 32 | 48 | 64 | 80 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gfx936 GB/s | 41.6 | 336.1 | 636.0 | 1064.5 | 1248.3 | 1322.1 | 1334.7 |
| gfx938 GB/s | 34.3 | 262.0 | 529.1 | 1033.1 | 1470.4 | 1839.6 | 1908.7（72 CTA） |

读出来的事实：gfx936 要 48 个 CTA 才到峰值的 94%，8 个只有 25%。所以「占满片上资源、每 CU 只驻留
一个 tile」的巨核，在**下限**里也必须被罚——这不是效率折扣，是流式微基准的实测最好成绩。

## `latency`

| 字段 | 被谁用 |
|---|---|
| `gmem_hbm_ws.ns` | `latency:hbm`（Little's law：每 CU 必读字节 × 延迟 ÷ 在途窗口） |
| `gmem_l2_ws` / `gmem_small_ws` / `lds` | **predict 当前未用**；`kra/model/bounds.py` 的事后归因路径在用 |
| `mmac_bf16_dependent` | 同上。接进 `T_split` 的依赖链项是下一步的候选 |
| `barrier.{syncthreads,lds_barrier}_{256..1024}` | 同上 |

## `launch`

`launch_stream.best_us` 乘 kernel 数得到 `fixed:launch`。`launch_graph` / `launch_single` 暂未用。

## `structural`（唯一手写的一块）

| 字段 | 值（gfx936） | 被谁用 | 泛化风险 |
|---|---|---|---|
| `mmac_tile_eff` | 0.92 | `compute:*` 的可达系数（算子用 `compute_eff` 指名引用） | 低：一个操作数常驻寄存器的 MMAC tile 微基准 |
| `vmem_cycles_per_kb` | read 21.0 / write 56.0 | `issue:vmem` | 低：行连续 16 B/lane 访存指令的每 CU 周期 |
| `recurrence_step_us` | `kda_step_with_o.1cu` = 1.241 等 | `serial:chain` | **高：KDA 专属**，键名带 `kda_`，换算子必须重标 |

gfx938 的文件里**整个 `structural` 块都没有**，所以它的报告直接写「串行链没有计入这个下限」，
而不是沿用 gfx936 的 1.241us。这是防过拟合的机制之一，不是疏漏。

## 两张卡的差异（泛化检验的来源）

| | gfx936 / BW1000 | gfx938 / BW1102 |
|---|---:|---:|
| CU | 80 | 72 |
| 频率 | 1.5 GHz | 1.3 GHz |
| 显存位宽 | 4096 | 256 |
| 寄存器文件 `regs_per_block` | 196608 | 131072 |
| `structural` | 有 | 无 |

寄存器文件差 1.5 倍，直接改变占用率上限，因此两张卡在同一个算子上会选出不同的拆法——这正是
「模型里真正起作用的是硬件参数，而不是写死的东西」的检验点。
