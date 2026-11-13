# M2e：依赖链与 barrier 延迟进入 `T_split`

机器文件里早已由 `kra calibrate` 实测好的两项延迟——`latency.mmac_bf16_dependent`（`k_mmac_latency`）与
`latency.barrier`（`barrier` 微基准）——此前只被事后归因（`kra/model/bounds.py`）使用。本轮把它们接进
下限求解器，**仍然只用机器文件实测值，算子侧不引入任何新常数**。

## 两个新下限项

### 1. `serial:chain` 的通用下限（依赖链）

`Work.serial` 里的每一步都依赖上一步的结果。对任意一步，数据流本身能保证的最弱结论是：

- 这一步的乘法要等上一步的乘法结果，所以**至少暴露一次依赖 MMAC 延迟**；
- 状态在步与步之间经 LDS 交给整个 block，所以**至少一次 barrier**。

于是每步下限 = `mmac_bf16_dependent + barrier(block 大小)`（`TheoryMachine.dep_step_s`）。
与已有的、算子专属的 `structural.recurrence_step_us` 取**最大值**：

| 机器 | 已标定步长（1 CTA/CU） | 通用下限 | 实际采用 |
|---|---:|---:|---|
| gfx936 | 1.241 us | 29.34 + 34.67 = 64 ns | 已标定值（通用下限约为其 1/19，不改变结果） |
| gfx938 | 无 | 33.86 + 40.00 = 74 ns | **通用下限**（报告里注明「比已标定值松」） |

gfx938 不再是「串行链完全不计」，而是用它**自己**测出的两个延迟给出一个更松但成立的下限；没有借用
gfx936 的 1.241 us。两项延迟都缺时才退回原来的「串行链未计入」声明。

### 2. `sync:barrier`（kernel 内交接的同步）

kernel 内一个 op 把值经 LDS（或落盘再读回）交给下一个 op 时，整个 block 必须先同步。`footprint()`
现在按语义数出每个 tile 至少要几次 barrier：

- 暂存到 LDS 的操作数（寄存器放不下的，`operands_moved_to_lds`）；
- 自己有 LDS tile 的 op（`core_lds > 0`）；
- 驻留 LDS 的交接张量；
- kernel 内落盘的交接张量。

驻留在寄存器里的交接是 lane 私有的，不需要 barrier。时间 =
`每 CU 的 tile 批数 × barrier 次数 × barrier(block 大小)`，其中批数 = `ceil(tiles / CU) / 每 CU 驻留 CTA 数`：
同一 CU 上其他驻留 CTA 可以掩藏一个 CTA 的 barrier，所以只按批计。barrier 延迟按 block 大小取**最近的
已标定档位，向上取**，超过最大档位就用最大档位，不外推。

**这是下限而非估计**：真实 kernel 每个 tile 内往往还有 K 维循环，每轮一次 barrier；IR 里没有循环次数，
所以只计「每个交接一次」这一最弱结论。

## 结果（16K，fwd+bwd，占用率开）

| | 之前（M2d） | 现在 | 最优拆法 |
|---|---:|---:|---|
| gfx936 H=12 | 2.780 ms | 2.780 ms | 7 kernel，dAqk_s4b + keep_state |
| gfx936 H=48 | 11.020 ms | 11.020 ms | 同上 |
| gfx938 H=12 | 1.891 ms | 1.891 ms | 同上 |
| gfx938 H=48 | 7.264 ms | 7.264 ms | 同上 |
| gfx936 生产分组 | 3691 us，286/1024 | 3691 us，286/1024 | 手写 IR 3729 us |
| gfx938 生产分组 | — | 2455 us，273/1024 | 手写 IR 2472 us |

**两个新项在 KDA 上都不起约束作用**，原因可以读出来：

| gfx938 H=12 最优拆法 | `T_us` | `sync:barrier` | `serial:chain` | 卡在 |
|---|---:|---:|---:|---|
| 前向融合 kernel | 247.4 | 8.6 | — | HBM |
| `scan_fwd_o` | 266.7 | — | 18.9 | HBM |
| `re_gate+re_wy+dav` | 381.2 | 1.7 | — | HBM |
| `scan_dh` | 255.9 | — | 18.9 | HBM |
| `wy_bwd_a+wy_bwd_b+tail` | 604.2 | 8.6 | — | HBM |

- barrier 项最大不到所在 kernel 的 4%：KDA 每个 kernel 的 tile 数多、每 CU 批数少；
- gfx938 的通用串行下限（256 步 × 74 ns ≈ 19 us）只有 HBM 项的 7%。gfx936 上真实步长是 1.241 us，
  是通用下限的 19 倍——**一步 KDA 递推远不止一次依赖 MMAC + 一次 barrier**。所以 gfx938 上 scan
  kernel 的下限仍然偏松，补 `recurrence_step_us` 的标定才能收紧，这项不能从 gfx936 搬。

结论：接入后下限成立、可解释，对 KDA 结果无影响；它们对「tile 少、交接多」或「依赖链长、带宽不紧」的
算子才会起作用（小 batch 解码、长序列递推）。

## 测试

- `test_barrier_latency_is_taken_at_the_nearest_calibrated_block_size`：向上取档、不外推、缺字段返回 None；
- `test_the_generic_step_floor_is_below_the_calibrated_step`：通用下限 < 已标定步长，max 保留标定值；
- `test_a_card_without_structural_still_bounds_the_serial_chain`：gfx938 串行链 > 0 并注明来源；
- `test_chain_falls_back_to_the_generic_floor_then_drops_out`：无 `structural` 用通用下限，再去掉
  `latency` 则声明未计入；
- `test_barriers_are_charged_per_tile_batch_and_only_when_derived`：未推出 barrier 次数时不计，次数线性；
- `test_fusing_more_handoffs_through_lds_costs_more_barriers`：融合引入 LDS 交接后 barrier 次数增加。
