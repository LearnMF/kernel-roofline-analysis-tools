# KDA 字节账目：当前生产版（P-65，bw46，2026-10-16）

阶段目标改为“逼近算法访存下限”（`docs/kda_phase2_summary.md` 第 6 节）。这份账目回答：25.8 KB/token·head 的实际访存流量花在哪里，哪些可以去掉。

**数据**（`raw/<T>_<H>/`）：
- `optrace.json`：kra LauncherCapture 记录的一次前向+反向中每次 launcher 调用的接口张量；
- `pmc_read/pmc_write.csv`：hipprof 预设计数器，取 TCC EA 请求数；
- `trace_1.hipkernel.csv`：kernel 耗时。

脚本 `examples/kda/byte_ledger.py` 汇总每个 launcher 和 kernel，`examples/kda/dataflow_graph.py` 汇总中间张量的生产方和消费方。算子按普通前向+反向测（keep-state=0、不开 checkpoint）。

## 1. 四个层次（16K/12，TP8；16K/48 的比例几乎相同）

| 层次 | MB | B/token·head | 相对算法必需量 |
|---|---:|---:|---:|
| 完全融合后不可省的部分（每个外部输入读一次 + 每个输出写一次） | 508 | 2584 | 0.63 倍 |
| 算法必需量（kda_bound P1：含前向保存给反向的状态和反向重读的输入） | 811 | 4124 | 1.0 倍 |
| launcher 接口字节（现有拆分方式） | 4422 | 22490 | 5.45 倍 |
| **实测 HBM** | **5075** | **25812** | **6.26 倍** |

16K/48：实测 20841 MB，即 26501 B/token·head，是必需量的 6.43 倍。

## 2. 每个 launcher（16K/12）

| launcher | µs | 接口读 / 写 MB | 实测 HBM 读 / 写 MB | 实测 / 接口 | 多出部分的来源 |
|---|---:|---|---|---:|---|
| fwd_prep（prep_a + prep_b） | 965 | 202 / 280 | 342 / 405 | **1.55** | prep_a → prep_b 的 R（fp32）和 Kb scratch，约 265 MB |
| fwd_h_o | 510 | 228 / 50 | 228 / 50 | 1.00 | |
| l2n_apply | 156 | 102 / 101 | 102 / 101 | 1.00 | |
| head_gate | 175 | 50 / 101 | 50 / 101 | 1.00 | |
| wu | 651 | 278 / 352 | 288 / 353 | 1.02 | |
| fwd_h_bn（重算递推 + gn_kp） | 445 | 154 / 151 | 154 / 156 | 1.02 | |
| dav | 279 | 126 / 151 | 128 / 151 | 1.01 | |
| dhu_bn | 667 | 253 / 151 | 253 / 151 | 1.00 | |
| wy_s4_split（s4a + s4b） | 1876 | 731 / 353 | 972 / 462 | **1.32** | s4a → s4b 交接的 dA/dwT/db_part（约 200 MB）；h 被两个 kernel 各读一遍（约 100 MB） |
| bwd_tail（+ reduce） | 638 | 456 / 152 | 474 / 153 | 1.03 | |
| **合计** | **6364** | 2580 / 1842 | 2992 / 2083 | 1.15 | |

## 3. 中间张量往返（写 + 所有读，MB，16K/12）

| 交接 | 张量 | 往返 MB | 生产方 → 消费方 |
|---|---|---:|---|
| **s4 → tail** | dq、dk、dg（fp32，各 101 MB） | **604** | wy_s4_split → bwd_tail |
| 前向 prep → 递推 | Wn、Kn、Un、QgA（各 101）、AqkA（50）、Gn | 457 | fwd_prep → fwd_h_o |
| wu 的输出 | Wn、Un、Kn → fwd_h_bn；KgA、Qn、Wb → dhu；kgB → s4 | 705 | wu → 3 个消费方（同一份 W/K 存了 Wn/Wb、Kn/KgA/kgB 多种布局） |
| **l2n_apply** | qn、kn（bf16，各 50 MB，各被读 3 次） | **402**（另读 q、k 共 102） | l2n_apply → wu、wy_s4_split、bwd_tail |
| head_gate | g（fp32） | 302 | head_gate → wu、wy_s4_split |
| 递推的输出 | bN_h、bN_dh（各 201）、Vn（151）、dV2n（101） | 654 | fwd_h_bn、dhu → s4、dav |
| dav 的输出 | dAqk（fp32，101）、dVn、dOT（各 101） | 302 | dav → s4、dhu |
| 前向保存 | Akk、Aqk（bf16，各 25）、rstd | 127 | fwd_prep → wu、s4、dav、l2n、tail |
| launcher 内部的 scratch | prep R/Kb，s4a → s4b | 616 | 见第 2 节 |

## 4. 候选（按“能省的字节 × 可做性”排序，都要求逐位一致）

| # | 做法 | 省掉的 | 代价与风险 | 预期 |
|---|---|---|---|---|
| **1** | **把 l2n_apply 并进它的消费方**：wu、s4a/s4b、tail 改为读原始 q、k 加 rstd，在核内计算 `bf16(x·rstd)`，算法与 l2n_apply 相同（RNE）。 | l2n_apply 读写 203 MB，以及这个 kernel 的 156 µs（16K/48 时 624 µs）。消费方读入的字节不变：q、k 与 qn、kn 一样大。 | 消费方每个元素多 2–4 条 VALU：wu 受访存限制，tail 受 TA 限制，大概率能被掩盖；s4b 情况混合。没有容量问题。按 R53 逐条钉住浮点运算顺序。 | **算子约 −1.5~−2.4%**，16K/12 与 16K/48 都适用 |
| 2 | **把 bwd_tail 并进 s4b 的收尾阶段**：s4b 直接写出 bf16 的 dq/dk 和 dg_raw，不再写 fp32 的 dq/dk/dg。 | 604 MB 的 fp32 往返，加上 tail 重读的 qn、kn、g_raw（约 150 MB），合计最多约 −750 MB（约占 15%）；tail kernel 本身 638 µs。 | tail 里的 l2norm 反向需要整行点积，dg 需要沿 64 行做逆向 cumsum。整个 chunk 的 dq/dk/dg 是 96 KB（fp32），要放在寄存器里（每个 lane 约 +96 VGPR）或 LDS 里，都会和 s4b 现在每个 CU 放 2 个 CTA 的配置冲突（P-37 系列：s4b 在每个 CU 1 个 CTA 时慢 9–26%）。行点积可以按 kt 分块、沿生产版的树结构累加，逐位一致。 | 上限最大，但要先出设计和容量测算 |
| 3 | **wu 少写重复布局**：例如去掉 Wb（dhu 改为从 Wn 经 LDS 转置读取），或合并 Kn/KgA/kgB。 | 每去掉一份约 −101 MB（wu 写 50 + 消费方读 50）。wu 受访存限制，每少写 50 MB 约快 40–50 µs。 | 消费方要增加 LDS 转置。P-74 显示 dhu 对读字节不敏感，所以消费方大概率持平。 | 每份约 −0.6~−0.8% |
| — | 把 head_gate 并进消费方 | 302 MB | gate 的数学计算（expf、IEEE 除法、cumsum）要在 wu、s4a、s4b 里各算一遍。gate 本身已经受 VALU 限制，重复 2–3 遍很可能得不偿失。 | 暂不做 |
| — | 合并 prep_a 和 prep_b，去掉 R scratch | 约 265 MB | 当初拆开就是因为寄存器不够；prep_a 受 VALU 限制。（excess-work 清单 D：收益不确定） | 暂不做 |
| — | 合并 s4a 和 s4b | 约 300 MB | P4=2（融合版）实测比拆分版慢 | 已证伪 |
