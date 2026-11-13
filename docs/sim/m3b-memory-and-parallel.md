# M3b：显存约束、时间-显存帕累托前沿、TP/CP 接口

> 结论先行：加上显存项之后，KDA 16K/TP8 的「最快拆法」不再是唯一答案。
> 61 层驻留时，`state=keep_state` 2780 us / 12.58 GB，`state=recompute` 3038 us / 4.14 GB：
> **+9.3% 时间换 −67% 显存**。只按时间排序时，工具永远选前者。

## 1. 为什么必须有这一项

拆 kernel 和「前向存下来而不是反向重算」都是用显存换时间。训练里显存是硬约束：
放不下的方案不是慢，是跑不起来。之前的搜索只按 `T_split` 排序，所以它结构性地偏向
`keep_state`——存状态只会减少工作量，不会增加任何被计价的代价。

## 2. 怎么算的（算子无关）

`kra/model/memory.py` 只读拆分搜索已经在用的那张容量图，不新增任何算子知识：

| 项 | 规则 |
|---|---|
| 值的生存期 | 从产出它的 kernel 到最后一个读它的 kernel |
| kernel 内部的交接 | 不过 HBM，不计；只有 `footprint()` 判定溢出的那些，在它所在 kernel 期间常驻 |
| 调用方张量 | 图的输入和算子的输出全程常驻 |
| **stage peak** | 各阶段活跃字节的**最大值**，不是所有张量之和 |
| **saved for backward** | 跨前向/反向边界活着的那组值（插件按 op 声明 `phase` 时才导出，否则显式报「未导出」而不是猜） |
| 驻留量 | `peak + saved × (layers − 1)`：同时只有一层的 workspace 峰值，但每层的 save 集都活着 |

预算来自机器文件自己的 `device.global_mem_bytes`，或者 `--mem-budget GB`（per-rank）。
没有任何关于权重、优化器状态、其它层的假设——报告的是这个算子自己的占用，以及它对
哪个预算而言超了。

两条帕累托前沿都输出：按驻留量，和按 save 集字节（后者才是乘层数的那一项）。

## 3. KDA 的数（gfx936-bw1000，T=16384，heads=96/tp=8，fwd+bwd）

| formulation | T_split | stage peak | saved for bwd | 61 层驻留 |
|---|---:|---:|---:|---:|
| `dAqk_s4b/keep_state` | **2780 us** | 1.200 GB | 0.190 GB | 12.58 GB |
| `dAqk_s4b/recompute` | 3038 us | 1.200 GB | **0.049 GB** | 4.14 GB |

- stage peak 在两种 formulation 下相同，峰值出现在反向的 `scan_dh`（`h`、`g`、`dh`、
  `v_new`…）。**这正是说明为什么不能只报峰值**：存还是重算的差别不在单层峰值上，
  而在跨前反向那一刀上（`h`、`v_new`、`Tinv`、`Aqk`、`rstd_q`、`rstd_k`、`beta`）。
- `--mem-budget 8 --layers 61` 时，`keep_state` 全部候选被判超预算，最优解自动切到
  `recompute`，时间变长。这就是工具该替人回答的那个决策。
- 预算下没有任何候选可行时，不再继续排序，直接拒绝并说明是片上容量还是显存预算。

## 4. TP / CP 接口

- `Op.parallel` 声明插件**实现了**哪些并行模式（shape 键 `tp` / `cp`）。
  未声明的模式传进来直接拒绝，不按通用规则硬算。
- KDA：`parallel = ("tp",)`。TP 切 head 轴，每个 rank 对 `H/tp` 个 head 跑完整递推，
  **算子自身不通信**（all-reduce 属于后面那个 row-parallel linear）。
  `--shape T=16384,heads=96,tp=8` 与 `H=12` 等价；两者同时给且不一致时报错。
- CP 只留接口：KDA 的 CP 是分相形式（本地只算 state → all_gather → merge → 再算
  `o`/`h`，反向 `compress_h0` / `expand_h0`），这是 **KDA 的一种 formulation**，
  和 ring/zigzag attention 没有共通之处，所以它的位置在插件里，而不是求解器里。
  未实现之前 `cp>1` 报错而不是用错误的数据流计价。

## 5. 留出验证（待做）

`TP4/CP1/32K 1.84 s/step、27.5 GB` 对 `TP4/CP2/32K 2.21 s/step、16.5 GB`
（+20% 时间 / −40% 显存）是帕累托前沿的理想留出点，但要等 KDA 的 CP formulation
落地后才能核对。这组数只做留出核对，不回头作为模型输入——否则就是过拟合。

## 6. 没有计入的东西

- 分配器碎片、显存池行为；报告的是张量字节数。
- 其它层、权重、优化器状态：只有在 `--mem-budget` 里由调用方给出。
- TP/CP 的集合通信时间：机器文件目前没有互联带宽/延迟段，缺就显式说明未计入。
