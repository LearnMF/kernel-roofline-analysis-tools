# M3a：可达上限、分层算术强度上限、最佳耗时

> 工具的输出口径从此是「这个算子在这张卡上优化后能达到的上限」。
> KDA 16K、heads=96/tp=8、fwd+bwd、gfx936-bw1000：
> **T_attain = 2.780 ms，31.0 TFLOP/s，占 mmac_bf16 峰值的 6.5%**，卡在 **HBM 层**。

## 1. 四段链条，每一段的身份不同

```
T_theory 0.837 ms  --x O_handoff 3.32-->  T_split 2.780 ms  --x O_idle (未标定)-->  T_attain 2.780 ms
```

| 段 | 含义 | 来源 |
|---|---|---|
| `T_theory` | 只看必需工作量的资源地板 | 推导 |
| `O_handoff` | 片上容量强制的 kernel 拆分代价（交接过 HBM、barrier、占用率） | 推导 |
| `T_split` | 最优 formulation 的最优可行拆法 | 推导（搜索） |
| `O_idle` | 这张卡在模型之外还会损失多少（发射限制、cache miss、尾部） | **只能标定**，机器文件 `attainable.idle` |
| `T_attain` | 优秀实现的可达耗时 | 前三段 × 第四段 |

**`O_redund` 不是一个系数。** 搜索会分别给每个 formulation 的 flops 和 bytes 计价，所以
「多算一遍」的代价已经在各自的 `T_split` 里；它在报告里表现为 **formulation 之间的差值
（本例 12.3%）**，这是能动手的东西，而不是一个拟合出来的数。

**`O_idle` 缺就缺，不猜。** 两张卡都没有这项标定，所以报告显式写：

> machine gfx936-bw1000 has no attainable.idle for kda: the attainable ceiling IS the
> capacity-constrained bound, so it is optimistic by whatever this card loses to stalls not
> modelled here. Calibrate it with one measured kernel sequence of this operator family on this card.

有标定时会带着出处一起打印，并注明「这是这张卡、这个算子族的标定，不是通用系数」。

## 2. 分层 roofline 怎么参与上限评估

每一层有自己的必需字节数，于是各有一个强度和一个拐点；**可达耗时取各层屋顶项的最大值**，
排在最上面那一层就是截断上限的那一层。

| level | 带宽 | 字节 | 强度 | 强度上限 | 拐点 | 屋顶耗时 | 受限侧 |
|---|---:|---:|---:|---:|---:|---:|---|
| **hbm** | 1339 GB/s | 3.154 GB | 25.4 | 26.2 | 327.0 | **2529 us** | memory |
| l2 | 6101 GB/s | 3.154 GB | 25.4 | 26.2 | 71.8 | 555 us | memory |
| lds | 15237 GB/s | 1.500 GB | 53.5 | 65.8 | 28.7 | 106 us | compute |

三条可以直接拿去做决策的结论：

1. **HBM 层屋顶 2529 us 占 T_split 的 91%**：这个算子在这张卡上的上限几乎完全由 HBM 流量
   决定，拐点 327 而强度只有 25.4，差 13 倍——不存在「再调一轮算力就追上」的空间，要动的
   是字节数本身（融合、交接留片上、少材料化）。
2. **`强度上限` 列是所有可行拆法里该层能达到的最好强度**，和当前拆法的差额就是片上容量
   吃掉的那部分上限：HBM 层 25.4 对 26.2（−3.0%），LDS 层 53.5 对 65.8（−18.8%）。
   LDS 层差得多但它只占 106 us，所以不是现在该动的地方——这正是分层 roofline 用来
   **排优先级**，而不只是画表。
3. 只保留 HBM 层会得出「只要带宽够就行」的结论，看不到 LDS 流量和交接造成的强度损失；
   所以分层是算上限的必要条件，不是附属展示。

## 3. 机器能力等级进了报告头

```
machine capability **L3**: + operator-dependent structure: attainable compute, memory issue, recurrence step
```

gfx938-bw1102 是 **L2**，报告头直接写缺 `structural.recurrence_step_us` 等，读者就不会把
L2 的报告当成含依赖链项的结果。等级判定按「每一类项需要哪些字段」来定，不是口味排序。

## 4. 留出参照（不回灌）

KDA 生产实测 5.32 ms。相对 `T_attain` 2.780 ms 是 1.91×。

**这个 1.91 不能当作 `O_idle` 标定**：它包含了当前实现自身的不足（多余指令和字节、
没掩藏住的停顿），把它写进机器文件就等于把「今天的实现」固化成「硬件特性」，以后任何
优化都会被这个系数吃掉。`O_idle` 的正确标定对象是一个已经打磨过的 kernel 序列。

## 5. 没计入的东西

- `O_idle`（两张卡都缺）。
- TP/CP 的集合通信时间（机器文件还没有互联段）。
- L2 层字节数在算子没有单独声明时按 HBM 流量取下界，所以 L2 行是偏松的。
