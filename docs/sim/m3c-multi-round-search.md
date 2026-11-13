# M3c：多轮启发式流程固化

> 工具不再只给一张排好序的表，而是**按流程走一遍**：从未融合的实现出发，每轮定位离自身屋顶
> 最远的因子、取作用在该因子上的手段、预估、采纳最好的那个、记下其余手段的代价和否证理由，
> 直到门禁之下无事可做。KDA 16K/gfx936 的台账如下。

## 1. 台账（实际输出）

| round | 手段 | 对象 | 预估 T | 收益 | 判定 |
|---:|---|---|---:|---:|---|
| 0 | baseline | 每个 op 一个 kernel | 4038 us | — | 实现的起点 |
| 1 | fuse | `wy_bwd_b + tail` | 3460 us | **−14.3%** | 采纳 |
| 2–9 | fuse ×8 | — | 3365 → 2780 us | 每步 2.2%~3.0% | **低于门禁：只能实测确认** |

结论行：

> stopped at 2780 us: no neighbour of the incumbent is faster. `memory:hbm` binds and is above its own
> roof. From round 2 on every step is below the 5% gate (**24.5% of the final time in total**): that part
> is a measurement campaign, not a prediction.

两条有价值的判断，都是流程给的，不是经验给的：

1. **只有一个手段是「大头」**：`wy_bwd_b + tail` 融合 −14.3%，一步到位。
2. **剩下 24.5% 由 8 个各 2~3% 的小步构成**，每一步都在 5% 门禁之下。按方法论，这部分
   **不是工具能下结论的**，它是一个实测清单。这正是过去逐轮试、容易被「静默空结果」骗到的
   那部分工作——工具的价值在于**提前告诉你哪些收益必须实测、总共值多少**。

贪心路径最终和穷举（每个 grouping × formulation）到达同一个候选 2780 us；若两者不一致，
报告会直接写「贪心陷在局部最优，这本身就是结论」。

## 2. 手段是搜索空间里的「一步」，所以不含算子知识

| 手段 | 作用 | 代价 |
|---|---|---|
| `fuse` | 把相邻两个 kernel 合成一个，少一次交接过 HBM | LDS / 寄存器 / barrier |
| `split` | 拆开一个 kernel，缓解容量、减少 spill、提高并发 | 多一次交接 |
| `reform` | 同一个分组换另一种 formulation（重算 vs 存状态、op 变体） | 多余工作量或显存 |

每个候选都由同一个模型计价，所以每轮的「预估」是模型的预估，而不是拍的；而因为手段只是
搜索空间里的一步移动，这套流程对任何有容量图的算子都成立，没有 KDA 专属规则。

## 3. 否证理由是从模型的项里读出来的

```
round 1, fuse wy_bwd_a + wy_bwd_b : latency:hbm +14 us, memory:lds +7 us, sync:barrier +3 us
round 1, fuse gate_cumsum + kkt_qkt: no term changed
round 9, reform state=keep_state   : memory:hbm +83 us, issue:vmem +32 us, memory:l2 +17 us
```

- 「哪一项变差了、变差多少」直接来自候选的项差，所以否证是可核对的。
- 「no term changed」是显式的一类否证：这个手段在这个模型里不动任何项，属于历史实践里
  「静默空结果」那一类——过去要靠人注意到，现在会被打印出来。

## 4. 两个门禁（方法论常数，不是物理常数）

- **收益 < 5% 必须实测**：低于门禁的步骤仍然列出来并累计总量，但判定写明「不是预测」。
- **不动任何项的手段直接否证**，并记录理由。

两者都标注为方法论，不是从数据拟合出来的系数。

## 5. 没做的

- 手段表目前只有 fuse / split / reform，因为这三类是搜索空间里可以严格计价的移动。
  tile 朝向、跨 lane 转置、生产-消费距离这类手段需要 IR 里还没有的信息（循环次数、
  lane 映射），缺就不列，不给无法计价的建议。
- 没有把历史上 KDA 的最优拆法作为搜索先验：台账是从未融合状态自己走出来的。
