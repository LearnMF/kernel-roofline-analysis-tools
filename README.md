# kernel-roofline-analysis-tools (kra)

Kernel 瓶颈分类与性能上限判定工具，首个目标平台为海光 HCU gfx936（BW1000）。

它要回答的问题：一个算子（可能由多个 kernel 组成）在某个具体 shape 下，时间花在哪里？每个 kernel 属于哪类瓶颈？离硬件上限还有多远？下一步该改什么？如果已经到顶，依据是什么？

设计说明见 [docs/design.md](docs/design.md)，测量口径见 [docs/methodology.md](docs/methodology.md)，瓶颈分类法见 [kra/taxonomy.json](kra/taxonomy.json)，判定阈值及其来源见 [kra/thresholds.json](kra/thresholds.json)。

## 状态

| 层 | 功能 | 状态 |
|---|---|---|
| L0 | `kra calibrate`：实测可达峰值（HBM/L2/LDS、MMAC 各 dtype、VALU、SFU）、延迟（访存/LDS/MMAC/barrier）、launch 代价、PCIe，并核对 ISA | ✅ P0 |
| L1 | `kra timeline`：hipprof trace → 空泡占比、gap 成因、launch 下限、隐藏的派发开销、可信性检查 | ✅ P0 |
| L2–L5 | 实测分类（PMC SOL%）、SQTT 延迟诊断、先验上限模型（op_spec）、规则引擎与结论 | 计划中（P1–P2） |

## 用法

只依赖 Python 标准库（≥3.8）。L0 需要在目标机上运行，并且能调用 `hipcc`；L1 的分析部分可以在任何机器上运行。

```bash
# L0：在 HCU 节点（容器内）标定
HIP_VISIBLE_DEVICES=0 python -m kra calibrate --id gfx936-bw1000 --out machines/gfx936-bw1000.json

# L1：采集 trace 并分析（示例：KDA 算子，见 examples/kda）
hipprof --hip-trace --output-type 0 -o /tmp/tl/run  python3 examples/kda/kda_iter.py --T 8192 --H 12 --iters 5
python -m kra timeline /tmp/tl/run.json --marker spin_kernel \
       --machine machines/gfx936-bw1000.json --reference-ms 4.84

# 测试
python -m unittest discover -s tests
```

## 目录

```
kra/calibrate/   L0：microbench.hip（HIP 微基准）+ run.py（构建、运行、汇总、ISA 核对）
kra/timeline/    L1：hipprof_json.py（trace 解析）+ analyze.py（窗口、空泡、gap 成因、结论）
kra/taxonomy.json, kra/thresholds.json
machines/        各机器的标定结果（machine.json + 原始输出）
examples/kda/    KDA 算子的采集驱动与脚本
results/         实际分析结果（含原始 trace 压缩包）
docs/            设计与方法说明
tests/           单元测试（不需要 GPU）
```
