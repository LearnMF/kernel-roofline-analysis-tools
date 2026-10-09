# 设计：kernel 瓶颈分类与性能上限判定工具（kra）

## 1. 要回答的问题

给定一个算子（可能由多个 kernel 组成）和具体 shape：

1. 时间是否真的花在 kernel 里（系统层空泡、launch、同步、通信、传输）？
2. 每个 kernel “应该”受什么限制（先验），“实际”受什么限制（实测）？
3. 离硬件上限还有多远？是“已到顶”“实现层面还有空间”，还是“只能改算法 / 换硬件”？
4. 如果还有空间，下一步该改什么？

结论要能对外解释：每个数字都有采集命令、原件、硬件/工具版本和计算口径。

## 2. 两层分类，反复进行

- **先验分类**（写代码前即可）：由 shape 和算法算出 FLOPs（按 MMAC / VALU / 特殊函数拆分）、最少搬运字节数、串行步数与单步关键链、并行单元数。与实测峰值（L0）组合得到四个下限分量：
  `T_comp`、`T_mem`、`T_cp`（关键路径）、`T_launch`，取最大者为先验类型，`T_下限 = max(...)`。
- **实测分类**：用 SOL%（各单元利用率取最大）、发射率、SQTT 判断当前实现实际受什么限制。
- **差距**：`剩余空间 = T_实测 / T_下限`；先验与实测不一致处，空间通常最大。

注意：分类对象是“单个 kernel + 具体 shape”；每轮优化后重新分类（瓶颈会转移）；kernel 分类之前先做系统层判断。

## 3. 瓶颈分类法

机器可读版本：[`kra/taxonomy.json`](../kra/taxonomy.json)。

| 层 | 含义 | 类别 |
|---|---|---|
| A | kernel 之外 | launch 受限、跨 kernel/跨流依赖、通信、主机-设备传输 |
| B | kernel 内吞吐受限（某单元接近峰值） | 算力：MMAC / VALU / 特殊函数 / 整数地址 / SALU / 发射槽；访存：HBM / L2 / LDS / 原子 |
| C | kernel 内延迟受限（所有单元都不忙） | 并行度不足、访存延迟暴露、串行依赖、同步、分支发散 |
| D | 病因（导致 B/C） | spill、occupancy 受限、LDS bank conflict、非合并访存、地址开销、负载不均衡 |
| 边界 | | 均衡型、问题规模太小 |

## 4. 流程

```
Step1 可信性检查  → 计时与 profiler 是否一致（trust）
Step2 系统层(A)   → L1 timeline：空泡占比、gap 成因、launch 下限
Step3 逐 kernel   → 3a 先验分类与 T_下限(L4)  3b 实测分类(L2/L3)
                    3c 先验 vs 实测、剩余空间   3d D 层病因 → 规则 → 修改建议
Step4 结论/停止   → 到顶(≥80% of T_下限) / 停滞且 <50% 换思路 / 只能改算法
改动后回到 Step1 重新分类
```

## 5. 模块

| 层 | 模块 | 状态 |
|---|---|---|
| L0 硬件标定 | `kra calibrate`：HBM/L2/LDS 带宽，各 dtype MMAC、VALU、SFU 峰值，访存/LDS/MMAC/barrier 延迟，launch 代价，PCIe；ISA 核对 | P0 ✅ |
| L1 系统层 | `kra timeline`：hipprof trace → 窗口切分、空泡、gap 成因、launch 下限、可信性 | P0 ✅ |
| L2 实测分类 | PMC → 各单元 SOL% → B/C 分类 | P1 |
| L3 延迟诊断 | SQTT(xprof + XCompute CLI) → C 类子类型与 D 层病因 | P1/P2 |
| L4 上限模型 | `op_spec` → FLOPs/字节/关键路径 → T_下限、SOL 得分 | P1 |
| L5 结论 | 规则引擎（触发→分析→建议）、停止条件、对外结论模板、水位表 | P2 |
| Agent | skill 封装，诊断结论以方法 ID 交给 hygon-hip-kernel-optimizer 执行 | P3 |

设计原则：确定性脚本负责计算与判定（输出 JSON），大模型只负责解释与选方案；阈值集中在 [`kra/thresholds.json`](../kra/thresholds.json)，每个阈值标明来源与是否经 HCU 校准。

## 6. 参考

- NVIDIA Nsight Compute Compute Triage Guide（D0–D7 决策树、60%/80% 阈值、停止条件）
- NVIDIA SOL-ExecBench / SOLAR（解析上限 `T_SOL = max(FLOPs/峰值, 字节/带宽)`，SOL score）
- KernelPro（先 roofline 分类，再确定性调用“触发→分析→建议”诊断工具）
- AutoKernel（Amdahl 排序；停滞时 <50% 换思路、>80% 接受）
- AMD GEAK v4（脚本管流程、大模型管判断；交替 A/B 噪声带）
- Primus-Turbo kernel-optimize（内部 HCU 迭代规则）
