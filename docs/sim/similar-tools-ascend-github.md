# 昇腾生态和 GitHub 上有没有同类工具？

一句话结论：**「只给硬件参数 + 算子 shape/dtype 就输出理论上限和瓶颈」这件事，昇腾官方开源的工具里没有；它们全是「先在真机上 profiling，再按流水利用率阈值判 bound」。GitHub 上最接近我们目标的不是大厂项目，而是少量个人/学术的「结构化 roofline + 校准」实现。所以我们要做的东西没有现成替代品，但有可直接抄的结构。**

## 一、华为昇腾生态（已核对开源仓代码，不是道听途说）

| 工具 | 仓库 | 是什么 | 要不要真机 | 和我们的关系 |
|---|---|---|---|---|
| **msprof-analyze**（MindStudio Profiler Analyze） | `Ascend/msprof-analyze`（gitcode 主仓，GitHub 有镜像） | 昇腾性能分析工具：**专家建议 advisor**、GPU/NPU 性能比对、集群分析、20+ recipe | **要**（基于 profiling 数据） | **方法论和我们第 6 章高度重合，但只能事后用**。见下面的阈值表 |
| **msadvisor** | `Ascend/msadvisor` | 「专家系统知识库」框架：规则集 / 模糊规则 / 案例匹配 / ML 来识别算子瓶颈并给调优建议，生态开发者自己写 Python 知识库 | 要 | 它是**规则库框架**，不含性能预测模型 |
| **att**（Ascend Training Tools，msprof-analyze 的前身） | `Ascend/att` | 迁移 + 精度 + 性能工具链 | 要 | — |
| **workload_analysis** | `Ascend/workload_analysis` | 「MindStudio 负载建模」众智合作仓 | — | 名字最接近「建模」，但**仓库是空模板**（README 还是 `xxxx` 占位），没有可用内容 |
| **msProf op simulator**（Ascend C 算子仿真流水） | 不开源，随 CANN 工具包发布 | 在 PC 上跑 Ascend C 算子的**指令流水仿真**，输出各 pipe 的流水图 | **不要真机，但要算子源码** | 这是昇腾那边唯一的「仿真」路线，相当于我们 SOC 团队的 C-model/EMU；**输入是算子实现，不是 shape**，属于文进辉说「暂时干不了」的那一档。注：此条基于 CANN 工具链的公开定位，GitHub 上查不到源码，细节需另行核实 |

**关键发现：msprof-analyze 的 advisor 判瓶颈的做法，和我们教程第 6 章的阈值法是同一个思路，但更粗。** 它的规则文件 `advisor/rules/cn/aicore_performance.yaml` 写得很直白：

```yaml
cube_operators:          # Cube（矩阵）算子
  - target: aic_mac_ratio    bound: mac     threshold: 0.8
  - target: aic_mte2_ratio   bound: mte2    threshold: 0.95
fa_operators:            # FlashAttention 类
  - target: aic_mte2_ratio   bound: mac     threshold: 0.8
  - target: aic_fixpipe_ratio bound: fixpipe threshold: 0.75
  - target: aiv_vec_ratio    bound: vec     threshold: 0.75
vector_operators:
  - target: total            bound: vec_mte2_mte3  threshold: 0.9
```

代码（`ai_core_performance_checker.py::del_cube_operator_bound`）的逻辑是：
- 各 pipe 利用率（mac / vec / scalar / mte1 / mte2 / mte3）取平均 → 超过阈值就判成对应的 bound；
- **都没超过时，「优化空间」直接取 `threshold - ratio` 的最大值**；
- 另外有一套「亲和性」硬规则：Cube 内轴不能被 256 整除、FA 的 D 或 S 不能被 128 整除 → 提示不亲和。

对比我们的做法，差距很清楚：
- 相同：**多资源（多 pipe）利用率 + 阈值分类**，和我们 `u_r ≥ 80% 饱和 / 60–80% 吞吐受限 / 全部 <60% 判延迟受限` 一致 —— 说明我们的资源判定法是业界主流，不是自创；
- 不同①：它**必须有 profiling 数据**，回答不了「还没写就想知道能到多少」；
- 不同②：它的「优化空间」就是阈值减利用率，**没有任何下限模型**，更没有「理论/拆分/满载/实测」四层分解和 kernel 间交接开销；
- 不同③：它不建模**串行依赖/递推步数**，也不做容量约束下的切分推导。

> 这条可以直接写进专项会材料：**我们要做的工具，在昇腾开源生态里对应物只有「事后 advisor」，前向预测这块是空的。**

## 二、GitHub 上的同类项目（按「像不像我们要做的东西」排序）

### 1. 最接近：`Vr978/LLM-Inference-Performance`（个人项目，但结构完整，强烈建议读）
自述是「pre-silicon style 的 LLM 推理解析性能模型」，回答的问题和我们一字不差：*给定一个 GPU（真实或假想的）和负载，要多久、瓶颈是哪个资源、换个硬件设计能买到什么*。核心公式：
```
t_kernel = t_fixed + smax_p(t_compute, t_memory)
t_compute = waves × (每 tile FLOPs) / (per-SM 峰值 × 效率)   # 枚举 8 种 tile 形状 + split-K 取最优 → 体现 wave quantization
t_memory  = 必需字节 / (HBM BW × 效率 × min(1, 活跃 SM / 饱和 HBM 所需 SM))
```
可直接搬到我们这边的四个点：
- **`t_fixed` 固定开销项** —— 正好对上 PV 表里那个 18–20us 地板；
- **`smax_p` 平滑 max**（承认计算和访存不可能完美重叠），比我们现在直接取 max 更稳；
- **SM/CU 数不足时带宽打折** `min(1, 活跃CU/饱和所需CU)` —— 我们机器文件里的 `hbm_read_vs_ctas` 就是这条曲线，已有数据可用；
- **「datasheet 参数 / 结构化模型 / 校准后」三档并列报告**，让人看出哪项建模真正有用。

它的实测误差数据非常值得当我们对外的误差锚点（T4，78 个 held-out kernel shape）：
```
纯手册 roofline 43.8%  →  加结构项 24.0%  →  一次校准 12.9%  →  二次校准 9.2%
层内各 op 15.3%，整层 11.7%
```
**即「只拿手册值做 roofline，kernel 级误差就是 40% 量级；要到 10% 必须加结构项再用微基准校准」** —— 这正好支撑我在 `02_requirements_qa.md` 里建议的口径，也能正面回答钱永军的「能不能像硅前 EMU 一样 5%」：不做指令仿真的话，10% 已经是加了校准后的好成绩。

### 2. ML 预测路线：`microsoft/nn-Meter`（364★）
按「kernel（融合后的算子块）」为单位，用**回归模型**预测推理延迟，针对不同边端设备各自采数据训练。思路是「先在设备上大量采样，再用模型外推到没测过的 shape」。
- 对我们的价值：**如果我们积累了足够多的 benchmark 数据（文进辉要求录入的那张表），可以在解析模型之外补一条 ML 校准/外推路线**；
- 局限：它只预测时间，**不给瓶颈归因、不给上限**，解释不了「为什么」和「还能优化多少」，所以不能替代我们的模型。

### 3. 学术解析/仿真：`TileSight`（tile 级解析模型，pooled MAPE ≈12%）、`GPU-Tile-Sim`（tile graph 仿真，MAPE 1.2%–8.7%）
见 `03_industry_survey.md`。需要 tile 配置（实现信息），属于我们分级输入的 L2 层。

### 4. 系统/模型级：`cli99/llm-analysis`、`microsoft/vidur`、`calculon-ai/calculon`、`HYGON-AI/hcu-train-simulator`
粒度到层/GEMM 为止，算子内部一律折成「峰值 × 效率系数」。是我们的**下游消费者**，不是竞品。

### 5. 实测 roofline / profiler：`MegEngine/mperf`、`ROCm/rocprofiler-compute`、`UoB-HPC/BabelStream`
事后分析 + 微基准标定。`mperf` 的「微基准测出各级带宽和发射能力 → 画多层屋顶」就是我们 `kra calibrate` 的成熟版。

### 6. 零散的小项目（可以翻但别指望）
`hemalii3/roofline-advisor`（判断某 kernel 值不值得 offload）、`Spielerr/Kernel-Performance-Prediction`（解析模型预测未知 GPU kernel 执行时间）、`jtang10/llm-simulator`（自述 analytical roofline simulator，仓库内容很少）。

## 三、给专项会的三条论断
1. **昇腾开源生态里没有「前向预测算子上限」的工具**，只有事后 advisor（阈值判 bound + 阈值差当优化空间）；他们的前向路线是闭源的 Ascend C 指令流水仿真，输入是算子源码，不是 shape。
2. **我们的资源阈值判定法与华为 advisor 同构**（他们 mac 0.8 / mte2 0.95 / vec 0.75，我们 80%/60%），这点可以放心对外讲；**我们多出来的是四层下限分解、kernel 间交接开销和串行依赖建模**，这是差异化价值。
3. **误差必须按「手册 roofline → 结构化模型 → 校准后」分档承诺**。公开可查的同类实现（`Vr978/LLM-Inference-Performance`，T4 实测）是 43.8% → 24.0% → 9.2%。拿这个当锚点，比承诺 5% 更站得住。
