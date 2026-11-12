# 业界调研：算子/kernel 级性能预测与仿真

调研目标：会上要求「调研业界算子 Roofline 实现方案，结合现有硬件参数完成模型扩展」。下面按**粒度**分五类，重点看「输入是什么、要不要实测、能不能只给 shape 出上限」。

## 结论先行

1. **没有现成可直接用的「算子 Roofline 预测器」**。公开项目要么是 profiler 驱动的事后 Roofline（必须实测），要么是面向专用加速器的 mapping/DSE 工具（输入是架构描述 + 张量算子，但建模对象是 systolic/PE 阵列，不是 GPU/CU wave 执行），要么是模型/系统级模拟器（粒度太粗）。
2. **最接近我们目标的是学术界的 tile-centric 解析模型**（TileSight / GPU-Tile-Sim 这一路）：它们证明「tile 级资源向量 + 核内 overlap + cache 层级」这套解析建模可以在不做指令仿真的前提下拿到个位数到十几个百分点的误差。但它们需要 tile 配置（即实现信息），属于我们分级输入里的 L2 层。
3. **我们自己 13 章的四层下限模型，粒度恰好在「传统 Roofline」和「tile 级解析模型」之间，而且多出一维业界模型普遍缺失的「串行依赖/递推步数」**（KDA/线性注意力这类 chunk 递推算子是刚需）。因此正确做法是：**以自有模型为主干，从业界借三样东西**——分层 Roofline（HBM/L2/L1/LDS 多条屋顶）、Instruction Roofline（把 TA 发射、LDS 带宽、指令吞吐也画成屋顶）、tile-centric 的「资源向量 + overlap 因子」表示法。

## 分类对照

### A. profiler 驱动的实测 Roofline（事后，必须真机）
| 项目 | 定位 | 输入 | 与我们的关系 |
|---|---|---|---|
| `ROCm/rocprofiler-compute`（原 omniperf） | AMD GPU 的 kernel 级 profiling + 分层 Roofline | 真机运行 | **同生态、最该对齐**：它的指标口径（MFMA/VALU busy、L2/HBM 流量、LDS bank conflict）可直接作为我们 `T_full`/`T_meas` 层的数据来源与字段命名参考 |
| `MegEngine/mperf` | 模块化 kernel 性能分析 toolkit：微架构参数探测 + 分层 Roofline + PMU 采集 + 瓶颈归因 | 真机 + 微基准 | **方法论最值得抄**：它的「先用微基准把各级带宽/发射能力测出来，再画多层屋顶」正是我们 `machines/*.json` 标定流程的成熟版；主要支持 ARM CPU、Mali/Adreno GPU |
| `UoB-HPC/BabelStream` | 可达内存带宽微基准 | 真机 | 用于标定机器文件里的实际 HBM 带宽（我们已经在做类似的事） |

要点：这一类**全部不能预测**，它们回答「现在卡在哪」，不回答「没写之前能到多少」。会上要的是后者。

### B. 解析式 / tile-centric 性能模型（预测，不需要实测，最相关）
| 项目 | 输入 | 建模粒度 | 报告精度（其论文自述） |
|---|---|---|---|
| `TileSight`（arXiv，first-principles tile-centric analytical GPU model） | 算子 + tile 配置 + 硬件参数 | tile 级资源向量、核内 pipeline overlap、cache 层级、跨 GPU 通信 | A100/H200/B200 等 pooled MAPE ≈ 12.35% |
| `GPU-Tile-Sim`（arXiv，tile-centric simulation for LLM HW/SW co-design） | warp-level tile graph（tile 操作 + 依赖） | tile 图仿真 | A100/H100 MAPE 1.22%–8.71% |
| `WaveTune`（arXiv） | kernel 调优参数 | wave-aware 双线性解析模型 | 面向 auto-tuning 的相对预测 |
| `RooflineBench`（arXiv） | 模型 + 设备 | 端侧 LLM 的 OI/Roofline 对比框架 | 偏 benchmark 方法 |

要点：**这是我们要靠拢的方向**。可直接借用的概念：
- 「资源向量」：一个 tile/kernel 对每种资源（MFMA、VALU、TA 发射、LDS、L2、HBM）的工作量向量 → 我们 `T_full` 的通用化表达；
- 「overlap 因子」：各资源之间能重叠多少（取 max 还是求和），正好对应钱永军问的「co-issue、能做多少重叠」；
- 「依赖图」：tile graph 里的依赖链 → 我们 op_spec 里的 `serial.steps / chain`。

注：会上钱永军提到的「GPU Sim」未给出确切项目名，公开世界里同名/近名候选不少（`Accel-Sim`/`GPGPU-Sim`、`mgpusim`、以及上面的 `GPU-Tile-Sim`），**需要在专项会上让他指认**，不要擅自对标。

### C. 指令驱动的微架构模拟器（最准，但要求指令级信息 + 很慢）
| 项目 | 说明 |
|---|---|
| `Accel-Sim` / `GPGPU-Sim` | NV SASS/PTX trace 驱动的 cycle-level GPU 模拟器 |
| `mgpusim` | AMD GCN 架构的多 GPU 模拟器（Go 实现），最贴近 CDNA 生态 |

要点：这正是文进辉说「指令级 replay，这事暂时干不了」的那一档，也是硅前 EMU 5% 误差的来源。**不作为第一版路线**，但 `mgpusim` 的 AMD 执行模型（wave 调度、LDS、CU 资源）对我们理解 `T_full` 的瓶颈项有参考价值。

### D. 加速器 mapping / 设计空间探索（输入张量算子 + 架构描述，输出最优 mapping 与代价）
| 项目 | 说明 | 与我们的关系 |
|---|---|---|
| `NVlabs/timeloop` | dense/sparse 张量代数的解析代价模型 + mapping 搜索 + 代码生成 | 它的「算子 = 嵌套循环 + 数据复用；架构 = 存储层级 + 带宽」抽象，可用来**自动推导切分（我们的 `T_split`）**：给定 LDS/L2 容量搜索可行 tiling，这块值得深入借 |
| `KULeuven-MICAS/zigzag` | ONNX/张量代数到硬件架构的代价估算与 DSE | 同上，更轻 |
| `Maestro`、`SCALE-Sim` | 数据流/systolic 阵列的代价模型 | 架构假设与 GPU CU 差异大，参考价值低 |

要点：**最值得借的一点是「用容量约束自动搜 tiling」**，这正好能把你会上说的「拆成 3 个还是 4 个 kernel 得靠自己判断」部分自动化。

### E. 模型 / 系统级模拟器（粒度太粗，但是我们的下游）
| 项目 | 输入 | 输出 | 算子怎么建模 |
|---|---|---|---|
| `HYGON-AI/hcu-train-simulator` | HF `config.json` + Megatron 并行策略 + 硬件参数 | 显存、计算/通信时间、训练吞吐报告 | GEMM = `2MNK / (fp16_tflops × gemm_efficiency)`，效率按最小维度分档（≤512: 0.15；513–1024: 0.40；1025–2048: 0.75；>2048: 1.00）；通信效率按 flow 大小分档。README 自述未精确建模数据搬移、kernel launch、框架调度、fused kernel 细节 |
| `cli99/llm-analysis` | 模型 + GPU + dtype + 并行配置 | 训练/推理延迟与显存 | 同级别的解析估算 |
| `microsoft/vidur` | 请求负载 + 调度策略（需少量 profiling 引导） | TTFT/TPOT、容量规划 | 推理系统级 |
| `calculon-ai/calculon` | 大规模并行 LLM co-design | 系统级最优配置 | 系统级 |

**与 hcu-train-simulator 的关系（回答你的疑问）**：它和算子仿真确实不是一回事。它把一个算子压成「峰值 × 一个分档系数」，这就是钱永军会上吐槽的「直接乘以系数，可能不太准」。我们要做的就是**替换掉那个系数**：对给定 shape/dtype，给出有依据的可达效率上限与瓶颈归因。所以正确姿势是**我们做算子级 perf model，作为库被这类系统级模拟器（余辉的模型）调用**——这也正是余果会上布置的「把你算子的 perf model 接到他的模型上」。

## 可直接落地的借鉴清单（给设计草案用）
1. 分层 Roofline（HBM / L2 / L1 / LDS 各一条屋顶）+ Instruction Roofline（TA 向量访存发射、LDS、VALU/MFMA 指令吞吐也画成屋顶）→ 覆盖余果说的「L2/L1/LDS 要能体现」，也覆盖我们已知结论「只看 HBM 和算力会漏掉 TA 发射瓶颈」。
2. tile-centric 的「资源向量 + overlap/co-issue 因子」→ 作为 `T_full` 的通用表达，为 TMA/co-issue/scaling 这些未来特性留扩展位（按版本管理，符合文进辉要求）。
3. timeloop 式的容量约束 tiling 搜索 → 自动推导 kernel 切分，产出 `T_split`。
4. mperf 式的微基准标定流程 → 机器文件的「实测档」，与「手册理论档」并存，同一份模型跑两套参数即可给出区间。
5. rocprofiler-compute 的指标口径 → `T_full`/`T_meas` 层字段对齐，便于把实测数据灌回模型做误差统计。
