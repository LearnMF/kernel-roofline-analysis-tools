# 在 DCU（gfx936 / gfx938）上做算子性能上限预估：Accel-Sim 能用到什么程度

> 结论：**Accel-Sim / GPGPU-Sim 不能直接“定制到 DCU”**——它从 tracer 到 ISA 前端到微架构模型
> 全是 NVIDIA 专用的；但它的**方法论和校验口径**值得照搬，而且它应该被放在我们这套工具的
> **下层**（产出机器文件里的几个字段），而不是上层（直接回答上限）。

## 1. 事实核对（2026-11-13 读上游 README）

| 项目 | 支持的 ISA | 怎么拿输入 | 对 DCU 的可用性 |
|---|---|---|---|
| `accel-sim/accel-sim-framework` | NVIDIA SASS（+ PTX） | **NVBit** 动态插桩真卡 | tracer、前端、配置（H100/A100/QV100）全绑 NVIDIA；无 AMD/ROCm 路径 |
| `accel-sim/gpgpu-sim_distribution` | 同上（GPGPU-Sim 4.x） | Accel-Sim 的 SASS trace | 同上 |
| `sarchlab/mgpusim` | **AMD GCN3** 指令集（稳定）；NVIDIA 支持未就绪 | 自带 AMD runtime/driver 模型 | 唯一现成的 AMD ISA 周期级仿真器，但是 GCN3 一代，到 CDNA/DCU 仍需移植 |

把 Accel-Sim 搬到 DCU 要重写三块：
1. **tracer**：NVBit 没有 AMD 对应物，要用 rocprof 的 thread-trace（ATT/SQTT）或自研插桩；
2. **ISA 前端**：gfx9 的 VALU/MFMA/LDS/标量指令、64 宽 wavefront、s_waitcnt 语义；
3. **微架构模型**：CU/SIMD 发射、LDS banking、L1/L2/MALL、HBM 控制器、XCD 间互连。

这三块加起来是重写一个模拟器的量级，产出还是**周期数**，而我们要的是**上限和该改什么**，
目标函数都不一样。

## 2. 分层：仿真器是机器文件的数据源，不是上限预估器

```
             算子语义（插件：必需 FLOPs / 必需字节 / 依赖链 / 容量 / 并行）
                                 |
   机器文件（L0..L3）  ------>  kra：上限 + 最佳耗时 + 分层强度 + 显存 + 多轮台账   秒级
         ^
         |  只提供这些字段
   指令级仿真 / 真卡微基准：指令与依赖延迟、barrier、HBM 延迟、带宽-CTA 曲线、launch
```

这正好对上余果说的那句「一个指令多少时钟周期，有卡能测出来，没卡就按硬件参数顺延估计」：

- **有卡**：微基准测出来，填进机器文件 → 报告是 L3；
- **没卡**：PV/SOC 仿真或手册参数填 → 报告自动降级为 L1/L2，并在头部声明哪些项没计入
  （见 `machine-capability-levels.md`）。

也就是说「没卡怎么估」这个问题，我们已经有了确定的答案和降级规则，不需要等一个周期级仿真器。

## 3. Accel-Sim 真正值得照搬的东西：校验口径

它的说服力来自 **34,000+ kernel 实例、22 个 benchmark、Pearson 0.99、MAPE 13.4%** 这种报法。
我们的留出验证应该按同样的口径组织：

- 逐 kernel 比，不只比总时间；
- 给出相关系数和误差分布，而不是一个平均百分比；
- 区分「下限不被打破」（零违例）和「离实测多远」（η 分层）。

## 4. 建议的分工（如果要建主仓 + 提 PR）

1. **上层主仓**：算子 pfmodel（本仓这套）——插件 + 机器文件 + 上限/显存/台账输出，JSON 带
   schema 版本，供下游（余晖的 train-simulator 等）直接读。
2. **输入规范先定下来**：`machine-capability-levels.md` 里那张字段清单就是对外接口——
   任何人按这张表交一张卡，工具就能跑；缺字段自动降级，不猜数。
3. **周期级仿真**（如果确有需求）：优先评估 MGPUSim（有 AMD ISA 前端）或硅前仿真组现成的
   内部工具，目标**只**是产出第 2 步那张表里的 L2/L3 字段，不承担上限预估。
4. **gfx938 现在缺的字段**：`structural.recurrence_step_us`、`max_vgpr_per_thread` 等，
   报告里已经逐条声明；补齐这几项就能从 L2 升到 L3。
