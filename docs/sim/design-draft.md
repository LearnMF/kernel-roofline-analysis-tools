# 设计草案：`kra predict` —— 算子 Roofline 上限预测器

> 状态：草案，待专项会确认 `02_requirements_qa.md` 末尾的 7 个问题后再动代码。
> 落地：仓库 <https://github.com/LearnMF/kernel-roofline-analysis-tools>，分支 `feat/roofline-sim`。

## 0. 一句话定位
**输入：硬件参数（可纯手册理论值）+ 算子名与 shape/dtype。输出：该算子在该硬件上的理论时间下限、各资源的屋顶与利用率、瓶颈类型、以及可达 MFU 上限区间和优化方向。**
不读实现源码、不做指令级 replay、不需要真机。实测数据只在「校准模式」里用来统计误差和修正系数。

## 1. 为什么长在现有仓库里而不是新工具
现有 `kra` 已经具备这套模型的 2/3：
- `machines/*.json`（`kra.machine/1`）：已有 `device`（CU 数、clock、bus width、L2、LDS/CU）、`peaks`（hbm_read/write/copy、l2_read、lds_read、mmac_bf16/f16/tf32/f32x8/i8、valu_fma_f32、sfu_exp2、**vmem_issue/vmem_store**、hbm_read_vs_ctas）、`ridge_points_flop_per_byte`、`latency`、`launch`。**每个 peak 同时带实测 `value` 和手册 `theoretical`** —— 这正好天然支持「理论档 / 实测档」双跑。
- `kra/model/bounds.py`：已有 `Machine`（`bw_for(ctas)`、`barrier_cycles`、`chain_cycles`）与 `T_mem_min / T_mem_impl / T_par_* / T_pipe / T_cp_min / T_cp_impl / T_launch` 的下限组合。
- `examples/kda/op_spec.json`（`kra.opspec/1`）：已有 `groups[].{launcher, phase(fwd/bwd), kernels, serial.{steps, chain}}` 和 `vars`（如 `NT = T // 64`）。
- `kra/__main__.py` 的子命令机制：现有 `calibrate / timeline / pmc / analyze`。

缺的只有一件事：**现在 op_spec 的字节数/工作量来自 capture 和 PMC（事后），需要补一条「由公式算出来」的前端。** 因此新增：
```
kra/ops/                 # 算子公式库（operator zoo），每个算子一个插件
  __init__.py            # 注册表：name -> spec builder
  gemm.py  attention.py  kda.py  elementwise.py  ...
kra/model/predict.py     # 纯理论四层下限求解（复用 bounds.py 的屋顶/资源项）
kra/model/pcli.py        # `python -m kra predict`
machines/gfx936-theory.json   # 纯手册值档（与实测档同 schema）
docs/sim/                # 调研 + 设计（= 本目录 03/04）
tests/test_predict.py    # KDA/GEMM 回归
```

## 2. 输入格式

### 2.1 命令行（面向刘恒业这类使用者，最常用）
```
python -m kra predict --machine machines/gfx936-theory.json \
    --op kda --phase fwd+bwd \
    --shape B=1,H=16,T=16384,D=128,chunk=64 --dtype bf16 \
    --tp 8 --out kda_16k_tp8
```

### 2.2 算子描述文件（需要覆盖默认切分/自定义算子时）
沿用并扩展 `kra.opspec/1`，新增可选字段（预测模式专用）：
```json
{
  "schema": "kra.opspec/1",
  "op": "kda", "dtype": "bf16",
  "shape": {"B":1,"H":16,"T":16384,"D":128,"chunk":64},
  "vars": {"NT": "T // chunk"},
  "groups": [
    {"phase":"fwd","launcher":"fwd_h_o",
     "flops":  "2*B*H*NT*chunk*D*D*4",        // 公式字符串，可由插件生成
     "io_bytes":{"read":"...","write":"..."},  // 算法必需流量
     "workset_bytes":"2*D*D*dtype_bytes",      // 片上工作集 -> 决定能否放进 LDS
     "serial":{"steps":"NT","chain":{"mmac_dep":12,"lds":2,"barrier":2}}}
  ]
}
```
**公式用字符串 + `vars` 求值**（现有 op_spec 已是这个风格），使用者只在「新算子」时写，常用算子由 `kra/ops/*` 自动生成。

### 2.3 机器描述
直接复用 `kra.machine/1`。新增 `--peaks theoretical|measured`（默认 `theoretical`，没有 `theoretical` 字段的项退化到 `value` 并在报告里标注）。未来硬件特性（TMA、co-issue、scaling、L1）作为**可选字段**加进 `peaks`/`features`，缺省即视为不具备——满足文进辉「版本化管理、缺啥就不建模啥」的要求。

## 3. 四层下限的「纯理论」算法

记 dtype 字节 `s`，资源集合 `R = {HBM, L2, LDS, MFMA, VALU, TA(vmem_issue), SFU}`。

**L0 `T_theory`（算法下限，完全融合）**
```
T_mem_min   = 必需 HBM 字节 / BW_hbm(有效)            # 只算输入必读 + 输出必写
T_cp_min    = 必需 FLOPs / peak(dtype)                 # MFMA 走 mmac_*，非矩阵部分走 valu/sfu
T_serial_min= steps * chain_min_cycles / clock         # 递推步数 × 不可消除的依赖链
T_theory    = max(T_mem_min, T_cp_min, T_serial_min)   # 假设各资源完全重叠
```
- `T_serial_min` 是本模型相对传统 Roofline 的关键增项：KDA/线性注意力这类 chunk 递推算子，`T` 很大时它可能直接主导（基础 Roofline 会严重高估性能上限）。
- 若 `steps` 的并行度（`B*H` 等）不足以填满 80 个 CU，还要加一条 **占用率屋顶**：`T_occ = T_cp_min * (CU 需求 / CU 实际可用)`。

**L1 `T_split`（容量约束下的切分下限）**
1. 由 `workset_bytes` 与 `lds_per_cu(64 KiB)` / `l2_bytes(8 MiB)` 判断能否单 kernel 融合；
2. 放不下则按 timeloop 式的容量约束搜索可行 tiling → 得到 kernel 数与每个 kernel 的**接口张量**（中间结果要落 HBM 再读回）；
3. `T_split = Σ_k max(接口字节/BW, kernel FLOPs/peak, 本 kernel 串行链) + n_kernel * T_launch`；
4. 输出 `O_handoff = T_split / T_theory`，并标注其中**多少是约束决定的、不可回收**。

**L2 `T_full`（可选，需要实现信息）**：沿用 `bounds.py` 现有 `T_*_impl` 路径，由 capture/PMC 填；预测模式下可用 tile 配置估算一个「乐观指令流」，并明确标注为估算。

**L3 `T_meas`**：仅校准模式。

**瓶颈判定**：对每种资源算 `u_r = T_r / T_bound`，沿用 13 章第 6 章阈值：`u_r ≥ 80%` 饱和；`60% ≤ u_r < 80%` 吞吐受限；全部 `< 60%` 判延迟/依赖受限。

## 4. 输出
`PREFIX.predict.json` + `PREFIX.predict.md`：
1. 算子基本信息：shape、dtype、FLOPs、必需字节、**计算强度（flop/byte）**、与 `ridge_points_flop_per_byte` 的位置关系（余果明确要这个）；
2. 每种资源一条屋顶、各自的时间与利用率，排序给出 **binding resource**；
3. `T_theory` / `T_split`（/ `T_full`）与对应的 **可达 MFU / 带宽利用率上限**；
4. 瓶颈类型 + 不可回收开销占比；
5. 优化方向建议（映射到 13 章第 8/9 章的手段：融合、切分调整、复用提升、依赖链缩短、占用率提升）；
6. **置信度与免责声明**（见 `02_requirements_qa.md` Q3 的分级口径）。

## 5. 误差校准方案
- 用 KDA 已有四层数据回归：16K/TP8 的 `T_theory 0.83ms / T_split 3.35ms / T_full 4.41ms / T_meas 5.32ms` 作为第一个固定用例（写进 `tests/test_predict.py`）；
- 第二个用例：钱永军给的 **PV 推理算子**（前向），验证「偏计算」一侧；
- 第三、四个用例：一个纯访存算子（RMSNorm/elementwise）与一个大 GEMM，覆盖余果/钱永军要求的「一个偏访存、一个偏计算」；
- 每个用例记录「预测 vs 实测」的 `MFU 上限误差` 与 `瓶颈判定是否一致`，在 `results/` 下累积成误差台账 —— 这就是会上要的「误差水位」。

## 6. 里程碑（对齐文进辉「小步快走」）
| M | 内容 | 产出 |
|---|---|---|
| M0 | 文档 + 调研（**当前**） | 13 章教程上传知识库；本调研与设计草案；开专项会对齐 7 个问题 |
| M1 | `kra predict` 骨架 + GEMM/elementwise 两个插件 + `kra validate` | 只给 shape 出「计算强度 + 瓶颈类型 + 理论上限」；BW1000 实测 profile 上下限零违例，留一法 MAPE 优于分档系数（见 `m1-bw1000-validation.md`） |
| M2 | KDA 插件（含递推串行项）+ 容量约束切分 → `T_split` | 复现 KDA 四层下限，回归测试通过 |
| M3 | PV（S4000/S2000 SOC 仿真）跨硬件外推验证 + 误差台账 + 对外使用文档 | 组内其他人可自助使用 |
| M4 | 与余辉端到端 perf model 对接（输出 JSON 契约）；新硬件特性（TMA/co-issue）扩展位 | 可做硬件设计探索 |

## 7. 明确不做（第一版）
- 不解析 FLA/Triton/HIP 源码自动提取算子语义；
- 不做指令级 cycle 仿真（文进辉已否）；
- 不承诺 5% 级误差（那是硅前 EMU 的指标，前提是拿得到指令周期信息）；
- 不把 `T_meas / T_theory` 当作可优化空间对外展示。

---

## 8. 先验算子知识：分层与获取方式（2026-11-12 追加）

### 8.1 为什么必须有先验
工具要算 `T_theory`，必须知道三件无法从 shape 猜出来的事：
1. **必需 FLOPs**（前向 / 反向分别计）；
2. **必需 HBM 字节**（输入必读 + 输出必写，按 dtype 折算；中间量是否必须落回显存属于「实现选择」，归到 `T_split`）；
3. **片上工作集与串行结构**（决定容量约束下能不能一趟做完、递推多少步、依赖链多长）。

第 1、2 项是算法定义；**第 3 项才是本工具区别于传统 roofline 的关键**——传统 roofline 对 KDA 这类 chunk 递推算子失效，正是因为它只有 1、2 没有 3。

### 8.2 工程量分层
| 层 | 内容 | 与算子相关？ | 占工程量 |
|---|---|---|---|
| 通用求解层 | 硬件模型（各级带宽/算力/发射/LDS/launch 固定开销）、屋顶与利用率判定、容量约束 tiling 搜索、串行依赖链求解、四层下限组装、报告 | 无关 | ~80% |
| 算子语义层 | 上述 3 项先验，一类算子一份 | 相关 | ~20%，但不可省 |

定位一句话：**「硬件无关的算子语义库」×「算子无关的硬件资源求解器」的笛卡尔积**。换硬件只换 `machines/*.json`，换算子只加一个插件。

### 8.3 先验的三种获取方式（成本递减）
1. **手写插件**（几十行公式）：有递推/状态传递结构的算子只能走这条路 —— KDA、线性注意力、DSA 稀疏注意力。
2. **家族模板复用**：GEMM / FlashAttention / elementwise / reduce / MoE 各一个模板，换 shape 不改代码。会上提到的算子基本都落在这几个家族内，这是覆盖率的主力。
3. **自动抽取（建议 M2 之后做）**：用 PyTorch `meta` / `FakeTensor` 把算子前反向跑一遍（不占显存、不上卡），自动拿到 op 序列 + 每个 op 的 shape/dtype，于是 FLOPs 与必需字节自动得到，先验只需下沉到 torch op 粒度（matmul/softmax/mul 等工具内置）。
   - 这条路**部分回答了「是否要输入 FLA 代码」**：可以接受一段 torch 前向代码，但只用它做 meta 推导拿 shape 与算子序列，**不解析指令流**；
   - 局限：能自动得到 `T_theory` 的 FLOPs/字节项，**拿不到串行依赖与拆分结构**，第 3 项仍需插件声明或人工标注。

### 8.4 必须守住的边界
**没有插件的算子，工具直接报「不支持」，不做任何猜测性外推。** 给一个没有依据的数字，比不给数字的危害大得多——第一次被发现估错，这个工具就不会再被信任。
