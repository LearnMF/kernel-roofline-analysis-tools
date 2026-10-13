# 任务说明：补全拆分评估器的盲区（实测标定）

> 给执行 agent 的完整说明，读完再动手。本任务**只做测量和建模**：不改生产代码，也**不直接修改** `examples/kda/decomp_eval.py`（主 agent 维护）。新的模型项写成独立模块，由主 agent 集成。

---

## 0. 背景与目标

`examples/kda/decomp_eval.py` 是 KDA 算子的拆分评估器，用来在动手之前，预测融合、重算、去掉副本、改 CTA 宽度等改动的耗时。
- **数据来源**：体检卡 `results/kda-ceiling-cards-20261013/cards.json`，16 个形态 × 每个生产 kernel 的实测。
- **模型**：
  ```
  t = max( R · λ ,  F · imb · e )
  ```
  - R：轮数，由驻留数决定；
  - λ：每个 CTA 的延迟；
  - F：现指令流下限；
  - imb：最忙 CU 的负载系数；
  - e：效率。

  细节和回测结果见 `results/kda-decomp-eval-20261013/README.md`。
- **回测**：对融合、重算、占用率与容量这几类已经准确，覆盖 B4b、P-13、P-51、F6、P-17、P-20、P-53。

**已知盲区**：它在两处明显失准，原因各自不同。
1. **P-55（F3a）**：在 s4b 里加入 dAqk 的计算，VGPR 不变（236），模型预测 s4b 快 9%，实测**慢 18–25%**。直接原因是：
   - 代码量从 5177 行涨到 9409 行（s4b 的 4 个 wave 各走一条特化的代码路径）；
   - 主循环从 456 条变成 499 条：VGPR 总数没变，但寄存器压力让编译器多读表、多插等待。

   见 `results/kda-gfx936-20261013-p55-f3a/README.md`，规则 R46。
2. **P-49 在 64K/48**：前向递推改成每个 CTA 处理 64 列，实测快 1.40 倍，模型只给出 1.11。原因是生产版在 64K 时有 L2 颠簸，模型没有 L2 容量项。

此外还有一个**一直按假设处理的量**：驻留数变化后，每个 CTA 的延迟 λ 和吞吐项怎么变。模型目前假设 λ 不随驻留数变化。主 agent 估算“s4a/s4b 每 CU 从 2 个 CTA 提到 3 个”的上限为 +6–9%，靠的就是这个假设，必须实测验证。

**目标**：用探针把下面 4 类效应测出来，写成可以计算的模型项，并让回测在 P-55 和 P-49 64K 上也对得上，同时原有回测不变差。

---

## 1. 边界

- **只读**：不改 `/opt/kda_env/g2_r5`、`/public/home/tanbo/g2_opt`、`ft_*` 这些生产树，也不在其中构建。探针一律在复制出来的树上做。
- **只用 bw7**：`ssh -J bw-zz tanbo@10.16.1.47`，容器 `tanbo_mega_k3`，PYTHONPATH 以 `/opt/kda_env` 开头（否则 `import fla` 会段错误），并设 `KDA_SKIP_FLA_PRUNE=1`。主 agent 在 bw52 上工作。
- **GPU**：先用 `hy-smi` 确认空闲，持有 flock 锁（`/public/home/tanbo/codex_validation/locks/bw7/hcu_<N>.lock`）后再测。最近 GPU 4、6、7 通常空闲，GPU 0–3 和 5 上常驻别人的服务。
- **git**：
  - 只 add 自己新增的路径，不要 `git add -A`；
  - **不要 amend 任何已有提交**，包括别人的，之前出现过 amend 改写他人已推送提交的情况；
  - 不要 push。
- **冻结树**：`cp -a /opt/kda_env/g2_r5 /opt/kda_env/g2_calib`，记录设备代码指纹（应为 `90a91fdd5fdd`，即 P-54a 生产版；提取方法见 `docs/tasks/kda_ceiling_cards_task.md` 第 2.2 节）。所有探针都在 `g2_calib` 的副本上改。
- **构建注意**：`csrc/g2/*_hip.cuh` 是构建时 hipify 自动生成的副本，只改原文件（`*.cuh`）。

---

## 2. 四项探针

### P1. 占用率与延迟、吞吐的关系（最重要，先做）

- **机制**：生产代码里已经有现成做法。`g2_engine.cuh` 中 dhu 的 P-24 限流：
  ```cpp
  auto dyn = [&](auto kern) { return g2_lds_topup(kern, 33 * 1024); };
  hipLaunchKernelGGL(kern, grid, block, dyn(kern), s, p);
  ```
  `g2_lds_topup` 把每个 CTA 的 LDS（静态 + 动态）补到指定字节数，从而压低每 CU 的驻留数。kernel 本身不用改。
- **做法**：
  1. 在 `g2_calib` 副本里，给以下 launcher 加一个环境变量开关：`KDA_PROBE_LDS_<名字>=字节数`。设了就把第 4 个参数从 `0` 换成 `g2_lds_topup(kern, 字节数)`，没设保持 0。
     - `g2_wu.cuh`：wu
     - `g2_wy.cuh`：s4a 和 s4b 两处
     - `g2_bwd_tail.cuh`：tail
     - `g2_dav.cuh`：dav
     - `g2_fwd_prep.cuh`：prep_a 和 prep_b
     - `g2_fwd_v8.cuh`：前向递推窄版，注意这里已有 P-24 的补齐逻辑，要与之兼容
  2. **每个 kernel 测到的驻留数档位**：
     - 生产驻留数（不补齐）；
     - 逐级往下压到 1：64 KB ÷ 补齐字节数，例如补到 22 KB 是 2 个，33 KB 是 1 个；
     - 递推在 192 个 CTA 时，测 4、3、2、1 各档。
  3. **形态**：至少 16K/12 和 16K/48；递推另加 32K/48。
  4. **计时**：每个档位 5 个进程，带显存位置抖动，取设备时间中位数。可以照 `examples/kda/ab_kernels.sh` 的方式，或者沿用体检卡的采集脚本。每个档位同时采一次 PMC（read/write 预设），得到 TA、VALU、HBM 利用率。
- **要回答的问题**：
  - 每个 kernel 的 t(驻留数) 曲线；
  - 每个 CTA 的延迟 λ = t ÷ 轮数随驻留数怎么变：是不变，还是因争用而变大；
  - 驻留数在哪一档以后，吞吐项（TA、HBM）开始封顶。
- **重点**：s4a、s4b 只能往下测（2→1），没法往上。要用往下测的斜率，加上其他能测 1/2/3 档的 kernel 的规律，**给出“2→3 档”收益的有依据的外推区间**，并说明可信度。主 agent 现在估的上限是 +6–9%（λ 不变、被吞吐下限封顶），这个数要被验证或修正。

### P2. 寄存器压力对主循环的影响

- **做法**：在 s4b、s4a、wu 的副本里，于 kernel 开头定义 N 个寄存器，并用 `asm volatile` 让它们一直存活到 kernel 结尾，不参与计算。例如：
  ```cpp
  float r0 = __int_as_float(threadIdx.x), ...;
  // 开头
  asm volatile("" : "+v"(r0), "+v"(r1), ...);
  // ... 原 kernel 体 ...
  // 结尾，再用一次，保证存活
  asm volatile("" : "+v"(r0), "+v"(r1), ...);
  if (r0 == 12345.f) p.<某个输出>[0] = r0 + r1 + ...;
  ```
  N 依次取 0、4、8、16、24、32，直到出现溢出（scratch > 0）为止。
- **每档记录**：
  - 编译器报告的 VGPR 和 scratch；
  - 主循环指令构成：用 `examples/kda/isa_loop_mix.py <file.s> --kernel <符号> [--whole]`，关注 LDS 读、waitcnt、VALU 的变化；
  - 设备时间：16K/12 和 16K/48，每档 5 个进程。
- **要回答的问题**：常驻寄存器每增加 1 个，主循环增加多少条指令、慢多少；拐点在哪里。再用这个关系去解释 P-55：P-55 新增计算的中间变量，在序言阶段大约同时占用 48 个 VGPR。

### P3. 代码量与指令缓存

- **合成 kernel**（`examples/kda/decomp_calib/icache_probe.hip`）：
  - 256 线程，按 wave 号用 switch 分成 4 条完全不同的代码路径，模仿 s4b 的 A=0..3 特化；
  - 每条路径是一个循环，循环体由 M 条互不依赖的 VALU 指令展开而成，M 递增，使总代码量从 8 KB 扫到 128 KB 左右；
  - 每个 CU 放 1–3 个 CTA。
  - 测每条指令的平均周期随总代码量的变化，找出断崖点，并判断指令缓存是每个 CU 独享还是几个 CU 共享。
- **对照真实 kernel**：从 code object 读出 s4b 生产版和 P-55 版的实际代码字节数（符号大小，用 `llvm-objdump` 或 `.size`），放到上面的曲线上，估算 P-55 中指令缓存贡献了多少变慢。
- P-55 的复现树在 bw52 的 `/public/home/tanbo/g2_p55`，补丁脚本是 `examples/kda/p55_f3a.py`（用 `probe` 模式），需要的话可以在 bw7 上复现。

### P4. L2 容量

- **数据**：先用现有数据做。体检卡里每个 kernel 每个形态都有 L2 请求字节和 HBM 字节，可以算出 L2 命中率。前向递推（窄版和宽版）在 16K、32K、64K 上的命中率变化，就是 L2 颠簸的证据。
- **模型**：工作集 = 同时驻留的 CTA 数 × 每个 CTA 需要重复使用的数据量。前向递推里，同一个 head 的 4 个 V-slab CTA 共享 K 侧片段，每步的共享读取见 `g2_fwd_v8.cuh`。当工作集 > L2（8 MB）× 有效系数时，共享读取变成 HBM 读取。
- 必要时补一个合成探针，验证这个有效系数。
- **目标**：解释 P-49 在 64K/48 上的 1.40（模型目前给 1.11），同时对 16K/48 和 32K/48 不能引入大的偏差。

---

## 3. 交付物

1. **探针代码**：`examples/kda/decomp_calib/`，包括 launcher 补丁脚本、寄存器注入补丁脚本、`icache_probe.hip`、采集与分析脚本、README（一条命令重跑）。
2. **新的模型项**：`examples/kda/decomp_terms.py`。只依赖标准库加 `json`，提供以下函数，常量都来自实测标定：
   ```python
   lam_scale(kernel_name, res_old, res_new, shape)            # 驻留数变化后 λ 的倍数（含争用）
   thr_cap(kernel_name, res_new, shape)                       # 驻留数提高后吞吐项的封顶（可选）
   loop_inflation(kernel_name, d_live_vgpr)                   # 常驻寄存器增加带来的主循环倍数
   icache_factor(code_bytes, n_paths, ctas_per_cu)            # 代码量带来的每条指令周期倍数
   l2_extra_hbm_bytes(kernel_name, shape, ctas_resident, ...) # L2 容量不足时多出的 HBM 字节
   ```
   标定数据放在 `results/kda-decomp-calib-20261013/calib.json`。
3. **集成说明和回测**：`results/kda-decomp-calib-20261013/README.md` 里给出：
   - 每个函数应该接到 `decomp_eval.py` 的哪个变换上（merge / add_work / geometry）。由主 agent 实际集成，你也可以附一份建议的 diff 作为参考；
   - 用这些新项重算以下回测，列出“原预测 / 新预测 / 实测”三栏：
     - **P-55**：s4b，16K/12、16K/48、64K/12、8K/96，实测 0.80 / 0.82 / 0.81 / 0.94；
     - **P-49**：64K/48，实测 1.40 / 1.12；
     - **原有回测全部重算**：`python3 examples/kda/decomp_eval.py backtest` 里的每一项。

     你可以在自己的副本或模块里用 monkeypatch 的方式接入新项来完成回测，不要改原文件。
4. **图**，放在 `results/kda-decomp-calib-20261013/fig/`：
   - P1：每个 kernel 的 t 和 λ 随驻留数的曲线，两个形态；
   - P2：每个 kernel 的 VGPR、主循环指令数、时间随注入寄存器数的变化；
   - P3：每条指令周期随代码量的变化，标出 s4b 生产版和 P-55 的位置；
   - P4：前向递推的 L2 命中率随 T 的变化，对比模型。

   图要清楚可读：坐标轴有单位，图例完整。

---

## 4. 验收标准

- **P1**：至少 6 个 kernel × 2 个形态的驻留数曲线；同一档位重复测量偏差 ≤3%；明确给出“s4a/s4b 从 2 档到 3 档”的外推区间，并说明依据。
- **P2**：s4b 和 wu 都给出溢出拐点，以及“每多 1 个常驻寄存器”的主循环和时间斜率。
- **P3**：断崖点的代码量误差在 ±10% 以内。
- **回测**：
  - P-55 的 s4b 新预测，与实测 0.80–0.82（8K/96 为 0.94）的偏差在 ±30% 以内（按变慢幅度计）；
  - P-49 64K 如果加了 L2 项，同样在 ±30% 以内；
  - 原有回测全部保持方向正确，偏差在 ±30% 以内。
- 有测不准或模型解释不了的地方，在 README 里明说，不要硬凑参数。**每个常数都要能追溯到某次实测**。

完成后只提交自己新增的路径，不要 push，然后向用户汇报。

---

## 5. 已知陷阱（来自 kra 的 playbook）

- **显存位置噪声**（R20/R41）：只改分配或代码大小，就会让机器码相同的带宽型 kernel 波动 2–8%。必须多进程加抖动，并报告离散度。
- **occupancy 要实测**（R38）：编译器的 `; Occupancy:` 注释按每 lane 256 个 VGPR 计算，结果偏低。实测在 VGPR 为 200–256 时，每个 SIMD 仍有 3 个 wave 同时驻留。
- **计数高不等于时间代价，探针必须保持其余工作量不变**（R42/R43）：每个探针都要用 `isa_loop_mix.py` 确认，除了被探测的因素，其余指令数不变。尤其要防止编译器把注入的寄存器或补齐逻辑优化掉，或者顺带删掉别的代码。
- **共享 GPU 有干扰**：测量前后都要记录 `hy-smi`。发现别人的进程在用这张卡，整组重测。
- **复杂的远程命令**：写成脚本文件，scp 过去再执行，不要用多层嵌套引号。
