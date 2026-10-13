# F8 调研：块内求逆与 KDA 训练算子的公开实现（2026-10-13，第二轮）

**目的**：回答两个问题：有没有比我们现在的块内求逆更好的算法？公开实现里，有没有能改变 gfx936 上结论的设计？

第一轮调研见 `results/kda-algo-explore-20261013/plan.md` 第 5 节，覆盖了 FlashKDA、NVlabs/kda、cuLA、Int21、FLA 0.5.x。本轮补充了以下来源：
- Kimi Linear 论文正文与附录；
- FLA 主线的最新代码，包括新增的 token 并行 kernel、tilelang 后端、昇腾后端；
- **cuDNN Frontend 新开源的 KDA 前向和反向 kernel**：有 SM100 的 FROST 和 Hopper 的 CUDA 两套。这个项目正是我们集成契约所在的项目（PR #556）。

## 1. 来源与要点

| 来源 | 平台 | chunk | 块内求逆 | 拆分方式 | 关键设计 |
|---|---|---|---|---|---|
| Kimi Linear 论文（arXiv 2510.26692，§3.1、附录 C） | — | 64 | UT 变换：M = (I + StrictTril(diag(β)(Γ⊙K)(K/Γ)ᵀ))⁻¹ diag(β)，**前代求逆**；P 的写法借自 Comba，少一次求逆 | 参考实现 | 与我们的算法相同 |
| FLA 主线 `fla/ops/kda` | Triton | 64，子块 16 | 对角 16×16 块前代，再用 Ai10 = −Ai11·A10·Ai00 这类矩阵乘逐级合并（`kkt_solve` 融合 kernel） | 前向 intra/wy/h/o；反向 dAv、wy_dqkg 融合、bwd_intra | dAkk = −Tᵀ(mask(dA·β))Tᵀ，两次 64×64 矩阵乘。新增 token 并行的 intra kernel（每个 token 一个 program，减少 varlen 的填充浪费） |
| FLA 昇腾后端 `backends/triton_ascend` | Ascend NPU | 64，**子块 32** | 对角块前代单独一个 kernel，块间合并放进融合 kernel；子块的门控以**子块中点**为参考点，提高数值稳定性 | 同 FLA | 非 NVIDIA 硬件的适配：只调整了子块大小和 kernel 切分，**算法没变** |
| **cuDNN FE FROST**（`linear_attention/frost`，SM100/103/107） | Blackwell | **16** | **分块 MMA 求逆**：4×4 对角块用两项级数 D4 = I − L4 + L4² − L4·L4²（L4 幂零），再做 4→8→16 的修正 C ← −D⁻¹CA⁻¹；没有标量前代 | **每个（序列，head）一个 CTA 走完整个序列**：前向一个 kernel；反向一个 bprop kernel，加一个 recompute kernel 生成每块的状态检查点；（序列 × head）数不足以占满 SM 时，用 **piece chain**：把序列切成段，每段先求摘要 H_j（零初值状态）和转移矩阵 M_j，以 fp32 串接 X_{j+1} = X_j·M_j + H_j，再给每段注入准确的初始状态 | **不显式构造 W**：v_new = T⁻¹·β(V − K_decay·S)；状态以 fp32 放在 TMEM；16 个 warp 专职分工 + TMA + 异步 MMA |
| **cuDNN FE Hopper CUDA**（`linear_attention/hopper`，`kda_hopper_cuda`） | H100 | 64，子块 16 | 16×16 前代 + 分块合并（同 FLA） | **反向只有 3 个主 kernel**：<br>① `k_prep`，按 chunk 并行：A、−A·kb、A·vβ、kg、qg、dv_local = tril(Aqk)ᵀdO，以及 **P = qgᵀdO**；<br>② `k_scan`：**前向状态递推和反向梯度递推放在同一个 grid 里并发执行**，两个方向互不依赖；<br>③ `k_bwd`，按 chunk 并行：dAqk + 状态梯度乘积 + WY 反向 + 块内 + 反向 cumsum 全部融合 | 每个 CTA 用 220–230 KB SMEM；机器生成（“Kernel Factory”）；比 cuTile 版反向快 5.58 倍。P 在并行阶段预先算好，**缩短了反向递推的每一步** |

## 2. 结论

1. **块内求逆本身没有更好的算法。** 所有实现用的都是“小对角块求逆 + 块间矩阵乘合并”，差别只在于：
   - 对角块是用标量前代，还是用 MMA 级数；
   - 子块取 16 还是 32。

   F8 原先设想的“换掉块内求逆就能动到 4.5–5.7 倍那一段”**不成立**。真正决定接口字节的，是融合粒度，而融合粒度由片上存储容量决定：
   - Hopper 版每个 CTA 用 225 KB SMEM，所以能把 dav pass 1、s4a、s4b、tail 合成一个 kernel；
   - FROST 靠 228 KB SMEM 加 256 KB TMEM，做到每个 head 一个 kernel；
   - gfx936 每个 CU 只有 64 KB LDS，F6 已证明 s4a+s4b 都放不下。
2. **FROST 的整体融合无法迁移**，原因与第一轮的判断一致：每个 head 只有一个 CTA，串行走 T/16 步，每一步有约 14 个 GEMM。它依赖 TMEM、异步 MMA 和按 warp 分工才能把单步做快。按我们递推 kernel 的实测效率（单步 MMAC 利用率约 10%）估算，16K 时每个 head 要约 12 ms，即使用 piece chain 切段，也赶不上现在的实现。
3. **可以借鉴的三点**，都不依赖大片上存储：

   | 编号 | 来源 | 做法 | 对我们的意义 | 数值类别 | 初步判断 |
   |---|---|---|---|---|---|
   | **C1：两个递推并发** | Hopper `k_scan` | 反向的 fwd_h_bn（重算 h、v_new）和 dhu（dh 递推）互不依赖，可以同时跑。前提是把 dav 拆开：pass 2（dVn、dOT，不依赖 Vn）放在两个递推之前，pass 1（dAqk，需要 Vn）放在之后 | TP8 时两个递推各只用 48 个 CTA（80 个 CU 里有 32 个空着），而且串行相加：16K/12 上 503 + 656 µs。并发后，理想情况下接近较长的那个 | A 类，逐位一致：只改调度，不改任何计算 | **新候选**。之前测过“递推和并行 kernel 并发”（P-41、F9：争用把收益吃掉），**两个递推之间并发没有测过**。可以用现有的 `f9_contention_probe.py` 先测上限 |
   | C2：预先算好 P = qgᵀdO | Hopper `k_prep` | dhu 每一步原来要读 Qn 和 dOT 片段，再做一次 GEMM；改为在并行阶段算出每块的 P（128×128），dhu 每步只读 P 的一个 slab | dhu 每步少一个 GEMM，读取量从约 20 KB 降到约 8 KB（bf16）。TP8 下 dhu 受延迟限制 | P 存成 bf16 时属于 **B 类**（多一次舍入）；存 fp32 时每 token·head 多 1 KB 读写 | 中等，需要先评估精度 |
   | C3：分块 MMA 求逆 | FROST | prep 里 16×16 对角块的前代（每个 lane 80 次 shuffle + 64 次 FMA）改成 MMA 级数，再用 MMA 修正 | prep_a 在 TP2 下受 VALU 限制。之前第 14 轮估过类似改法，约 0.5–1% | **B 类**（舍入顺序变了） | 小 |

   第一轮已经排除或否决的（chunk 16 用于训练反向、近似 exp、持久化 megakernel 等），本轮没有新证据推翻。

## 3. 建议
1. **先测 C1 的上限**：成本最低（一个探针），A 类，结论直接。用 `f9_contention_probe.py` 把 fwd_h_bn 和 dhu 放在两条 stream 上并发回放，测 16K/12、64K/12、16K/48 三个形态。
   - 并发总时间比串行少 15% 以上，就做调度改动（拆 dav，调整 launcher 的顺序），过逐位门禁；
   - 否则 C1 关闭。
2. C2、C3 都属于 B 类，与“数值格式”方向（fp32 中间结果改用 bf16 交接）一起，放进 B 类的精度评估里统一处理。
3. F8（换块内算法）**正式关闭**：没有可迁移的更优算法，收益上限在 2–4%（第一轮的容量复核结论不变）。
