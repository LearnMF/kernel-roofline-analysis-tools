# P-82：方案 d——前向保存 h，wu 计算 v_new，反向去掉重算递推（2026-10-16，用户已接受显存代价）

**改法**（`examples/kda/p82_save_h.py`）：
- **A. 前向递推 MODE 5**：只输出 Ob 和 HbN。各条路径都配了对应实例：
  - sg+kp：在生产版 MODE 1 的模板上加 BP（从寄存器直接写出 bN）；
  - 只有 kp：与 MODE 2 的 kp 实例相同；
  - 其他情况：`<VL,5>`，宽版 v8o4w 也一样。
- **B. wu VNEW**：
  - W 阶段算出的 A 片段，与 fwd_h_bn 的 wf 在同一 wave、同一 lane 上完全相同，直接保留下来；
  - 在 u 阶段，按 bN_emit 的逆映射把保存的 bN 还原成 hB（v 为行、k 按自然顺序），每次放两个 V-slab，2 × 32 × 132 bf16 正好放进现有的 17.4 KB slab；
  - 按 fwd_h_bn 中 GEMM1 的 kt=0..7 链式 MMAC 计算，再算 `bf_rne(u − acc)`，以 fwd_h_bn 的布局写出 Vn；
  - wu 不再写 Wn/Un/Kn。
  - 第一版因为 pr 循环没有展开（`unroll 1`），we8 按运行时下标访问，被放进了 scratch（80 B，R55）。改为完全展开后，**VGPR 219（≤256，仍是每 CU 2 个 CTA），LDS 17408 B 不变，没有 scratch**。
- **C. k3_g2**：
  - 前向：R4 路线下，只要后面有反向，并且 H ≤ `KDA_G2_SAVE_H_MAX_H`（默认 20），就输出 bN（`KDA_G2_SAVE_H` 默认开）；
  - 反向：把 bN 交给 wu（emit_p1=False），跳过 fwd_h_bn；
  - keep-state 仍走 MODE 2。

**正确性**：全量门禁 GATE_ALL PASS（10/10，对照生产版 g2_r5），FI=3 PASS。

**性能**（bw46 HCU1，同一进程内切换 `KDA_G2_SAVE_H`，测 wall 时间；这一轮测量时还没加 H 门限）：

| 形态 | 前向 | 前向+反向 |
|---|---:|---:|
| **16K/12** | 1866 → 1928 µs（0.968） | 6613 → 6358 µs（**1.040**，−255 µs） |
| **64K/12** | 5843 → 6096（0.959） | 23660 → 22889（**1.034**，−771 µs） |
| 16K/48 | 6143 → 6957（0.883） | 26109 → 25981（1.005） |
| 8K/96 | 6170 → 7013（0.880） | 28724 → 29298（**0.980**） |

各 kernel 与生产版对比（abk46）：

| 形态 | 前向递推（MODE 1 → 5） | wu | 重算递推（去掉） |
|---|---:|---:|---:|
| 16K/12 | 492 → 566 µs（+73） | 652 → 780（+129） | −440 |
| 64K/12 | +275 | +474 | −1510 |
| 16K/48 | 1620 → 2440（**+820**） | +742 | −1539 |

**门限**：H=48/96 时走非 kp 实例，bN 改用 LDS 写出，前向递推慢了 51%，抵掉了收益。所以默认只在 **H ≤ 20** 时启用（kp/BP 实例，即 TP8）；H 更大时退回原路径，显存也不增加。

**显存**（`kda_op_bench_tp.py`，16K/12）：

| | 保存字节（B/token·head） | 峰值 GiB |
|---|---:|---:|
| FLA 默认 | 1032 | 1.65 |
| G2，SAVE_H=0 | 1032 | 1.32 |
| G2，SAVE_H=1 | **1544** | **1.32** |

说明：这个测试程序本身持有作为叶子张量的 q、k，所以 P-77b 保存的 qn、kn 在这里额外多算了 512 B。在 Megatron K3 里，q、k 是 conv 的输出，conv 不会保留它们，按 2026-10-13 的口径，SAVE_H=0 是 520 B，SAVE_H=1 是 1032 B，等于 FLA 默认。前向+反向的峰值不变（1.32 GiB，FLA 为 1.65 GiB）。

## 合入记录（2026-10-16，带 H ≤ 20 门限的版本）

| 位置 | 状态 |
|---|---|
| bw52 `g2_opt` `a41a5db` | 对照合入前备份 `g2_opt_bak_p82`：GATE_ALL PASS（10/10），FI=3 PASS；指纹 `aafc8fa027ea` |
| bw7（bw47 tanbo_mega_k3）`/opt/kda_env/g2_r5` | 指纹一致（`aafc8fa027ea`）；对照 `g2_r5_bak_p82`：GATE_ALL PASS |
| bw46 探针基线 | 已同步到 P-82，指纹 `aafc8fa027ea` |
| flash-train | 未移植：可以复用现有 opaque `state` 端口来放 h，不需要改 ABI，但要改 FT 的 runner 和 torch 封装，另行安排 |

---

# P-82b：saved-h 路线扩展到 TP4 和 TP2 的长序列，并替换 keep-state（2026-10-16）

**先探索**（`b83a.out`，`examples/kda/pipe_cfg.py`，同一进程内轮流切换几组环境变量；全部输出与 base 逐位一致）：

| 形态 | kp 布局 | kp + 保存 h | 只保存 h（非 kp） |
|---|---:|---:|---:|
| 16K/24 | 1.004 | 1.031 | **1.029** |
| 16K/48 | 0.994 | 1.016 | 1.003 |
| 32K/48 | 0.980 | 0.996 | **1.017**（走宽版 v8o4w MODE 5） |
| 8K/96 | 0.994 | 0.989 | 0.983 |

**策略**（`examples/kda/p82b_save_h_policy.py`，只改 Python）：
- 满足以下任一条件时启用 saved-h 路线：
  - H ≤ `KDA_G2_SAVE_H_MAX_H`（24）；
  - `KDA_G2_V8O4W_MIN_H`（40）≤ H ≤ `KDA_G2_SAVE_H_WIDE_MAX_H`（48），并且 NT ≥ `KDA_G2_V8O4W_MIN_NT`（512），即走宽版递推的情况。
- 在这些形态上，keep-state 也改走 saved-h 路线：原来 MODE 2 要交接 bN、Vn、aN 三份，每 token·head 1024 B；现在只交 h，512 B。

**正确性**：全量门禁（8K/12、8K/48、**32K/48**，含 keep-state、gate-extreme、varlen）GATE_ALL PASS，FI=3 PASS。

**性能**（`b82b.out`，KDA_G2_SAVE_H 0 vs 1）：

| 形态 | 前向+反向 |
|---|---:|
| 16K/24（TP4） | 13050 → 12686 µs（**1.029**） |
| 32K/48（TP2） | 51389 → 50406（**1.020**） |
| 64K/48（TP2） | 97908 → 93565（**1.046**） |
| 16K/48 | 1.000（NT=256，不启用） |

keep-state 改走 h 路线后：
- 16K/12：6401 → 6287 µs（1.018）；
- 16K/24：12712 → 12616 µs（1.008）。

**合入**：
- bw52 `g2_opt` `f256f7a`：对照备份 GATE_ALL PASS（含 32K/48），FI=3 PASS；
- bw46 已同步；
- bw7 见下文。

## flash-train 移植（`examples/kda/p82_ft.py`，`ft_p82.out`）：逐位一致，但没有收益，**不提交**

**做法**：
- vendored 的 kernel 源码与 hip_kda 逐字节对齐，直接执行 p82_save_h.py 里的 A、B 两段；
- runner 端：调用方给了 `state` 时，在 P-82b 覆盖的形态上，state 里只放 bN(h)（MODE 5），反向由 wu 生成 Vn（workspace 多 Vn，每 token·head 256 B）；
- 没给 state 时行为不变。不需要改 ABI。

**正确性**：r26 对照 r25 和 hip_kda，default 和 keep-state 两种模式、dense 和 varlen、8K/12、8K/48、32K/48 全部 GATE PASS；X1 PASS；pytest 29 passed。

**性能**（`ft_pipe_cfg.py`，keep-state 模式）：

| 形态 | MODE 2（原 state 内容） | h-only state | k0（无 state） |
|---|---:|---:|---:|
| 16K/12 | 6706 µs | 6765（**0.991**） | 6943（0.966） |
| 16K/24 | 13511 | 13574（**0.995**） | 13806（0.979） |

**结论**：
- FT 里 state 缓冲的大小由公开公式决定，固定为每 token·head 1024 B，只放 h 省不了显存；而在 FT 里 MODE 2 本身就比 h 路线略快（hip_kda 里正好相反，1.018；原因待查，FT 的前向 MODE 2 只比 k0 多 188 µs，hip_kda 多 335 µs），所以移植过去没有收益。
- FT 默认（不传 state）想拿到方案 d 的收益，只有两条路：
  - torch 封装在 H ≤ 24 时默认开启 keep-state：每 token·head 多 1024 B，16K/12 实测快 3.5%；
  - 扩展接口 V2，提供只存 h 的 state 尺寸：多 512 B。
- 两者都需要用户决定。
