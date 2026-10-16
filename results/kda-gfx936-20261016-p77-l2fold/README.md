# P-77：把 l2n_apply 并进消费方（字节账目候选 1，2026-10-16，bw46 HCU1–3/7，**不合入**）

**改法**（`examples/kda/p77_l2fold.py`；`KDA_G2_L2FOLD` 默认开，置 0 可退回原路径）：
- 反向不再启动 l2n_apply，改为把原始 q、k 和前向保存的 q_rstd、k_rstd 传给 wu、wy_s4_split（s4b）、bwd_tail。各消费方读入数据时按 l2n_apply 的原式自己计算 `bf_rne(bf2f(x) · rstd[tok·H + h])`。
  - wu：3 个读入点；
  - s4b：每轮 kt 的 kk/qq，每个 lane 的 rstd 只读一次；
  - tail：y 复用已经在寄存器里的 r4。
- 只在每个消费方都支持这种读法的路径上启用：HIP tail、S4 拆分（S4=1，P4=3）、kernel 内做 l2norm。
- 接口改动：`g2_wu`、`g2_wy_s4_split_r` 末尾各加 q_rstd、k_rstd，`g2_bwd_tail` 末尾加 raw_y；pybind 和 Python 包装同步修改。

**第一版的 bug**：wu kernel 入口按字段逐个拷贝参数（P-35 用 g2_glb 把指针标成 global），新增的两个 rstd 字段没有拷进去，等于没生效。`p77_unit.py` 逐个消费方单独比对后定位到问题：s4b 和 tail 一致，wu 不一致。补上拷贝后，**全量门禁 GATE_ALL PASS（10/10），FI=3 PASS**。

**性能**：
- 同进程内切换 `KDA_G2_L2FOLD`，测前向+反向 wall 时间（比值 > 1 表示开启更快）：16K/12 为 0.990，64K/12 为 0.995，16K/48 为 0.990，8K/96 为 0.997，全部变慢。
- 各 kernel 与生产版对比（abk46）：

| 形态 | l2n_apply（去掉） | wu | s4b | tail |
|---|---:|---:|---:|---:|
| 16K/12 | −155.8 µs | 651.7 → 770.3（0.846） | 1302.6 → 1443.4（0.902） | 0.992 |
| 64K/12 | −624.9 µs | 2532 → 2997（0.845） | 5114 → 5547（0.922） | 1.007 |
| 16K/48 | −613.1 µs | 2422 → 2935（0.825） | 5339 → 5559（0.960） | 1.038 |

**原因（新规则 R57）**：l2n_apply 的 VALU 只算一遍，而且本身受访存限制，代价很低。并进去以后，同样的逐元素运算（unpack + mul + RNE + pack，每元素约 5 条 VALU）要在 wu（每线程 96 个元素）和 s4b（每个 lane 64 个元素）里各算一遍，这两个 kernel 的 VALU 没有富余，所以分别慢了 15–18% 和 4–10%。
- **R57**：把一个逐元素的生产方并进多个消费方，会把它的 VALU 复制到每个消费方里。只有当各消费方的 VALU 富余都大于复制进来的工作量时才划算；只省字节，不等于省时间。

**剩下的形式（P-77b，需要用户决定）**：qn、kn 只算一次，由前向直接写出。fwd_prep 本来就在寄存器里算过 qn、kn，前向保存时用 qn、kn 替换 q、k（字节数相同），反向就不再需要 l2n_apply，各消费方也不用改。
- 代价：前向多写 101 MB（16K/12），约 +60–90 µs，prep 受 VALU/TA 限制，写出能部分被掩盖。
- 预计净收益：约 −70~−100 µs，约占 1–1.5%。
- 风险：如果调用方在别处还持有 q、k，保存 qn、kn 就会多占约 100 MB/层（16K/TP8）的激活显存。这属于显存语义的变化，需要用户确认。

---

# P-77b：前向写出 qn/kn 并用它们替换 q/k 的保存（2026-10-16，用户同意，**合入**）

**改法**（`examples/kda/p77b_save_qkn.py`；`KDA_G2_SAVE_QKN` 默认开，置 0 可退回原路径）：
- prep_a 里每个 lane 本来就有 4 行 × 8 列的 `bf_rne(bf2f(x) · rstd)`，与 l2n_apply 的算式相同、用的是同一个 rstd。现在每行每个张量多发两条 8 B 写，按行主序 [T,H,128] 写出 qn、kn。
- `fwd_prep` 增加参数 emit_qkn，打开时额外返回 qn、kn。
- k3_g2 的前向保存 qn、kn 来代替 q、k，并置 ctx.qkn。反向检测到 ctx.qkn 就不再启动 l2n_apply；wu、s4、tail 一律不改。
- 适用范围：R4 prep 路线（FI=4），kernel 内做 l2norm，并且后面要跑反向。

**显存**：保存的字节不变（qn、kn 与 q、k 一样大）。在 Megatron-Bridge 的 K3 里，q、k 来自 FLA 的 ShortConvolution，它的 autograd 保存的是卷积的**输入** x（`fla/modules/conv/{triton,cuda,cp}/ops.py` 中的 `save_for_backward(x, weight, …)`），不保留 q、k，所以总显存不变。如果调用方自己还持有 q、k，就会每层多占 2·T·H·128·2 B（16K/TP8 约 100 MB）。

**正确性**：全量门禁 GATE_ALL PASS（10/10，对照生产版 g2_r5），FI=3 PASS。

**性能**（bw46 HCU7）：
- 同进程内切换 `KDA_G2_SAVE_QKN`，wall 时间：

| 形态 | 前向 | 前向+反向 |
|---|---:|---:|
| 16K/12 | 1740 → 1801 µs（0.966） | 6655 → 6544 µs（**1.017**，−111 µs） |
| 64K/12 | 0.961 | 24040 → 23694 µs（**1.015**，−346 µs） |
| 16K/48 | 0.934 | 26410 → 26242 µs（1.006） |
| 8K/96 | 0.937 | 28975 → 28746 µs（1.008） |

- 各 kernel（与生产版对比）：

| 形态 | prep_a | l2n_apply |
|---|---|---|
| 16K/12 | 609 → 657 µs（+47 µs，多写 101 MB） | 去掉 −156 µs |
| 64K/12 | +232 µs | 去掉 −624 µs |

**flash-train 暂不移植**：FT 的 API 里，反向从调用方拿原始 q、k（`BackwardArgs.q/k`），l2n_apply 的结果暂存在 dq/dk 的输出缓冲里。要用 P-77b，前向必须新增 qn/kn 两个输出、反向要改为接收它们，这会改变 PR #556 的接口约定，需要另行决定。FT 内置的 `g2_fwd_prep` 声明保持旧签名；以后同步 kernel 源码时，要在调用处传 `nullptr, nullptr`。

## 合入记录（2026-10-16）

| 位置 | 状态 |
|---|---|
| bw52 `g2_opt` `524594f` | 对照合入前备份 `g2_opt_bak_p77b`：GATE_ALL PASS（10/10），FI=3 PASS |
| bw7（bw47 容器 tanbo_mega_k3）`/opt/kda_env/g2_r5` | 指纹与 bw52 一致（`6e75c288088e`，取 g2_fwd_v8.cuh、g2_fwd_prep.cuh、g2_wu.cuh、g2_bindings.cpp、k3_g2.py、g2_ops.py 六个文件）；对照 `g2_r5_bak_p77b`：GATE_ALL PASS |
| bw46 探针基线 `/opt/kda_env/g2_r5` | 同步到 P-77b，指纹 `6e75c288088e`，备份为 `g2_r5_bak_p77b` |
| flash-train | 不移植：需要改 PR #556 的接口约定（见上文） |
