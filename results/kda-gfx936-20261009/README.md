# KDA 算子 L1 系统层分析（gfx936 / BW1000，2026-10-09）

## 环境与口径

- 节点 bw7，容器 `tanbo_mega_k3`，`HIP_VISIBLE_DEVICES=0`，DTK 25.10 hipprof `--hip-trace`。
- 算子：Kimi K3 KDA，一次前向 + 反向（`examples/kda/kda_iter.py`，复用 `g2_r5/tests/megatron/kda_op_bench_tp.py` 的输入和调用方式）。G2 = 手写 HIP 实现（R5 树），FLA = Triton 默认路线。
- 每个 shape 测 5 个窗口（每个窗口前用 `torch.cuda._sleep` 做标记），表中取中位数；参考耗时为不开 profiler 时 20 次的中位数。
- 机器标定：[`machines/gfx936-bw1000.json`](../../machines/gfx936-bw1000.json)。原始 trace 在 `raw/*.json.gz`，每个 shape 的完整报告见同目录下的 `*.timeline.md`。

## 结果

| 实现 | T | H | 参考耗时 ms | trace 窗口 ms | 可信比 | kernel 数 | 空泡占比 | 主机提交过晚造成的空泡 µs | 隐藏派发开销（上限） | launch 下限 µs | A 层标记 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| G2 | 8K | 12 | 4.84 | 4.48 | 0.93 | 15 | 0.00% | 0 | 0.73% | 35 | — |
| G2 | 8K | 24 | 8.73 | 8.44 | 0.97 | 15 | 0.00% | 0 | 0.39% | 35 | — |
| G2 | 8K | 48 | 16.52 | 16.18 | 0.98 | 15 | 0.00% | 0 | 0.20% | 35 | — |
| G2 | 16K | 12 | 8.87 | 8.53 | 0.96 | 15 | 0.00% | 0 | 0.38% | 35 | — |
| G2 | 32K | 12 | 16.86 | 16.57 | 0.98 | 15 | 0.00% | 0 | 0.20% | 35 | — |
| FLA | 8K | 12 | 7.48 | 7.00 | 0.94 | 27 | **11.77%** | 829 | 0.65% | 61 | **A.launch** |
| FLA | 8K | 48 | 22.80 | 22.23 | 0.98 | 27 | 0.40% | 86 | 0.24% | 61 | — |

## 结论

1. **G2 不受系统层限制。** 所有 shape 的可见空泡都是 0。主机提交始终领先 GPU 0.5–3.8 ms，每个 kernel 都已在队列中等待。被并入 kernel 时长的派发开销上限是 span 的 0.2–0.73%，即用 HIP graph 或融合 kernel 最多只能省下这么多。因此 G2 超过 99% 的时间花在 kernel 内部，剩余优化空间只能从逐 kernel 分析（L2–L4）中找。
2. **FLA 在小 shape 上受 launch 限制。** 8K/12 时有 11.8% 的时间 GPU 是空闲的，5 处主要间隔各为 130–235 µs，全部属于“主机提交过晚”，即 Triton 在 Python 侧的 launch 开销。到 8K/48 时 kernel 变长，主机能跟上，空泡降到 0.4%。这说明在 8K/12 上，G2 相对 FLA 的提速中约有 0.8 ms（FLA 窗口的 12%）来自系统层，而不是 kernel 本身更快。
3. **可信性。** trace 窗口与参考耗时之比为 0.93–0.98。两者固定相差约 0.3 ms，这是参考计时包含了窗口开头主机发出第一个 kernel 之前的时间（每轮之前都做了同步，队列是空的）。稳态训练中队列不会空，这段时间会被隐藏。

## 局限与后续

- 这是单算子 bench 的结果。在 Megatron 中，主机侧还有其他算子和框架开销，空泡可能不同，需要用同样的工具在模型内采集一次。
- hipprof 会把相互依赖的连续 kernel 之间的派发开销并入 kernel 时长（见 [docs/methodology.md](../../docs/methodology.md) §2.3）。所以 trace 里 kernel 时长本身不能直接用于判断短 kernel 的类型，逐 kernel 分类要用 L2 的口径。
- P1（逐 kernel 先验上限与实测 SOL% 分类）的结果见 [waterlevel/README.md](waterlevel/README.md)。
