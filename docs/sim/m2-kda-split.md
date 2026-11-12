# M2：KDA 插件，只给 shape 出 T_theory 和 T_split

复现：`python -m kra predict --machine machines/gfx936-bw1000.json --op kda --shape T=16384,H=12 --phase fwd+bwd`

| 形态 | 量 | kra predict（只给 shape） | 参考值（docs/kda_report.md，基于 optrace/pmc） |
| --- | --- | ---: | ---: |
| 16K/TP8 (H=12) | T_theory | 0.837 ms（受串行链限制） | 0.83 |
| | T_split p54a（10-13 拆法） | 3.710 | 3.70 |
| | T_split p82（最终拆法） | 3.357 | 3.35 |
| | 接口字节 / 必需字节 | 5.08x | 5.08x |
| 16K/TP2 (H=48) | T_theory | 2.619 ms（受 HBM 限制） | 2.62 |
| | T_split p54a | 14.40 | 14.38 |
| | T_split p82 | 13.39 | 13.37 |

`T_full`（4.41）和 `T_meas`（5.32）需要实现的指令和字节计数，不在 shape-only 的范围内。

## 怎么算的

- **算子语义**：`kra/ops/kda_ir.py` 是 `examples/kda/kda_ir.py` 搬过来的数据流 IR，包含 39 个张量和 18 个运算。
  它对 p54a 拆法的 16 个形态 × 10 个 launcher 做过逐个读写字节核对，和 optrace 的误差为 0%。
  这次新增了 p77b 和 p82 两种拆法。它们是按 P-82 README 重建的，没有逐张量对过 optrace。
- **T_theory**：融合后的整个算子。必需 HBM 字节中读按 hbm_read 计、写按 hbm_write 计；
  MMAC 按 0.92 的 tile 可达效率计；另外计入 VALU、SFU 和访存发射。
  再与串行链（T/C 步 × 每步延迟，前向 1 遍、反向 2 遍状态递推）取 max。
- **T_split**：拆法里的每个 kernel 只读写自己的接口张量，每个 kernel 单独取 max，再求和。
  递推 kernel 的每步延迟按每 CU 驻留的 CTA 数（1 或 2）查表。
- **机器侧**：新增 `machines/gfx936-bw1000.json` 的 `structural` 块，包括 mmac_tile_eff、vmem_cycles_per_kb
  和 recurrence_step_us，都来自 BW1000 微基准。机器文件里没有某一项时，报告会写明串行链没有计入，不会自己去猜。

## 已知差异与未做

- p82 在生产中只在 H ≤ 20 时启用，按 README 的说法 TP2 应走 p77b（14.10 ms）。但 kda_report 的 TP2 拆分下限是 13.37，
  与 p82 吻合，口径需要回头核对。
- 还没做：按片上容量（LDS 64 KB / 256 VGPR）**自动搜索**拆法。目前只评估给定的拆法。
  `examples/kda/kda_partition.py` 已经有反向分组的 64 种枚举和容量核算，下一步把它接到 IR 上，就能回答"拆成几个 kernel"。
