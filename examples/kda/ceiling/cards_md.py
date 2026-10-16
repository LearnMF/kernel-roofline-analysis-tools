"""Emit results/kda-ceiling-cards-20261013/cards.md from cards.json.

Layout per the task: summary table first, then per shape three tables
(utilization / floor decomposition / work gaps), verification appendix.
Unreliable cells render as "-- r:reason" or gray numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # examples/kda (kda_bound)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root (kra)
from kda_bound import bounds as algo_bounds                     # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = Path(__import__("os").environ.get("KRA_CARDS_RES", ROOT / "results" / "kda-ceiling-cards-20261013"))
CARDS = json.load(open(RES / "cards.json"))

MEASURED = {(16384, 48): 26.43, (16384, 24): 13.09, (16384, 12): 6.75,
            (32768, 48): 50.13, (32768, 24): 25.63, (32768, 12): 12.94,
            (65536, 48): 98.37, (65536, 24): 48.36, (65536, 12): 24.38}


def g(x, nd=2, dash="—"):
    return f"{x:.{nd}f}" if x is not None else dash


EVIDENCE = {
    "dav": "LDS 冲突消融 +0.9–2.4%→算子≤0.1%（已排除）；窄读保留",
    "wu": "冲突+swizzle 消融 +0.5–2.5%→≤0.25%（已排除）；w_gap≈1",
    "fwd_h_v8o4": "转置冲突消融 0 收益（排除）；h 寄存器复用 P-53 +0–0.3%算子（已合入）",
    "fwd_h_v8o4w": "同上（v8o4w 宽变体）",
    "dhu_v8b": "16B 写出消融 ≤0.23%；转置冲突 0（均排除）",
    "dhu_v8b2": "同上",
    "dhu_v8b4": "同上",
    "fwd_prep_a": "R scratch 消融 ≈0（排除）；64 位地址 P-46 −0.8%~+0.45%",
    "fwd_prep_b": "同 prep_a",
    "bwd_tail": "P-54a 已合入 +3.2–5.3%→算子 0.3–0.5%；加宽 4 列 P-54b 慢 1–2.7%",
    "gate": "—",
    "l2n_apply": "已贴 HBM 顶",
    "exp2": "—",
    "bwd_tail_reduce": "—",
    "wy_s4a": "MMAC K 维置换 25.7% 不逐位一致 → 归 B 类约束",
    "wy_s4b": "同 s4a",
}


def md() -> str:
    L = []
    L += ["# KDA G2 (P-52) 逐 kernel 见顶体检卡 — gfx936/bw7", "",
          "冻结树 `/opt/kda_env/g2_card_p52`（指纹 `35b7bd3a8862`）。"
          "口径：每 kernel 5 个抖动进程的 trace 中位数；"
          "带宽/算力取 kra 标定可达值；FLOPs = SQ_INSTS_MMOP × ISA 标注的单条 FLOPs "
          "（纯 bf16 = 8192，纯 fp32x8 = 4096，混合按 ISA 静态占比加权，卡片注明）。",
          "",
          "> **口径提醒**：表2 的“下限”= **现有指令流全速发射所需时间**"
          "（实测 TA/VALU/LDS 活跃周期 ÷ 活跃 CU ÷ 时钟），"
          "它说明“这条指令流已经多接近满速”，**不是算子真下限**；"
          "真正的余量看表3 的指令/字节放大倍数与总表的算子第一性下限。"
          "LDS 冲突计数偏高不等于病灶——消融实验（保工作量不变、消去全部冲突）"
          "在 dav/wu/递推上只快 0.9–2.4% / 0.5–2.5% / 0（R42），大部分冲突被其他瓶颈掩盖。"
          "TA 放大高也不等于“减指令就变快”：能兑现的是去掉纯浪费的指令"
          "（全 wave 读同址、2B 读、重复 LDS 读——P-53/P-54a 验证），"
          "单纯加宽访问或减指令数不一定变快（R45，P-54b 加宽反而慢 1–2.7%）。", "",
          "## 总表（每形态合计）", "",
          "| 形态 | Σ kernel 时间 (ms) | Σ 现指令流下限 (ms) | Σ 接口下限 (ms) | 算子第一性下限 (ms) | 实测/Σ流下限 | Σ流/Σ接口 | Σ接口/第一性 | 主要限制（按时间占比） |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for s, c in CARDS.items():
        T, H = c["T"], c["H"]
        tot = sum(v["t_us"] for v in c["cards"].values()) / 1e3
        fl = sum(v["floor_us"] for v in c["cards"].values()) / 1e3
        b = algo_bounds("BW1000", T, H)
        share = {}
        for k, v in c["cards"].items():
            share[v["binding"]] = share.get(v["binding"], 0) + v["t_us"] / 1e3
        bind = ", ".join(f"{k} {v / tot * 100:.0f}%" for k, v in
                         sorted(share.items(), key=lambda x: -x[1]) if v / tot > 0.05)
        fi = sum(v["f_iface_us"] for v in c["cards"].values() if v.get("f_iface_us")) / 1e3
        L.append(f"| {T // 1024}K/H{H} | {tot:.2f} | {fl:.2f} | {fi:.2f} | {b['bound'] * 1e3:.2f} "
                 f"({b['binding']}) | {tot / fl:.2f}x | {fl / fi:.2f}x | "
                 f"{fi / (b['bound'] * 1e3):.2f}x | {bind} |")
    L += ["", "图A：每形态一张分层 roofline（`figA_<T>_<H>.png`）；图B/C/D 见 `figB.png`/`figC.png`/`figD.png`。",
          "无 MMAC 的搬运型 kernel 不上 roofline，在图A脚注列出。", ""]
    for s, c in sorted(CARDS.items(), key=lambda x: (x[1]["T"], x[1]["H"])):
        T, H, NT = c["T"], c["H"], c["NT"]
        L += [f"## {T // 1024}K / H{H}（NT={NT}）", "",
              "### 表1：实测与利用率",
              "| kernel | µs | CTAs | HBM B/s | L2 B/s | LDS B/s | TA（全部CU） | VALU槽 | LDS idx | MMAC槽 | 冲突/条 | 等待/条 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, v in sorted(c["cards"].items(), key=lambda x: -x[1]["t_us"]):
            un = " *" if v["t_us"] < 10 else ""
            L.append(f"| {k}{un} | {v['t_us']:.0f} | {v['ctas']} | {v['hbm_gbs']:.0f} GB/s "
                     f"| {v['l2_gbs'] / 1e3:.2f} TB/s | {v['lds_gbs'] / 1e3:.2f} TB/s "
                     f"| {v['u_ta_act']:.2f} | {v['u_valu_act']:.2f} | {v['u_lds_act']:.2f} "
                     f"| {v['u_mmop_act']:.2f} | {v['conf_i']:.2f} | {v['wait_i']:.2f} |")
        L += ["", "*<10µs 的 kernel 计数不可靠。*", "",
              "### 表2：现有指令流下限分解（重要：这是当前指令流全速发射所需的时间，不是算子真下限）",
              "| kernel | µs | 带宽项 | 算力项 | 发射项（TA） | 发射项（VALU） | 发射项（LDS） | 链项 | **流下限** | 绑定项 | 实测/流 | **接口下限** | 实测/接口 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
        for k, v in sorted(c["cards"].items(), key=lambda x: -x[1]["t_us"]):
            fi = v.get("f_iface_us") or 0
            L.append(f"| {k} | {v['t_us']:.0f} | {v['f_mem']:.0f} | {v['f_cmp']:.0f} "
                     f"| {v['f_ta']:.0f} | {v['f_val']:.0f} | {v['f_lds']:.0f} | {v['f_chain']:.0f} "
                     f"| **{v['floor_us']:.0f}** | {v['binding']} | {g(v['gap'])}x "
                     f"| **{fi:.0f}** | {v['t_us'] / fi:.1f}x |")
        L += ["", "### 表3：工作量缺口（实际/必需）",
              "| kernel | 实测 HBM (MB) | 接口必需 (MB) | 字节倍率 w_gap | **TA 放大** | VMEM 指令 | VMEM 指令倍数* | MMAC FLOPs | MMAC 类型 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for k, v in sorted(c["cards"].items(), key=lambda x: -x[1]["t_us"]):
            req = v["req_bytes"]
            vreq = req / 1024 if req else None          # best-pattern 1 KB per instruction
            wg = v["w_gap"]
            vi = v["vmem"] / vreq if vreq else None
            ta = f"{v['ta_amp']:.2f}x" if v.get("ta_amp") else "—"
            L.append(f"| {k} | {(v['hbm_rd'] + v['hbm_wr']) / 1e6:.0f} | "
                     f"{f'{req / 1e6:.0f}' if req else '—'} | "
                     f"{f'{wg:.2f}x' if wg else '—'} | "
                     f"**{ta}** | {v['vmem'] / 1e6:.2f}M "
                     f"| {f'{vi:.1f}x' if vi else '—'} "
                     f"| {v['flops'] / 1e9:.1f}G | {v['mmac_kind']} |")
        L += ["", "### 表4：判定（任务 §5）",
              "| kernel | 主约束 | 利用率 | 次约束 | 利用率 | 工作量比 | 是否见顶 | 缺口类型 | 消融/实测证据 |",
              "|---|---|---:|---|---:|---:|---|---|---|"]
        for k, v in sorted(c["cards"].items(), key=lambda x: -x[1]["t_us"]):
            wr = v.get("work_ratio")
            ev = EVIDENCE.get(k.split("#")[0], "")
            L.append(f"| {k} | {v.get('primary', '—')} | {v.get('u_primary', 0):.2f} "
                     f"| {v.get('secondary', '—')} | {v.get('u_secondary', 0):.2f} "
                     f"| {f'{wr:.2f}x' if wr else '—'} | {'是' if v.get('at_ceiling') else '否'} "
                     f"| {v.get('gap_types', '—')} | {ev} |")
        L += [""]
    # ---- F5: P-53 + P-54a production delta (16K only)
    p54_path = RES / "raw" / "p54_times.json"
    if p54_path.exists():
        import statistics as _st
        delta = json.load(open(p54_path))
        unchanged = {"wy_s4a#1", "wy_s4b#1", "wu#1", "dhu_v8b#1", "dhu_v8b4#1",
                     "fwd_h_v8o4#2", "fwd_prep_a#1", "fwd_prep_b#1", "dav#1",
                     "gate#1", "l2n_apply#1"}
        L += ["", "## 增量（P-53 + P-54a，生产树 g2_card_p54，指纹 90a91fdd5fdd）", "",
              "P-53：fwd_h_v8o4#1 的 GEMM1 h 片段寄存器复用；P-54a：bwd_tail 减 VMEM 读。",
              "跨卡整体位移（未改 kernel 的中位 delta，主要为功率/显存位置噪声 R41）："
              "每形态给出 baseline shift 与剔除后的净增量。", "",
              "| 形态 | 总时间 P-52→P-54 | baseline shift | fwd_h#1 净增量 | bwd_tail 净增量 | fwd_h#1 HBM GB/s | bwd_tail HBM GB/s |",
              "|---|---|---|---|---|---|---|"]
        for s in sorted(delta, key=lambda s: CARDS[s]["H"]):
            old = CARDS[s]["cards"]; new = delta[s]
            shifts = [(new[k]["t_us"] - old[k]["t_us"]) / old[k]["t_us"]
                      for k in unchanged if k in new and k in old]
            base = _st.median(shifts)
            tot_o = sum(v["t_us"] for v in old.values()) / 1e3
            tot_n = sum(v["t_us"] for v in new.values()) / 1e3
            cells = [f"{tot_o:.1f}→{tot_n:.1f} ms ({(tot_n - tot_o) / tot_o * 100:+.1f}%)",
                     f"{base * 100:+.1f}%"]
            for k in ("fwd_h_v8o4#1", "bwd_tail#1"):
                o, n = old[k]["t_us"], new[k]["t_us"]
                cells.append(f"{(n - o) / o * 100:+.1f} raw / {(n - o) / o * 100 - base * 100:+.1f} net")
            for k in ("fwd_h_v8o4#1", "bwd_tail#1"):
                t_s = new[k]["t_us"] * 1e-6
                bw = (new[k]["rd"] + new[k]["wr"]) / t_s / 1e9 if new[k].get("rd") else 0
                bw_o = old[k]["hbm_gbs"]
                cells.append(f"{bw_o:.0f}→{bw:.0f}")
            L.append(f"| {CARDS[s]['T'] // 1024}K/H{CARDS[s]['H']} | " + " | ".join(cells) + " |")
        L += ["", "*±3% 以内的未改 kernel 记为显存位置噪声；净增量 = raw delta − baseline shift。*", ""]

    L += ["## 附：launcher 接口账目（optrace）与抽查核对", ""]
    for s, c in sorted(CARDS.items(), key=lambda x: (x[1]["T"], x[1]["H"])):
        if not c["launchers"]:
            continue
        L.append(f"**{c['T'] // 1024}K/H{c['H']}**: " + "；".join(
            f"{ln} {v['in_bytes'] / 1e6:.0f}+{v['out_bytes'] / 1e6:.0f}MB"
            for ln, v in c["launchers"].items()))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    (RES / "cards.md").write_text(md())
    print("wrote", RES / "cards.md")
