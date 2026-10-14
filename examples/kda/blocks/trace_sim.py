"""Trace-driven simulation of a chunk-parallel kernel on one CU (constructive model, phase 2 step 3).
Input: one CTA's per-wave dynamic instruction sequences (SQTT, production binary; cta_trace.py), the residency
(co-resident CTAs per CU), the number of CTAs the busiest CU runs, and the kernel's memory behaviour (hbm_frac =
HBM read bytes / loaded bytes).  The same issue / dependency / LDS / TA / HBM rules and library costs as isa_sim.py;
cp CTAs share the CU (wave slot s on SIMD s % 4), finished CTAs are replaced by the next one (rounds).
    python3 trace_sim.py <cta_trace.json> --ctas N --res R [--hbm-frac F] [--active-cus 80]"""
import argparse
import heapq
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isa_sim as S  # noqa: E402

WIDTH = {"dword": 4, "dwordx2": 8, "dwordx4": 16, "ushort": 2, "short": 2, "ubyte": 1, "byte": 1, "dwordx3": 12}


def decode_prog(asm_lines):
    prog = []
    for a in asm_lines:
        parts = a.split(None, 1)
        opc, ops = parts[0], (parts[1] if len(parts) > 1 else "")
        if opc == "s_endpgm":
            prog.append((opc, ops, "end", set(), set(), None))
            continue
        cls = S.classify(opc)
        if cls is None or cls == "br":
            continue
        d, s_, ex = S.decode(opc, ops, cls)
        prog.append((opc, ops, cls, d, s_, ex))
    return prog


def simulate(progs, res, n_ctas, hbm_frac=0.0, active_cus=80, vmem_lat=None, st_lat=430.0, ta_occ=None):
    """ta_occ: optional per-wave list (one entry per VMEM instruction, None = default by width) of TA occupancy"""
    W = len(progs)
    nslot = res * W
    bw_rd = S.HBM_RD / active_cus / S.CLK
    bw_wr = S.HBM_WR / active_cus / S.CLK
    simd_free = [0.0] * 4
    lds_free = ta_free = hbm_free = 0.0
    slot_cta = [-1] * nslot
    pc = [0] * nslot
    t = [0.0] * nslot
    vready = [dict() for _ in range(nslot)]
    vfwd = [dict() for _ in range(nslot)]
    vm_q = [[] for _ in range(nslot)]
    lg_q = [[] for _ in range(nslot)]
    vq_leave = [[] for _ in range(nslot)]
    lq_leave = [[] for _ in range(nslot)]
    wave_lds = [0.0] * nslot
    bar_cnt = [0] * nslot
    vmem_k = [0] * nslot
    bar_arr = {}
    next_cta = 0
    done_waves = {}
    heap = []
    end_time = 0.0

    def start_cta(slot0, when):
        nonlocal next_cta
        if next_cta >= n_ctas:
            return
        c = next_cta
        next_cta += 1
        for k in range(W):
            s = slot0 + k
            slot_cta[s] = c
            pc[s] = 0
            t[s] = when
            vready[s].clear(); vfwd[s].clear(); vm_q[s] = []; lg_q[s] = []; vq_leave[s] = []; lq_leave[s] = []
            bar_cnt[s] = 0
            vmem_k[s] = 0
            heapq.heappush(heap, (when, s))
        done_waves[c] = 0

    for r in range(res):
        start_cta(r * W, 0.0)
    while heap:
        tw, s = heapq.heappop(heap)
        w = s % W
        prog = progs[w]
        if pc[s] >= len(prog):
            continue
        opc, ops, cls, dst, src, extra = prog[pc[s]]
        dep = max((vready[s].get(r_, 0.0) for r_ in src), default=0.0)
        if cls in ("mmac16", "mmac8f") and extra:
            dep = max(dep, max((vfwd[s].get(r_, vready[s].get(r_, 0.0)) for r_ in extra), default=0.0))
        start = max(t[s], dep)
        if cls == "end":
            vm_end = max(vm_q[s], default=start)
            fin = max(start, vm_end)
            c = slot_cta[s]
            done_waves[c] += 1
            end_time = max(end_time, fin)
            pc[s] = len(prog)
            if done_waves[c] == W:
                start_cta(s - w, fin)
            continue
        if cls == "wait":
            vm, lg = extra
            if vm is not None and len(vm_q[s]) > vm:
                start = max(start, vm_q[s][len(vm_q[s]) - vm - 1])
            if lg is not None and len(lg_q[s]) > lg:
                start = max(start, lg_q[s][len(lg_q[s]) - lg - 1])
            vm_q[s] = [x for x in vm_q[s] if x > start]
            lg_q[s] = [x for x in lg_q[s] if x > start]
            t[s] = start + S.ISSUE
        elif cls == "bar":
            key = (slot_cta[s], bar_cnt[s])
            bar_arr.setdefault(key, {})[s] = start
            if len(bar_arr[key]) < W:
                t[s] = start
                continue                        # parked until the last wave of the CTA arrives
            rel = max(bar_arr[key].values()) + S.BAR.get(W, 10.0)
            for v in bar_arr[key]:
                t[v] = rel
                bar_cnt[v] += 1
                pc[v] += 1
                if v != s:
                    heapq.heappush(heap, (t[v], v))
            del bar_arr[key]
            heapq.heappush(heap, (t[s], s))
            continue
        elif cls in S.COST and cls != "salu":
            issue, lat = S.COST[cls]
            sm = s % 4
            start = max(start, simd_free[sm])
            simd_free[sm] = start + issue
            for r_ in dst:
                vready[s][r_] = start + lat
                if cls in S.MMAC_FWD:
                    vfwd[s][r_] = start + S.MMAC_FWD[cls]
                else:
                    vfwd[s].pop(r_, None)
            t[s] = start + issue
        elif cls == "salu":
            t[s] = start + S.ISSUE
        elif cls == "nop":
            t[s] = start + S.ISSUE + (extra or 0)
        elif cls == "lds":
            occ = S.LDS_OCC.get(opc, 6.0)
            occ_w = S.LDS_WAVE.get(opc, max(occ, 8.0))
            pend = [x for x in lq_leave[s] if x > start]
            s_iss = start if len(pend) < S.Q_WAVE else sorted(pend)[len(pend) - S.Q_WAVE]
            s_port = max(s_iss, wave_lds[s])
            wave_lds[s] = s_port + occ_w
            s_cu = max(s_port, lds_free)
            lds_free = s_cu + occ
            lq_leave[s] = pend + [s_cu]
            lg_q[s].append(max(s_cu + max(occ, occ_w) + S.LDS_LAT, lg_q[s][-1] if lg_q[s] else 0.0))
            t[s] = s_iss + S.LDS_ISSUE.get(opc, S.ISSUE)
        elif cls == "smem":
            lg_q[s].append(max(start + 200.0, lg_q[s][-1] if lg_q[s] else 0.0))
            t[s] = start + S.ISSUE
        elif cls in ("vld", "vst"):
            kind = "ld" if cls == "vld" else "st"
            width = opc.split("_", 2)[-1].replace("_d16_hi", "").replace("_d16", "")
            occ = S.TA_OCC[kind].get(width, 22.0 if kind == "ld" else 29.0)
            if ta_occ is not None and vmem_k[s] < len(ta_occ[w]) and ta_occ[w][vmem_k[s]] is not None:
                occ = ta_occ[w][vmem_k[s]]
            vmem_k[s] += 1
            pend = [x for x in vq_leave[s] if x > start]
            s_iss = start if len(pend) < S.Q_WAVE else sorted(pend)[len(pend) - S.Q_WAVE]
            s0 = max(s_iss, ta_free)
            ta_free = s0 + occ
            vq_leave[s] = pend + [s0]
            nbytes = 64 * WIDTH.get(width, 4)
            hb = nbytes * (hbm_frac if cls == "vld" else 1.0) / (bw_rd if cls == "vld" else bw_wr)
            h0 = max(s0, hbm_free)
            hbm_free = h0 + hb
            lat = (vmem_lat if vmem_lat is not None else S.LAT_L2 + hbm_frac * (S.LAT_HBM - S.LAT_L2)) if cls == "vld" else st_lat
            vm_q[s].append(max(max(s0 + occ, h0 + hb) + lat, vm_q[s][-1] if vm_q[s] else 0.0))
            t[s] = s_iss + S.ISSUE
        pc[s] += 1
        heapq.heappush(heap, (t[s], s))
    return end_time


def loaded_bytes(progs):
    b = 0
    for p in progs:
        for opc, ops, cls, *_ in p:
            if cls == "vld":
                b += 64 * WIDTH.get(opc.split("_", 2)[-1].replace("_d16_hi", "").replace("_d16", ""), 4)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--ctas", type=int, required=True, help="CTAs of the kernel (all CUs)")
    ap.add_argument("--res", type=int, required=True, help="co-resident CTAs per CU")
    ap.add_argument("--hbm-frac", type=float, default=None)
    ap.add_argument("--hbm-rd", type=float, default=None, help="kernel HBM read bytes (derives hbm_frac)")
    ap.add_argument("--active-cus", type=int, default=80)
    a = ap.parse_args()
    tr = json.load(open(a.trace))
    progs = [decode_prog(w) for w in tr["waves"]]
    hf = a.hbm_frac if a.hbm_frac is not None else min(1.0, (a.hbm_rd or 0.0) / (loaded_bytes(progs) * a.ctas))
    n = math.ceil(a.ctas / 80)
    cyc = simulate(progs, a.res, n, hf, a.active_cus)
    print(f"{a.trace}: {n} CTAs on the busiest CU at residency {a.res}, hbm_frac {hf:.3f} -> {cyc:.0f} cycles = {cyc / 1500:.1f} us")


if __name__ == "__main__":
    main()
