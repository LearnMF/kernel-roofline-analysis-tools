"""ISA-level in-order issue simulator for one CTA on one gfx936 CU (building-block library P0-1).

A loop body (the 'Loop Header' block of a kernel's .s) is executed for N iterations by the CTA's W waves:
  * a wave issues its instructions in order; wave w runs on SIMD w % 4 and shares that SIMD's vector pipe;
  * VALU / MMAC / transcendental results interlock on VGPRs (ready = issue + latency); memory results are only
    ordered by s_waitcnt (vmcnt / lgkmcnt, in-order completion) as on the hardware;
  * LDS instructions occupy the CU's LDS pipe, VMEM instructions the CU's TA (per-instruction occupancy);
  * s_barrier releases when all W waves have arrived (+ barrier cost).
Every cost / latency comes from the building-block library measurements (results/kda-blocks-20261014), nothing
is fitted to the kernels being predicted.  The steady-state time per iteration is reported.
    python3 isa_sim.py kernel.s --waves 4 [--iters 16] [--vmem-lat 700] [--trace]"""
import argparse
import heapq
import re

# ---------------------------------------------------------------------------------- costs (library, 1.5 GHz)
COST = {
    "valu": (4.0, 8.9),        # (issue cycles on the SIMD, result latency): v_fma 4.85 thr @4 waves, 8.9 dependent
    "valu64": (4.0, 7.0),      # v_mov_b64 / v_lshlrev_b64: full rate (4.6 thr, 6.8-7.0 dep)
    "carry": (8.3, 9.4),       # v_add_co / v_addc_co pairs through VCC: 16.7 per pair
    "pk": (6.0, 11.3),         # v_pk_fma_f32 6.06 thr, 11.3 dep
    "trans": (16.0, 20.0),     # v_exp / v_rcp / v_log 16.6 thr, 18-22 dep
    "mmac16": (8.0, 30.0),     # bf16 16x16x16: 9.3 cyc/SIMD saturated; result -> VALU / srcA ~30 (mmac_dep_bench:
                               # 38.5 per mmac->valu->srcA round incl. two VALU); srcC accumulate forwarding: MMAC_FWD
    "mmac8f": (32.0, 60.0),    # f32 16x16x8: ~34 cyc/SIMD saturated (dependent latency not yet re-measured)
    "salu": (1.0, 2.0),
}
LDS_OCC = {   # cycles of the CU's LDS pipe per wave instruction (conflict-free, lds_bench)
    "ds_read_b32": 3.1, "ds_read_b64": 5.1, "ds_read_b128": 8.1, "ds_read2_b32": 8.1, "ds_read2_b64": 9.9, "ds_read2st64_b64": 9.9,
    "ds_read_u16": 4.7, "ds_read_u8": 4.7, "ds_read_u16_d16": 4.7, "ds_read_u16_d16_hi": 4.7,
    "ds_write_b16": 4.7, "ds_write_b16_d16_hi": 4.7, "ds_write_b8": 4.7, "ds_write_b32": 4.7, "ds_write_b64": 8.0,
    "ds_write_b128": 10.3, "ds_write2_b32": 8.0, "ds_write2_b64": 16.0, "ds_bpermute_b32": 6.4,
    "ds_swizzle_b32": 3.1,
}
MMAC_FWD = {"mmac16": 10.5, "mmac8f": 34.0}   # dependent srcC accumulate chain, straight-line (mmac_dep_bench)
LDS_LAT = 80.0          # issue -> data (dependent ds_read_b32 chain 88.6 incl. its own occupancy)
import os as _os
# dwordx4 = 30: CU acceptance with few waves / CU (issue_burst_bench, 2 waves, SQTT); 22 at 32 waves (vmem_bench)
TA_OCC = {"ld": {"dword": 12.0, "dwordx2": 21.0, "dwordx4": float(_os.environ.get("TA_X4", 30.0)), "ushort": 13.0, "short": 13.0},
          "st": {"dword": 15.0, "dwordx2": 29.0, "dwordx4": 59.0, "short": 11.0}}   # L2-resident contiguous rows
BAR = {4: 10.0, 8: 16.0, 16: 45.0}   # s_barrier itself when waves arrive together: lds_bench BAR 57 / 65 / 94 minus
                                       # the ~47-51-cycle loop overhead (rec_step_gen: empty 51 -> bar2 64 cycles)
BRANCH = 8.0            # taken backward branch (assumption: pipeline refill)
Q_WAVE = 16             # per-wave issue queue for VMEM / LDS (issue_burst_bench SQTT: ~16-20 issue at 4 cycles,
                        # then issue proceeds at the unit's processing rate; no hard in-flight limit up to 32)
LDS_WAVE = {"ds_read2_b64": 16.0, "ds_read2st64_b64": 16.0, "ds_write_b16": 8.0, "ds_write_b16_d16_hi": 8.0}
                        # single-wave LDS rate (issue_burst_bench, 1 wave / CU); other forms: max(CU occupancy, 8)
LDS_ISSUE = {"ds_write_b16": 8.0, "ds_write_b16_d16_hi": 8.0, "ds_write_b8": 8.0}
                        # issue_burst_bench SQTT: a wave issues ds_write_b16 every 8 cycles from the first one (no queueing)
ISSUE = 4.0             # every instruction occupies the wave's issue slot for 4 cycles (SQTT clks = 4)

VREG = re.compile(r"\bv\[(\d+):(\d+)\]|\bv(\d+)\b")


def vregs(op):
    out = set()
    for a, b, c in VREG.findall(op):
        if c:
            out.add(int(c))
        else:
            out.update(range(int(a), int(b) + 1))
    return out


def classify(opc):
    if opc.startswith("v_mmac"):
        return "mmac16" if "bf16" in opc or "f16" in opc else "mmac8f"
    if opc.startswith(("v_exp", "v_log", "v_rcp", "v_rsq", "v_sqrt", "v_sin", "v_cos")):
        return "trans"
    if opc.startswith("v_pk_"):
        return "pk"
    if opc.startswith(("v_mov_b64", "v_lshlrev_b64", "v_lshrrev_b64", "v_lshl_add_u64", "v_add_u64")):
        return "valu64"
    if opc.startswith(("v_add_co_u32", "v_addc_co_u32", "v_sub_co_u32", "v_subb_co_u32", "v_subrev_co_u32")):
        return "carry"
    if opc.startswith("v_"):
        return "valu"
    if opc.startswith("ds_"):
        return "lds"
    if opc.startswith(("global_load", "buffer_load", "flat_load")):
        return "vld"
    if opc.startswith(("global_store", "buffer_store", "flat_store", "global_atomic", "buffer_atomic")):
        return "vst"
    if opc.startswith(("s_load", "s_buffer_load")):
        return "smem"
    if opc == "s_waitcnt":
        return "wait"
    if opc == "s_barrier":
        return "bar"
    if opc == "s_nop":
        return "nop"
    if opc.startswith(("s_cbranch", "s_branch")):
        return "br"
    if opc.startswith("s_"):
        return "salu"
    return None


def parse_loop(path, which=0):
    lines = open(path).read().splitlines()
    heads = [i for i, l in enumerate(lines) if "Loop Header" in l]
    h = heads[which]
    label = lines[h].split(":")[0]
    body = []
    for l in lines[h + 1:]:
        s = l.split(";")[0].strip()
        if not s or s.startswith("."):
            if s.startswith(".LBB") and body:
                pass
            continue
        parts = s.split(None, 1)
        opc = parts[0]
        ops = parts[1] if len(parts) > 1 else ""
        cls = classify(opc)
        if cls is None:
            continue
        body.append((opc, ops, cls))
        if cls == "br" and label in ops:
            break
    return body


def decode(opc, ops, cls):
    """-> (dst VGPRs, src VGPRs, extra) for one instruction"""
    fields = [f.strip() for f in ops.split(",")] if ops else []
    dst, src = set(), set()
    if cls in ("valu", "valu64", "carry", "pk", "trans", "mmac16", "mmac8f", "vld"):
        if fields and VREG.match(fields[0]):
            dst = vregs(fields[0])
            src = set().union(*[vregs(f) for f in fields[1:]]) if len(fields) > 1 else set()
        else:
            src = set().union(*[vregs(f) for f in fields]) if fields else set()
    elif cls == "lds":
        if opc.startswith(("ds_read", "ds_bpermute", "ds_swizzle")) and fields:
            dst = vregs(fields[0])
            src = set().union(*[vregs(f) for f in fields[1:]]) if len(fields) > 1 else set()
        else:
            src = set().union(*[vregs(f) for f in fields]) if fields else set()
    elif cls == "vst":
        src = set().union(*[vregs(f) for f in fields]) if fields else set()
    extra = None
    if cls in ("mmac16", "mmac8f") and len(fields) >= 4:
        extra = vregs(fields[3])                       # srcC (accumulator input)
        src = vregs(fields[1]) | vregs(fields[2])
    if cls == "wait":
        vm = re.search(r"vmcnt\((\d+)\)", ops)
        lg = re.search(r"lgkmcnt\((\d+)\)", ops)
        extra = (int(vm.group(1)) if vm else None, int(lg.group(1)) if lg else None)
    if cls == "nop":
        extra = int(ops or 0)
    return dst, src, extra


HBM_RD, HBM_WR = 1339.2e9, 1100.5e9   # achievable device bandwidth (machines/gfx936-bw1000.json)
CLK = 1.5e9


LAT_L2, LAT_HBM = 700.0, 950.0   # load latency, fresh lines, pointer chase (vmem_bench lat_cyc: L2 664-735, HBM 861-977);
                                 # vmem_compl_bench (386 / 588) re-touched lines (L1 / L2 hits), lat_load: +2% at 64%, +14% at 82%, +45% at 88% HBM load


def simulate(body, W=4, iters=16, vmem_lat=None, st_lat=430.0, trace=False, hbm_frac=0.0, active_cus=80,
             ctas_per_cu=1):
    """hbm_frac: fraction of the loaded bytes that come from HBM (the rest hit L2) -- a memory-behaviour input
    (e.g. unique bytes / requested bytes from the PMC or the dataflow), active_cus: CUs sharing the bandwidth."""
    bw_rd = HBM_RD / active_cus / CLK / ctas_per_cu      # bytes / cycle available to this CTA
    bw_wr = HBM_WR / active_cus / CLK / ctas_per_cu
    hbm_free = 0.0
    prog = [(o, a, c) + decode(o, a, c) for o, a, c in body]
    n = len(prog)
    simd_free = [0.0] * 4
    lds_free = ta_free = 0.0
    vready = [dict() for _ in range(W)]
    vfwd = [dict() for _ in range(W)]          # MMAC results reaching a following MMAC's srcC
    t = [0.0] * W
    pc = [0] * W
    it = [0] * W
    vm_q = [[] for _ in range(W)]     # completion times of outstanding VMEM ops (in order)
    vq_leave = [[] for _ in range(W)]  # times the wave's queued VMEM instructions enter the TA
    lq_leave = [[] for _ in range(W)]
    wave_lds = [0.0] * W
    lg_q = [[] for _ in range(W)]
    bar_arr, bar_wait = {}, set()
    iter_end = [[] for _ in range(W)]
    stall = {}
    pos_t = {}
    heap = [(0.0, 0, w) for w in range(W)]
    heapq.heapify(heap)
    while heap:
        tw, _, w = heapq.heappop(heap)
        if it[w] >= iters:
            continue
        opc, ops, cls, dst, src, extra = prog[pc[w]]
        dep = max((vready[w].get(r, 0.0) for r in src), default=0.0)
        if cls in ("mmac16", "mmac8f") and extra:
            dep = max(dep, max((vfwd[w].get(r, vready[w].get(r, 0.0)) for r in extra), default=0.0))
        start = max(t[w], dep)
        if w == 0 and it[w] >= iters // 2 and dep > t[w]:
            stall["dep:" + cls] = stall.get("dep:" + cls, 0.0) + dep - t[w]
        if cls == "wait":
            vm, lg = extra
            if vm is not None and len(vm_q[w]) > vm:
                start = max(start, vm_q[w][len(vm_q[w]) - vm - 1])
            if lg is not None and len(lg_q[w]) > lg:
                start = max(start, lg_q[w][len(lg_q[w]) - lg - 1])
            if w == 0 and it[w] >= iters // 2 and start > t[w]:
                stall["wait:" + ops.split()[0][:6]] = stall.get("wait:" + ops.split()[0][:6], 0.0) + start - t[w]
            vm_q[w] = [c for c in vm_q[w] if c > start]
            lg_q[w] = [c for c in lg_q[w] if c > start]
            t[w] = start + ISSUE
        elif cls == "bar":
            key = (it[w], pc[w])
            bar_arr.setdefault(key, {})[w] = start
            if len(bar_arr[key]) < W:
                bar_wait.add(w)
                t[w] = start
                pc[w] = pc[w]          # stay; released by the last arrival
                continue
            rel = max(bar_arr[key].values()) + BAR.get(W, 60.0)
            if it[0] >= iters // 2 and 0 in bar_arr[key]:
                stall["barrier"] = stall.get("barrier", 0.0) + rel - bar_arr[key][0]
            for v in list(bar_arr[key]):
                t[v] = rel
                if v != w:
                    bar_wait.discard(v)
                    pc[v] += 1
                    if pc[v] == n:
                        pc[v] = 0; it[v] += 1; iter_end[v].append(t[v])
                    heapq.heappush(heap, (t[v], (v - it[v]) % W, v))
            del bar_arr[key]
        elif cls in COST and cls != "salu":
            issue, lat = COST[cls]
            s = w % 4
            if w == 0 and it[w] >= iters // 2 and simd_free[s] > start:
                stall["simd_busy"] = stall.get("simd_busy", 0.0) + simd_free[s] - start
            start = max(start, simd_free[s])
            simd_free[s] = start + issue
            for r in dst:
                vready[w][r] = start + lat
                if cls in MMAC_FWD:
                    vfwd[w][r] = start + MMAC_FWD[cls]
                else:
                    vfwd[w].pop(r, None)
            t[w] = start + issue
        elif cls == "salu":
            t[w] = start + ISSUE
        elif cls == "nop":
            t[w] = start + ISSUE + extra
        elif cls == "br":
            t[w] = start + BRANCH
        elif cls == "lds":
            occ = LDS_OCC.get(opc, 6.0)
            occ_w = LDS_WAVE.get(opc, max(occ, 8.0))
            pend = [x for x in lq_leave[w] if x > start]
            s_iss = start if len(pend) < Q_WAVE else sorted(pend)[len(pend) - Q_WAVE]
            if w == 0 and it[w] >= iters // 2 and s_iss > start:
                stall["issue:lds"] = stall.get("issue:lds", 0.0) + s_iss - start
            s_port = max(s_iss, wave_lds[w])
            wave_lds[w] = s_port + occ_w
            s_cu = max(s_port, lds_free)
            lds_free = s_cu + occ
            lq_leave[w] = pend + [s_cu]
            done = s_cu + max(occ, occ_w) + LDS_LAT
            lg_q[w].append(max(done, lg_q[w][-1] if lg_q[w] else 0.0))
            t[w] = s_iss + LDS_ISSUE.get(opc, ISSUE)
        elif cls == "smem":
            lg_q[w].append(max(start + 200.0, lg_q[w][-1] if lg_q[w] else 0.0))
            t[w] = start + ISSUE
        elif cls in ("vld", "vst"):
            kind = "ld" if cls == "vld" else "st"
            width = opc.split("_", 2)[-1].replace("_d16_hi", "").replace("_d16", "")
            occ = TA_OCC[kind].get(width, 22.0 if kind == "ld" else 59.0)
            pend = [x for x in vq_leave[w] if x > start]
            s_iss = start if len(pend) < Q_WAVE else sorted(pend)[len(pend) - Q_WAVE]
            if w == 0 and it[w] >= iters // 2 and s_iss > start:
                stall["issue:vmem"] = stall.get("issue:vmem", 0.0) + s_iss - start
            s0 = max(s_iss, ta_free)
            ta_free = s0 + occ
            vq_leave[w] = pend + [s0]
            nbytes = 64 * {"dword": 4, "dwordx2": 8, "dwordx4": 16, "ushort": 2, "short": 2}.get(width, 8)
            hb = nbytes * (hbm_frac if cls == "vld" else 1.0) / (bw_rd if cls == "vld" else bw_wr)
            h0 = max(s0, hbm_free)
            hbm_free = h0 + hb
            lat = (vmem_lat if vmem_lat is not None else LAT_L2 + hbm_frac * (LAT_HBM - LAT_L2)) if cls == "vld" else st_lat
            done = max(s0 + occ, h0 + hb) + lat
            vm_q[w].append(max(done, vm_q[w][-1] if vm_q[w] else 0.0))
            t[w] = s_iss + ISSUE
        if w == 0 and it[w] >= iters // 2:
            pos_t.setdefault(pc[w], []).append(t[w])
        if trace and w == 0 and it[w] == iters - 2:
            print(f"{t[w]:9.0f} w{w} {opc} {ops[:60]}")
        pc[w] += 1
        if pc[w] == n:
            pc[w] = 0
            it[w] += 1
            iter_end[w].append(t[w])
        heapq.heappush(heap, (t[w], (w - it[w]) % W, w))
    k0 = iters // 2
    per = [(e[-1] - e[k0 - 1]) / (iters - k0) for e in iter_end if len(e) >= iters]
    simulate.stall = {k: v / (iters - k0) for k, v in sorted(stall.items(), key=lambda kv: -kv[1])}
    simulate.pos_t = pos_t
    return max(per), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("isa")
    ap.add_argument("--waves", type=int, default=4)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--vmem-lat", type=float, default=None, help="load latency override (default: L2 386 / HBM 588 mix)")
    ap.add_argument("--st-lat", type=float, default=430.0, help="store issue -> vmcnt (vmem_compl_bench 427-430)")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--hbm-frac", type=float, default=0.0)
    ap.add_argument("--active-cus", type=int, default=80)
    ap.add_argument("--ctas-per-cu", type=int, default=1)
    a = ap.parse_args()
    body = parse_loop(a.isa)
    cyc, per = simulate(body, a.waves, a.iters, a.vmem_lat, a.st_lat, a.trace, a.hbm_frac, a.active_cus, a.ctas_per_cu)
    mix = {}
    for _, _, c in body:
        mix[c] = mix.get(c, 0) + 1
    print(f"{a.isa}: {len(body)} instructions/iteration {mix}")
    print(f"  steady state: {cyc:.0f} cycles = {cyc / 1500:.3f} us per iteration (waves: {[round(p) for p in per]})")
    print("  wave-0 stalls per iteration (cycles):", {k: round(v) for k, v in simulate.stall.items()})


if __name__ == "__main__":
    main()
