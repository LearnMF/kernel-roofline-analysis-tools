"""L0 machine calibration: build + run microbench.hip on the target, emit machine.json.

Runs ON the HCU/GPU host (inside the container that has hipcc).  Standard library only.

    python -m kra calibrate --out machines/gfx936-bw1000.json [--arch gfx936] [--bench hbm ...]

Every peak is the best measured configuration ("attainable peak"), stored together with
its median, the winning configuration, and -- where the device reports enough to derive
one -- the datasheet-style theoretical value.  The generated ISA is checked so that a
peak is never attributed to an instruction the compiler did not actually emit.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

SCHEMA = "kra.machine/1"
HERE = Path(__file__).resolve().parent
SRC = HERE / "microbench.hip"

# kernel-symbol substring -> instruction regexes that MUST appear in its body.
ISA_EXPECT = {
    "k_mmac_bf16": [r"v_mmac_f32_16x16x16_bf16"],
    "k_mmac_f16": [r"v_mmac_f32_16x16x16_f16"],
    "k_mmac_f32x8": [r"v_mmac_(f32_)?16x16x8_f32"],  # gfx936 spells it v_mmac_16x16x8_f32
    "k_mmac_tf32": [r"v_mmac_f32_16x16x8_tf32"],
    "k_mmac_i8": [r"v_mmac_i32_16x16x32_i8"],
    "k_fma_f32": [r"v_(pk_)?fma(c)?_f32"],
    "k_exp2": [r"v_exp_f32"],
    "k_readILi1": [r"global_load_dwordx4"],
    "k_read_nt": [r"global_load_dwordx4"],
    "k_copyILi1": [r"global_load_dwordx4", r"global_store_dwordx4"],
    "k_writeILi1": [r"global_store_dwordx4"],
    "k_lds_read": [r"ds_read_b128|ds_read2_b64"],
    "k_chase": [r"global_load_dword\b"],
    "k_lds_chase": [r"ds_read_b32"],
}
ISA_RECORD = r"(v_mmac_[a-z0-9_]+|v_pk_fma_f32|v_fma_f32|v_fmac_f32|v_exp_f32|global_load_dword[x0-9]*|global_store_dword[x0-9]*|s_load_dword[x0-9]*|ds_read[0-9a-z_]*|s_barrier)"


def _sh(cmd: str, check: bool = True) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n{r.stderr[-4000:]}")
    return r.stdout + r.stderr


def detect_arch() -> str:
    # rocminfo lists the agents that are physically present.  (DTK's
    # rocm_agent_enumerator prints every arch the toolchain supports, so it is NOT used.)
    if shutil.which("rocminfo"):
        m = re.findall(r"^\s*Name:\s+(gfx[0-9a-f]+)\s*$", _sh("rocminfo", check=False), re.M)
        if m:
            return m[0]
    raise RuntimeError("cannot detect GPU arch; pass --arch")


def build(arch: str, build_dir: Path, hipcc: str = "hipcc") -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / "microbench"
    _sh(f"cd {build_dir} && {hipcc} -O3 -std=c++17 --offload-arch={arch} -save-temps=obj "
        f"-o {exe} {SRC}")
    return exe


def isa_check(build_dir: Path, arch: str) -> dict:
    asm = sorted(build_dir.glob(f"*{arch}*.s"))
    if not asm:
        return {"ok": False, "error": "no device assembly produced"}
    text = asm[0].read_text(errors="replace")
    bodies, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^(_Z\S+|k_\S+):\s*(;.*)?$", line)
        if m:
            cur = m.group(1)
            bodies[cur] = []
        elif cur is not None:
            if "s_endpgm" in line:
                cur = None
            else:
                bodies[cur].append(line)
    out, ok = {}, True
    for key, pats in ISA_EXPECT.items():
        syms = [s for s in bodies if key in s]
        if not syms:
            out[key] = {"ok": False, "error": "symbol not found"}
            ok = False
            continue
        body = "\n".join(bodies[syms[0]])
        missing = [p for p in pats if not re.search(p, body)]
        counts = {}
        for ins in re.findall(ISA_RECORD, body):
            counts[ins] = counts.get(ins, 0) + 1
        out[key] = {"symbol": syms[0], "ok": not missing, "missing": missing, "counts": counts}
        ok &= not missing
    return {"ok": ok, "asm": str(asm[0]), "kernels": out}


def run_bench(exe: Path, benches: list[str], raw_path: Path) -> list[dict]:
    r = subprocess.run([str(exe), *benches], capture_output=True, text=True)
    raw_path.write_text(r.stdout + ("\n# stderr\n" + r.stderr if r.stderr else ""))
    if r.returncode != 0:
        raise RuntimeError(f"microbench failed ({r.returncode}):\n{r.stderr[-4000:]}")
    lines, warnings = [], [l for l in r.stderr.splitlines() + r.stdout.splitlines()
                           if l and not l.startswith("{")]
    for l in r.stdout.splitlines():
        if l.startswith("{"):
            lines.append(json.loads(l))
    if warnings:
        lines.append({"bench": "_warnings", "lines": warnings[:50]})
    return lines


def _best(rows: list[dict], bench: str, key: str, where=None) -> dict | None:
    cand = [r for r in rows if r.get("bench") == bench and key in r and (where is None or where(r))]
    if not cand:
        return None
    b = max(cand, key=lambda r: r[key])
    med_key = key.replace("_best", "_med")
    return {"value": b[key], "median_of_best_config": b.get(med_key), "config": b.get("config"),
            "n_configs": len(cand)}


def _lat(rows, bench, where=None):
    cand = [r for r in rows if r.get("bench") == bench and (where is None or where(r))]
    if not cand:
        return None
    r = cand[0]
    return {"cycles": r["cycles_per_step"], "ns": r["ns_per_step"], "config": r.get("config")}


def summarize(rows: list[dict]) -> dict:
    dev = next((r for r in rows if r.get("bench") == "device"), {})
    dev = {k: v for k, v in dev.items() if k != "bench"}
    peaks = {}

    def put(name, bench, key, scale=1.0, unit="", where=None):
        b = _best(rows, bench, key, where)
        if b:
            b["value"] = round(b["value"] * scale, 3)
            if b.get("median_of_best_config") is not None:
                b["median_of_best_config"] = round(b["median_of_best_config"] * scale, 3)
            b["unit"] = unit
            peaks[name] = b

    put("hbm_read", "hbm_read", "GBps_best", unit="GB/s")
    put("hbm_read_nt", "hbm_read_nt", "GBps_best", unit="GB/s")
    put("hbm_write", "hbm_write", "GBps_best", unit="GB/s")
    put("hbm_copy", "hbm_copy", "GBps_best", unit="GB/s (read+write bytes)")
    put("l2_read", "l2_read", "GBps_best", unit="GB/s",
        where=lambda r: r["config"].get("resident"))
    put("lds_read", "lds_read", "GBps_best", unit="GB/s")
    for name in ("mmac_bf16", "mmac_f16", "mmac_tf32", "mmac_f32x8", "mmac_i8", "valu_fma_f32"):
        put(name, name, "TOPS_best", unit="TOPS" if name == "mmac_i8" else "TFLOPS")
    put("sfu_exp2", "sfu_exp2", "TOPS_best", scale=1000.0, unit="Gop/s")

    # Theoretical HBM bandwidth from device properties (assumes 2 transfers per memory
    # clock, i.e. DDR signalling -- recorded as an assumption, not a measurement).
    mck, bus = dev.get("memory_clock_khz"), dev.get("memory_bus_width")
    hbm_theory = mck * 1e3 * 2 * bus / 8 / 1e9 if mck and bus else None
    if hbm_theory:
        for k in ("hbm_read", "hbm_read_nt", "hbm_write", "hbm_copy"):
            if k in peaks:
                peaks[k]["theoretical"] = round(hbm_theory, 1)
                peaks[k]["pct_of_theoretical"] = round(100 * peaks[k]["value"] / hbm_theory, 1)

    # Per-CU per-shader-clock throughput (lets readers compare with the instruction
    # issue rate in the ISA manual without trusting any marketing peak).
    cus, clk = dev.get("cu_count"), dev.get("clock_khz")
    if cus and clk:
        for k, v in peaks.items():
            if "value" not in v:
                continue
            if v["unit"] in ("TFLOPS", "TOPS"):
                v["ops_per_cu_per_clock"] = round(v["value"] * 1e12 / (cus * clk * 1e3), 1)
            elif v["unit"].startswith("GB/s") and k.startswith(("lds", "l2")):
                v["bytes_per_cu_per_clock"] = round(v["value"] * 1e9 / (cus * clk * 1e3), 1)

    hbm_bw = max((peaks[k]["value"] for k in ("hbm_read", "hbm_read_nt") if k in peaks),
                 default=None)
    ridge = {}
    if hbm_bw:
        for k in ("mmac_bf16", "mmac_f16", "mmac_tf32", "mmac_i8", "valu_fma_f32"):
            if k in peaks:
                ridge[f"{k}_vs_hbm_read"] = round(peaks[k]["value"] * 1e12 / (hbm_bw * 1e9), 1)

    latency = {
        "gmem_small_ws": _lat(rows, "gmem_latency", lambda r: r["config"]["working_set"] <= 64 << 10),
        "gmem_l2_ws": _lat(rows, "gmem_latency",
                           lambda r: 64 << 10 < r["config"]["working_set"] <= 16 << 20),
        "gmem_hbm_ws": _lat(rows, "gmem_latency", lambda r: r["config"]["working_set"] > 16 << 20),
        "lds": _lat(rows, "lds_latency"),
        "mmac_bf16_dependent": _lat(rows, "mmac_bf16_latency"),
        "barrier": {
            f'{r["config"]["mode"]}_{r["config"]["block"]}': {"cycles": r["cycles_per_step"],
                                                              "ns": r["ns_per_step"]}
            for r in rows if r.get("bench") == "barrier"},
    }
    clocks = [r["shader_clock_mhz"] for r in rows if r.get("shader_clock_mhz")]
    launch = {}
    for b, key in (("launch_stream", "us_per_kernel_best"), ("launch_graph", "us_per_kernel_best"),
                   ("launch_single", "us_best")):
        r = next((x for x in rows if x.get("bench") == b), None)
        if r:
            launch[b] = {"best_us": r[key], "median_us": r[key.replace("best", "med")]}
    pcie = {r["bench"]: r["GBps_best"] for r in rows if r.get("bench", "").startswith("pcie")}
    # Attainable HBM read bandwidth vs number of resident CTAs (one per CU).
    curve = {}
    for r in rows:
        if r.get("bench") == "hbm_read_ctas":
            n = r["config"]["ctas"]
            curve[n] = max(curve.get(n, 0.0), r["GBps_best"])
    if curve:
        peaks["hbm_read_vs_ctas"] = {"unit": "GB/s", "curve": {str(k): curve[k] for k in sorted(curve)},
                                     "note": "best of block 256/512, 4 loads in flight per thread"}
    # Per-CU vector-memory issue cost for cache-resident loads, by width (best config).
    vm = {}
    for r in rows:
        if r.get("bench") == "vmem_issue":
            c = r["config"]
            key = f"{c['width']}_{'l1' if c['working_set'] <= 65536 else 'l2'}"
            if key not in vm or r["clk_per_wave_inst"] < vm[key]["clk_per_wave_inst"]:
                vm[key] = {"clk_per_wave_inst": r["clk_per_wave_inst"],
                           "bytes_per_cu_per_clk": r["bytes_per_cu_per_clk"], "block": c["block"]}
    if vm:
        peaks["vmem_issue"] = {"unit": "per CU", "by_width": vm,
                               "note": "one CTA per CU, 8 independent loads in flight per thread"}
    vs = {r["config"]["pattern"]: {"clk_per_wave_inst": r["clk_per_wave_inst"],
                                   "bytes_spanned_per_inst": r["config"]["bytes_spanned_per_inst"]}
          for r in rows if r.get("bench") == "vmem_store"}
    if vs:
        peaks["vmem_store"] = {"unit": "per CU", "by_pattern": vs,
                               "note": "one CTA (8 waves) per CU, L2-resident 4 KB window per wave"}
    return {
        "device": dev,
        "peaks": peaks,
        "ridge_points_flop_per_byte": ridge,
        "latency": latency,
        "launch": launch,
        "pcie_GBps": pcie,
        "observed_shader_clock_mhz": {"min": min(clocks), "max": max(clocks)} if clocks else None,
        "warnings": next((r["lines"] for r in rows if r.get("bench") == "_warnings"), []),
    }


def provenance(arch: str, hipcc: str) -> dict:
    def tryrun(c):
        try:
            return _sh(c, check=False).strip()[:4000]
        except Exception as e:  # pragma: no cover
            return f"<error {e}>"
    tool_sha = tryrun(f"git -C {HERE} rev-parse HEAD 2>/dev/null") or None
    return {
        "host": socket.gethostname(),
        "arch": arch,
        "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "timestamp_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "hipcc_version": tryrun(f"{hipcc} --version"),
        "smi": tryrun("hy-smi 2>/dev/null || rocm-smi 2>/dev/null"),
        "tool_git_sha": tool_sha if tool_sha and len(tool_sha) == 40 else None,
        "source": str(SRC),
    }


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="kra calibrate", description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="machine.json path")
    p.add_argument("--arch", default=None)
    p.add_argument("--hipcc", default="hipcc")
    p.add_argument("--build-dir", default="/tmp/kra_calibrate_build")
    p.add_argument("--id", default=None, help="machine id, e.g. gfx936-bw1000")
    p.add_argument("--bench", nargs="*", default=[],
                   help="subset: hbm hbm_ctas l2 lds compute latency launch pcie (default all)")
    p.add_argument("--merge", default=None,
                   help="existing machine.json: keep its rows and add/replace the benches run now")
    a = p.parse_args(argv)

    arch = a.arch or detect_arch()
    bdir = Path(a.build_dir)
    exe = build(arch, bdir, a.hipcc)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out.with_suffix(".raw.jsonl")
    old_rows = []
    if a.merge:
        old_raw = Path(a.merge).with_suffix(".raw.jsonl")
        old_rows = [json.loads(l) for l in old_raw.read_text().splitlines() if l.startswith("{")]
    rows = run_bench(exe, a.bench, raw_path)
    if old_rows:
        fresh = {r["bench"] for r in rows if r.get("bench") != "device"}
        rows = [r for r in old_rows if r.get("bench") not in fresh] + \
               [r for r in rows if r.get("bench") != "device"]
        raw_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = summarize(rows)
    doc = {
        "schema": SCHEMA,
        "id": a.id or f"{arch}-{summary['device'].get('name', 'unknown').replace(' ', '_')}",
        "provenance": provenance(arch, a.hipcc),
        **summary,
        "isa_check": isa_check(bdir, arch),
        "raw": raw_path.name,
    }
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out} (raw: {raw_path})")
    if not doc["isa_check"]["ok"]:
        print("WARNING: ISA check failed -- see isa_check in the output", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
