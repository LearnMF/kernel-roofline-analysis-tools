"""通用交替计时：每个样本都是一个新启动的进程，按 A B B A 的顺序重复多轮，输出中位数、比值与离散度。

被测命令需在标准输出打印一行 JSON（取最后一行以 '{' 开头的行），计时字段由 --key 指定（可多个）。
可选 --idle-cmd：每个进程启动前反复执行，直到它输出的数字（例如 HCU 利用率）为 0，用于共享机器。

    python -m kra.tools.abba --key fwd_ms --key bwd_ms --rounds 2 \
        --a "env PYTHONPATH=/path/base python3 bench.py 8192 12" \
        --b "env PYTHONPATH=/path/cand python3 bench.py 8192 12" \
        [--idle-cmd "hy-smi --showuse | awk '/^0 /{print $7}' | tr -d %"] [--out ab.jsonl]

输出：每个键 A、B 的中位数，比值 A/B（>1 表示 B 更快），以及各自的相对离散度（(max-min)/median）。
"""
import argparse
import json
import shlex
import statistics as st
import subprocess
import sys
import time


def _run(cmd):
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"command failed ({out.returncode}): {cmd}\n{out.stderr[-2000:]}")
    lines = [l for l in out.stdout.splitlines() if l.lstrip().startswith("{")]
    if not lines:
        raise RuntimeError(f"no JSON line in output of: {cmd}")
    return json.loads(lines[-1])


def _wait_idle(cmd, timeout, poll):
    t0 = time.time()
    while True:
        try:
            v = float(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip().split()[0])
        except (ValueError, IndexError):
            v = 0.0
        if v == 0:
            return time.time() - t0
        if time.time() - t0 > timeout:
            raise TimeoutError("device never became idle")
        time.sleep(poll)


def abba(cmd_a, cmd_b, keys, rounds=2, idle_cmd=None, idle_timeout=3600, poll=2.0, log=None):
    rows = []
    for r in range(rounds):
        for tag in ("A", "B", "B", "A"):
            waited = _wait_idle(idle_cmd, idle_timeout, poll) if idle_cmd else 0.0
            res = _run(cmd_a if tag == "A" else cmd_b)
            row = {"round": r, "tag": tag, "wait_s": round(waited, 1), **{k: res[k] for k in keys}}
            rows.append(row)
            if log:
                log.write(json.dumps(row) + "\n"); log.flush()
    summary = {}
    for k in keys:
        xa = [x[k] for x in rows if x["tag"] == "A"]; xb = [x[k] for x in rows if x["tag"] == "B"]
        ma, mb = st.median(xa), st.median(xb)
        summary[k] = {"A_median": ma, "B_median": mb, "ratio_A_over_B": ma / mb,
                      "A_spread": (max(xa) - min(xa)) / ma, "B_spread": (max(xb) - min(xb)) / mb, "n_per_side": len(xa)}
    return rows, summary


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m kra.tools.abba", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--key", action="append", required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--idle-cmd"); p.add_argument("--idle-timeout", type=float, default=3600)
    p.add_argument("--out")
    a = p.parse_args(argv)
    log = open(a.out, "w") if a.out else None
    _, summary = abba(a.a, a.b, a.key, a.rounds, a.idle_cmd, a.idle_timeout, log=log)
    for k, s in summary.items():
        print(json.dumps({"key": k, **{n: round(v, 4) if isinstance(v, float) else v for n, v in s.items()}}))
        noisy = max(s["A_spread"], s["B_spread"])
        if abs(s["ratio_A_over_B"] - 1) < noisy / 2:
            print(f"# {k}: 差异 {abs(s['ratio_A_over_B'] - 1):.3%} 小于离散度的一半（{noisy:.3%}），视为不显著")
    return 0


if __name__ == "__main__":
    sys.exit(main())
