"""通用正确性门禁：比较两份输出转储（.npz），支持逐位一致与相对 FP64 参考的误差比两种模式。

被测程序只需把所有输出和梯度存成 .npz（每个张量一个键，按需先转成 numpy；bf16 请先 .float() 再存，
逐位比较时用 view 成整数或直接存原始字节均可）。可用本模块的 save() 辅助函数：

    from kra.tools.gate import save
    save("cand.npz", {"o": o, "dq": dq, ...})      # 接受 numpy 数组或 torch 张量

命令：
    python -m kra.tools.gate bitwise base.npz cand.npz
    python -m kra.tools.gate ref64 ref64.npz base.npz cand.npz [--max-ratio 1.0] [--ref-impl other.npz --max-ratio-ref 1.0]
退出码 0 = 通过，1 = 不通过。
"""
import argparse
import json
import sys

import numpy as np


def _np(x):
    if hasattr(x, "detach"):                      # torch.Tensor
        x = x.detach().cpu()
        if str(x.dtype) == "torch.bfloat16":
            x = x.float()
        x = x.numpy()
    return np.asarray(x)


def save(path, tensors):
    np.savez(path, **{k: _np(v) for k, v in tensors.items()})


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def bitwise(a, b):
    """逐位比较两份转储；返回 (是否通过, 每个张量的结果)。键集合必须相同。"""
    rows, ok = [], set(a) == set(b)
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            rows.append({"tensor": k, "status": "missing"}); continue
        x, y = a[k], b[k]
        same = x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()
        n = int(np.count_nonzero(x.view(np.uint8) != y.view(np.uint8))) if x.shape == y.shape and x.dtype == y.dtype else -1
        rows.append({"tensor": k, "status": "identical" if same else "differ", "differing_bytes": 0 if same else n})
        ok &= same
    return ok, rows


def rel_rms(x, ref):
    x, ref = x.astype(np.float64), ref.astype(np.float64)
    return float(np.sqrt(np.mean((x - ref) ** 2)) / max(np.sqrt(np.mean(ref ** 2)), 1e-300))


def ref64(ref, base, cand, max_ratio=1.0, other=None, max_ratio_ref=None):
    """以 FP64 参考为基准：每个张量的相对 RMS 误差，候选不得超过基准的 max_ratio 倍
    （以及可选的参考实现 other 的 max_ratio_ref 倍）。"""
    rows, ok = [], True
    for k in sorted(ref):
        e_c, e_b = rel_rms(cand[k], ref[k]), rel_rms(base[k], ref[k])
        r = e_c / max(e_b, 1e-300)
        row = {"tensor": k, "err_cand": e_c, "err_base": e_b, "ratio_vs_base": r}
        good = r <= max_ratio
        if other is not None:
            e_o = rel_rms(other[k], ref[k]); row["err_ref_impl"] = e_o; row["ratio_vs_ref_impl"] = e_c / max(e_o, 1e-300)
            good &= row["ratio_vs_ref_impl"] <= max_ratio_ref
        row["pass"] = bool(good); ok &= good
        rows.append(row)
    return ok, rows


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m kra.tools.gate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("bitwise"); b.add_argument("base"); b.add_argument("cand")
    r = sub.add_parser("ref64"); r.add_argument("ref"); r.add_argument("base"); r.add_argument("cand")
    r.add_argument("--max-ratio", type=float, default=1.0)
    r.add_argument("--ref-impl"); r.add_argument("--max-ratio-ref", type=float, default=1.0)
    a = p.parse_args(argv)
    if a.mode == "bitwise":
        ok, rows = bitwise(load(a.base), load(a.cand))
    else:
        ok, rows = ref64(load(a.ref), load(a.base), load(a.cand), a.max_ratio,
                         load(a.ref_impl) if a.ref_impl else None, a.max_ratio_ref)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print(f"# {a.mode}: {'PASS' if ok else 'FAIL'} ({sum(1 for x in rows if x.get('status') == 'identical' or x.get('pass'))}/{len(rows)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
