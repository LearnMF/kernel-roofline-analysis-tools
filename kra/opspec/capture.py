"""Capture the interface of every call into an operator's launcher functions.

For each call it records the tensors going in and coming out (shape, dtype, bytes,
counting every storage once).  "Interface bytes" = bytes of all input tensors (each
must be read at least once) + bytes of all returned tensors (each written at least
once): the minimum HBM traffic of that launcher if it kept nothing on chip that it
did not receive.  This is the memory lower bound used by L4 (`T_mem_min`).

Usage inside the application (no profiler needed):

    from kra.opspec.capture import LauncherCapture
    import hip_kda.ops.g2_ops as g2_ops
    with LauncherCapture(g2_ops, ["fwd_prep", "fwd_h_o", ...]) as cap:
        run_one_iteration()
    cap.dump("optrace.json")

Caveats (recorded in the output): tensors passed but not read are counted (upper side
of "minimum"); outputs written in place into an input tensor are counted once, as input.
"""
from __future__ import annotations

import json
from pathlib import Path


def _tensors(obj, out):
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _tensors(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            _tensors(x, out)


def _describe(ts, seen):
    items, total = [], 0
    for t in ts:
        try:
            key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
            nbytes = t.untyped_storage().nbytes()
        except Exception:  # pragma: no cover - older torch
            key = (t.data_ptr(), t.numel() * t.element_size())
            nbytes = t.numel() * t.element_size()
        dup = key in seen
        seen.add(key)
        items.append({"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                      "bytes": nbytes, "dup": dup, "ptr": key[0]})
        if not dup:
            total += nbytes
    return items, total


class LauncherCapture:
    def __init__(self, module, names=None):
        self.module = module
        self.names = names or [n for n in dir(module)
                               if not n.startswith("_") and callable(getattr(module, n))]
        self.records: list[dict] = []
        self._orig = {}

    def _wrap(self, name, fn):
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            ins, outs = [], []
            _tensors(list(args) + list(kwargs.values()), ins)
            _tensors(out, outs)
            seen: set = set()
            in_desc, in_b = _describe(ins, seen)
            out_desc, out_b = _describe(outs, seen)
            self.records.append({"seq": len(self.records), "launcher": name,
                                 "in_bytes": in_b, "out_bytes": out_b,
                                 "inputs": in_desc, "outputs": out_desc})
            return out
        wrapper.__wrapped__ = fn
        return wrapper

    def __enter__(self):
        for n in self.names:
            fn = getattr(self.module, n)
            self._orig[n] = fn
            setattr(self.module, n, self._wrap(n, fn))
        return self

    def __exit__(self, *exc):
        for n, fn in self._orig.items():
            setattr(self.module, n, fn)
        return False

    def dump(self, path, meta=None):
        doc = {"schema": "kra.optrace/1", "meta": meta or {}, "calls": self.records,
               "caveats": ["inputs counted fully even if only partly read",
                           "in-place outputs counted once (as input)"]}
        Path(path).write_text(json.dumps(doc, indent=2) + "\n")
        return doc
