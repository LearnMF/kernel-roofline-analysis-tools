#!/usr/bin/env python3
"""P2 register-pressure probe: inject N pinned live VGPRs into a named kernel.
Applied to a copy tree (e.g. /opt/kda_env/g2_calib_p2).

    python3 patch_reg_inject.py <tree> <header> <kernel_name> <N>

Keeps N live float registers across the kernel body (zero work; asm "+v" pins).
Entry: right after the kernel's '{';  exit: before its matching '}'.
"""
import sys
from pathlib import Path


def find_span(src: str, name: str):
    """(entry_insert, exit_insert) = (just after kernel '{', just before its '}')."""
    # find the function DEFINITION: 'name (' preceded by __global__/__device__/void
    # nearby (skips launch sites and forward references).
    import re
    i = -1
    for m in re.finditer(re.escape(name) + r"\s*\(", src):
        pre = src[max(0, m.start() - 400):m.start()]
        if "__global__" in pre or "__device__" in pre or "forceinline" in pre:
            i = m.start()
            break
    assert i >= 0, f"{name} definition not found"
    j = src.index("{", i)
    depth, k = 1, j + 1
    while depth:
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
        k += 1
    return j + 1, k - 1


def pin_asm(n):
    names = ", ".join(f"pin{i}" for i in range(n))
    decl = "".join(f"    float pin{i} = __int_as_float(threadIdx.x + {i});\n" for i in range(n))
    use = f'    asm volatile("" : ' + ", ".join(f'"+v"(pin{i})' for i in range(n)) + ");\n"
    return decl, use


def inject(src: str, name: str, n: int) -> str:
    e, x = find_span(src, name)
    decl, use = pin_asm(n)
    marker = f"// P2-regprobe n={n}\n"
    return (src[:e] + "\n    " + marker + decl + use +
            src[e:x] + "\n    " + use +
            # keep pins alive past the asm via a dead conditional sink
            (f"    if ({' + '.join(f'pin{i}' for i in range(n))} == 123456.f) "
             "__threadfence();\n" if n else "") + src[x:])


if __name__ == "__main__":
    tree, hdr, kname, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    f = Path(tree) / "csrc" / "g2" / hdr
    s = f.read_text()
    tag = f"// P2-regprobe n={n}"
    if tag in s:
        print("already")
    else:
        f.write_text(inject(s, kname, n))
        print(f"injected n={n} -> {hdr}:{kname}")
