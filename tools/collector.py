#!/usr/bin/env python3
"""Collect source files into one text stream, to upload as a FILE (not pasted) to a new chat.

An uploaded file lets the assistant run the real parser, AST builder and type checker off-GPU.

    python3 tools/collector.py > code.txt                  # preset "core" (the default)
    python3 tools/collector.py --preset probes > probes.txt
    python3 tools/collector.py --preset docs > docs.txt    # README, RELEASES and docs/*.md
    python3 tools/collector.py tensorlang/compiler.py tests/mlp.tl > some.txt   # explicit files
    python3 tools/collector.py "docs/social/**/*.md" > social.txt               # globs are expanded
    python3 tools/collector.py --list                      # show the presets and what is missing
    python3 tools/collector.py --out code.txt              # write the file directly

Read-only: prints to stdout (or --out); the summary and any missing files go to stderr. Paths are
relative to the repo root, wherever you run it from. Format per file:

    ================================================================================
    # FILE: tensorlang/compiler.py
    ================================================================================
    <contents>
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BAR = "=" * 80
GLOB_CHARS = set("*?[")


def glob(pattern):
    """Sorted repo-relative paths of the files matching a pattern (empty if none or no such dir)."""
    return sorted(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern) if p.is_file())


PRESETS = {
    # Everything needed to run the parser, AST builder, type checker and to read codegen/execution.
    "core": [
        "tensorlang.lark",
        "tensorlang/ast_builder.py", "tensorlang/type_checker.py", "tensorlang/codegen.py",
        "tensorlang/compiler.py", "tensorlang/execution.py", "tensorlang/autograd.py", "tensorlang/io.py",
        "tensorlang/constants.py", "tensorlang/tensor_lang.py", "tensorlang/fusion.py",
        "tensorlang/kernel_generator.py", "tensorlang/interop.py", "tensorlang/gpu_backward.py",
        "tensorlang/test_runner.py", "tensorlang/tensor_verifier.py",
        "tensorlang/generators/base.py", "tensorlang/generators/nn.py", "tensorlang/generators/shapes.py",
        "tensorlang/generators/comparisons.py", "tensorlang/generators/elementwise.py",
        "tensorlang/generators/reductions.py", "tensorlang/generators/unary.py",
        "tensorlang/generators/optimizers.py",
        "build.sh",
    ],
    # The transformer probes and the tools that generate / debug them.
    "probes": [
        "tests/probe_attention_block.tl", "tests/probe_broadcast.tl", "tests/probe_causal_mask.tl",
        "tests/probe_embedding_onehot.tl", "tests/probe_gelu.tl", "tests/probe_gpt2_forward.tl",
        "tests/probe_growing_concat_loop.tl", "tests/probe_layernorm.tl",
        "tools/make_gpt2_probe.py", "tools/compare_probe_intermediates.py",
    ],
    # The sweep / check scripts (so a new chat can see how each was written).
    "tools": glob("tools/*.py"),
    # The README, the release history and the guides in docs/ (top level only: docs/social and
    # docs/handovers are left out; use the "handovers" preset or an explicit glob for those).
    "docs": ["README.md", "RELEASES.md"] + glob("docs/*.md"),
    # Session handover notes.
    "handovers": glob("docs/handovers/*.md"),
}


def expand(args):
    """Expand glob patterns in explicit arguments; a pattern with no match is kept so it shows as MISSING."""
    files = []
    for a in args:
        found = glob(a) if GLOB_CHARS & set(a) else []
        files.extend(found or [a])
    return files


def render(files):
    missing, out, total = [], [], 0
    for f in files:
        p = ROOT / f
        out.append(f"\n{BAR}\n# FILE: {f}\n{BAR}")
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            total += len(text.encode("utf-8"))
            out.append(text if text.endswith("\n") else text + "\n")
        else:
            missing.append(f)
            out.append("MISSING\n")
    return "\n".join(out), missing, total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*",
                    help="explicit files or glob patterns (relative to the repo root); overrides --preset")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="core")
    ap.add_argument("--list", action="store_true", help="list the presets and flag missing files, then exit")
    ap.add_argument("--out", help="write to this file instead of stdout")
    a = ap.parse_args()

    if a.list:
        for name, fs in PRESETS.items():
            gone = [f for f in fs if not (ROOT / f).is_file()]
            print(f"{name}: {len(fs)} files" + (f", MISSING: {', '.join(gone)}" if gone else ""))
        return 0

    files = expand(a.files) if a.files else PRESETS[a.preset]
    text, missing, total = render(files)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    print(f"collected {len(files) - len(missing)} of {len(files)} files, {total / 1024:.0f} KB"
          + (f" -> {a.out}" if a.out else ""), file=sys.stderr)
    if missing:
        print("MISSING: " + ", ".join(missing), file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
