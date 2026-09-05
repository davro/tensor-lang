"""
tlkit: small shared toolkit for building visual/interactive apps on top
of TensorLang.

TensorLang compiles ahead-of-time per .tl file: save() targets are
fixed string literals, and rebind (`x = x_updated`) needs a fixed-shape
GPU buffer each iteration. Practically, that means a single main.tl
can't accumulate a growing animation history inside one `for` loop.
The pattern that does work — proven out in apps/examples/decision_boundary
— is: main.tl load()s/save()s its own state each run, and gets invoked
repeatedly as separate OS processes, each one a "chunk" resuming from
the last. tlkit packages up the pieces every such app needs:

    chunked_runner   — drive repeated `tensorlang.py --app ...` invocations,
                        collect one frame per chunk into a stacked array
    sequence_player  — pygame play/pause/scrub/speed harness for the result
    colormap         — small numpy colormap + 2D grid-generation helpers

See apps/tlkit/README.md for a walkthrough, and
apps/examples/decision_boundary/tools/ for a full worked example.
"""
from . import chunked_runner
from . import sequence_player
from . import colormap

__all__ = ["chunked_runner", "sequence_player", "colormap"]
