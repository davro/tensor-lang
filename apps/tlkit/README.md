# tlkit

Shared helpers for building visual/interactive apps on top of TensorLang.
Extracted from `apps/examples/decision_boundary` once that pattern proved
out — use it so the next example doesn't reinvent (and re-debug) the
same plumbing.

## Why this exists

TensorLang compiles ahead-of-time from a single `.tl` file, `save()`
targets are fixed string literals (no `save(x, f"frame_{i}.npy")`), and
rebind (`x = x_updated`) needs a fixed-shape GPU buffer every
iteration. Net effect: a single `main.tl` can't accumulate a growing
animation history inside one `for` loop.

The pattern that does work: `main.tl` `load()`s/`save()`s its own state
to a fixed path every run, and gets invoked repeatedly as separate
processes — each invocation is a "chunk" that resumes from the last.
If your app is pure playback with no training/simulation loop (e.g. a
physics sim computed once and just scrubbed), you don't need chunking
at all — just `sequence_player` + `colormap`.

## Modules

- **`chunked_runner`** — `run_chunks(app_category_path, n_chunks, collect_fn)`
  drives repeated `python3 tensorlang.py --app ...` invocations and
  stacks one collected frame per chunk into a single array.
  `cache_dir_for(app_category_path)` computes exactly where that app's
  `save()` calls land, so you don't have to hand-write the
  `cache/apps/.../main.tl/...` path yourself.
- **`sequence_player`** — `SequencePlayer(frames, render_fn, overlay_fn=...).run()`
  is the pygame play/pause/scrub/speed loop. `render_fn` turns one
  frame into a `pygame.Surface`; `overlay_fn` draws anything extra
  (scatter points, labels) on top each frame.
- **`colormap`** — `diverging`/`sequential` numpy colormaps,
  `grid_to_surface` (frame array -> scaled Surface), `make_grid` (2D
  evaluation grid for e.g. a decision-boundary sweep), `data_to_pixel`
  (line up overlays with the rendered grid).

## Using it from an app's tools/ scripts

`apps/tlkit` isn't itself a runnable app, so import it by adding
`apps/` to `sys.path`:

```python
import sys
from pathlib import Path
# from apps/examples/<name>/tools/script.py, parents[3] is apps/ itself
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner, sequence_player, colormap
```

See `apps/examples/decision_boundary/tools/` for a full worked example
(training-chunk driver + viewer, both built on tlkit).
