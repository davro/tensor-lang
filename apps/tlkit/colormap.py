"""Small numpy colormap + 2D-grid-generation helpers for tlkit apps."""
from typing import Tuple

import numpy as np


def diverging(
    p: np.ndarray,
    low: Tuple[int, int, int] = (40, 70, 220),
    mid: Tuple[int, int, int] = (245, 245, 245),
    high: Tuple[int, int, int] = (220, 55, 55),
) -> np.ndarray:
    """p: (...,) float array in [0,1] -> (..., 3) uint8.
    Default is blue (0) -> white (0.5) -> red (1), good for binary
    classification / signed-quantity fields."""
    low_c, mid_c, high_c = (np.array(c, dtype=np.float32) for c in (low, mid, high))
    p = np.clip(p, 0.0, 1.0)
    t_low = np.clip(p * 2.0, 0.0, 1.0)[..., None]
    lower = low_c * (1 - t_low) + mid_c * t_low
    t_high = np.clip((p - 0.5) * 2.0, 0.0, 1.0)[..., None]
    upper = mid_c * (1 - t_high) + high_c * t_high
    color = np.where((p < 0.5)[..., None], lower, upper)
    return color.astype(np.uint8)


def sequential(
    p: np.ndarray,
    low: Tuple[int, int, int] = (20, 20, 30),
    high: Tuple[int, int, int] = (255, 220, 60),
) -> np.ndarray:
    """p: (...,) float array in [0,1] -> (..., 3) uint8. Plain linear
    ramp, good for magnitude/intensity fields (e.g. a density map)
    rather than signed ones."""
    low_c, high_c = (np.array(c, dtype=np.float32) for c in (low, high))
    t = np.clip(p, 0.0, 1.0)[..., None]
    return (low_c * (1 - t) + high_c * t).astype(np.uint8)


def grid_to_surface(frame: np.ndarray, window_size: int, cmap=diverging):
    """frame: (side, side) float array, [row=y_index, col=x_index] ->
    a scaled pygame Surface, flipped so larger y renders higher on
    screen. Requires pygame (imported lazily so colormap.py itself has
    no hard pygame dependency for non-visual uses/tests)."""
    import pygame

    color = cmap(frame)             # (side, side, 3), [y, x]
    color = np.flipud(color)        # larger y -> higher on screen
    color = color.transpose(1, 0, 2)  # -> [x, y, 3], what surfarray expects
    surf = pygame.surfarray.make_surface(color)
    return pygame.transform.smoothscale(surf, (window_size, window_size))


def make_grid(side: int, extent: Tuple[float, float] = (-0.5, 1.5)) -> np.ndarray:
    """Flattened (side*side, 2) float32 grid of 2D points over
    extent x extent, row-major — matches the [y, x] reshape order
    grid_to_surface expects when you reshape a model's per-point
    predictions back to (side, side)."""
    lo, hi = extent
    xs = np.linspace(lo, hi, side, dtype=np.float32)
    ys = np.linspace(lo, hi, side, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


def data_to_pixel(x: float, y: float, lo: float, hi: float, window_size: int) -> Tuple[int, int]:
    """Map a data-space (x, y) in [lo, hi]^2 to pixel coords matching
    grid_to_surface's flip, so overlays (e.g. scatter points) line up
    with the rendered heatmap."""
    px = (x - lo) / (hi - lo) * window_size
    py = window_size - (y - lo) / (hi - lo) * window_size
    return int(px), int(py)
