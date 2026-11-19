"""Inspection, bounds checking, and waveshaper signal analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BoundsReport:
    target_lo: float
    target_hi: float
    y_min: float
    y_max: float
    y_mean: float
    y_rms: float
    start_val: float
    mid_val: float
    end_val: float
    clip_low_pct: float
    clip_high_pct: float
    nan_count: int
    inf_count: int
    monotonic: str  # "Increasing", "Decreasing", "Constant", "Non-monotonic"
    discontinuities: int
    within_bounds: bool

    @property
    def total_clip_pct(self) -> float:
        return self.clip_low_pct + self.clip_high_pct

    @property
    def span(self) -> float:
        return self.y_max - self.y_min


def analyze_curve(
    xs: np.ndarray,
    ys: np.ndarray,
    target_lo: float = 0.0,
    target_hi: float = 1.0,
) -> BoundsReport:
    n = len(ys)
    nan_mask = np.isnan(ys)
    inf_mask = np.isinf(ys)
    finite_mask = ~(nan_mask | inf_mask)

    nan_count = int(np.sum(nan_mask))
    inf_count = int(np.sum(inf_mask))

    if not np.any(finite_mask):
        return BoundsReport(
            target_lo=target_lo,
            target_hi=target_hi,
            y_min=float("nan"),
            y_max=float("nan"),
            y_mean=float("nan"),
            y_rms=float("nan"),
            start_val=float("nan"),
            mid_val=float("nan"),
            end_val=float("nan"),
            clip_low_pct=0.0,
            clip_high_pct=0.0,
            nan_count=nan_count,
            inf_count=inf_count,
            monotonic="Invalid",
            discontinuities=0,
            within_bounds=False,
        )

    finite_y = ys[finite_mask]
    y_min = float(np.min(finite_y))
    y_max = float(np.max(finite_y))
    y_mean = float(np.mean(finite_y))
    y_rms = float(np.sqrt(np.mean(np.square(finite_y))))

    start_val = float(ys[0]) if np.isfinite(ys[0]) else float("nan")
    mid_val = float(ys[n // 2]) if np.isfinite(ys[n // 2]) else float("nan")
    end_val = float(ys[-1]) if np.isfinite(ys[-1]) else float("nan")

    eps = 1e-7
    clip_low = int(np.sum(finite_y < (target_lo - eps)))
    clip_high = int(np.sum(finite_y > (target_hi + eps)))
    clip_low_pct = 100.0 * clip_low / max(len(finite_y), 1)
    clip_high_pct = 100.0 * clip_high / max(len(finite_y), 1)

    # Monotonicity check on finite samples
    diffs = np.diff(finite_y)
    if len(diffs) == 0 or np.all(np.abs(diffs) <= 1e-10):
        monotonic = "Constant"
    elif np.all(diffs >= -1e-9):
        monotonic = "Increasing"
    elif np.all(diffs <= 1e-9):
        monotonic = "Decreasing"
    else:
        monotonic = "Non-monotonic"

    # Discontinuity heuristic: single-step jump exceeding 20% of total span (min 0.08)
    span = max(y_max - y_min, target_hi - target_lo)
    jump_thresh = max(0.08, 0.2 * span)
    discontinuities = int(np.sum(np.abs(diffs) > jump_thresh))

    within_bounds = (
        nan_count == 0
        and inf_count == 0
        and clip_low == 0
        and clip_high == 0
    )

    return BoundsReport(
        target_lo=target_lo,
        target_hi=target_hi,
        y_min=y_min,
        y_max=y_max,
        y_mean=y_mean,
        y_rms=y_rms,
        start_val=start_val,
        mid_val=mid_val,
        end_val=end_val,
        clip_low_pct=clip_low_pct,
        clip_high_pct=clip_high_pct,
        nan_count=nan_count,
        inf_count=inf_count,
        monotonic=monotonic,
        discontinuities=discontinuities,
        within_bounds=within_bounds,
    )


def generate_test_waveform(
    kind: str = "Sine",
    cycles: float = 2.0,
    samples: int = 2048,
    pre_gain: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a bipolar test waveform in [-pre_gain, +pre_gain] for waveshaper preview."""
    t = np.linspace(0.0, cycles, samples, endpoint=False)
    phase = t % 1.0
    if kind == "Sine":
        wave = np.sin(2.0 * np.pi * t)
    elif kind == "Triangle":
        wave = 1.0 - 4.0 * np.abs(phase - 0.5)
    elif kind == "Sawtooth":
        wave = 2.0 * phase - 1.0
    else:
        wave = np.sin(2.0 * np.pi * t)
    return t, wave * pre_gain


def compute_harmonic_spectrum(
    shaped_wave: np.ndarray,
    cycles: int = 8,
    max_harmonic: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute harmonic magnitudes in dB (relative to 0 dBFS) for integer-cycle test wave."""
    clean = np.nan_to_num(shaped_wave, nan=0.0, posinf=1.0, neginf=-1.0)
    n = len(clean)
    spec = np.abs(np.fft.rfft(clean)) * (2.0 / n)
    # Bin 0 is DC, bin `cycles * k` is k-th harmonic
    harmonics = np.arange(0, max_harmonic + 1)
    bins = np.clip(harmonics * cycles, 0, len(spec) - 1)
    mags = spec[bins]
    # DC bin shouldn't be doubled
    mags[0] = spec[0] * 0.5
    db = 20.0 * np.log10(np.maximum(mags, 1e-6))
    return harmonics.astype(np.float64), np.clip(db, -96.0, 12.0)
