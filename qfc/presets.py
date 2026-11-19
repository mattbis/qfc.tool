"""Curated formula presets for FL Studio Mapping, Formula Controller, and Waveshapers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class FormulaPreset:
    name: str
    category: str
    mode: str  # "mapping" | "controller" | "transfer"
    dialect: str
    formula: str
    description: str
    defaults: Dict[str, float] = field(default_factory=dict)


PRESETS: List[FormulaPreset] = [
    # --- FL Mapping Formulas (Input in [0, 1]) ---
    FormulaPreset(
        name="FL Mapping: Tension Curve",
        category="FL Mapping Formula",
        mode="mapping",
        dialect="fl_studio",
        formula="Tension(Input, rate)",
        description="FL Studio's native Tension curve (+rate concave, -rate convex).",
        defaults={"rate": 25.0},
    ),
    FormulaPreset(
        name="FL Mapping: S-Curve (Dual Tension)",
        category="FL Mapping Formula",
        mode="mapping",
        dialect="fl_studio",
        formula=(
            "// Smooth S-curve using FL's native Tension()\n"
            "Case(\n"
            "  IfL(Input, 0.5),\n"
            "  Tension(Input * 2, bend) * 0.5,\n"
            "  0.5 + Tension((Input - 0.5) * 2, -bend) * 0.5\n"
            ")"
        ),
        description="Symmetric S-curve — great for crossfades or responsive filter sweeps.",
        defaults={"bend": 20.0},
    ),
    FormulaPreset(
        name="FL Mapping: Smoothstep",
        category="FL Mapping Formula",
        mode="mapping",
        dialect="fl_studio",
        formula="Input^2 * (3 - 2 * Input)",
        description="Classic cubic Hermite ease-in / ease-out curve.",
    ),
    FormulaPreset(
        name="FL Mapping: Quantized Steps",
        category="FL Mapping Formula",
        mode="mapping",
        dialect="fl_studio",
        formula="Int(Input * steps) / Max(1, steps - 1)",
        description="Snaps continuous automation into discrete equal steps.",
        defaults={"steps": 8.0},
    ),
    FormulaPreset(
        name="FL Mapping: Center Deadzone",
        category="FL Mapping Formula",
        mode="mapping",
        dialect="fl_studio",
        formula="Case(Inter(Input, 0.5 - dz, 0.5 + dz), 0.5, Input)",
        description="Locks knob output at 0.5 within a configurable center deadzone.",
        defaults={"dz": 0.12},
    ),
    FormulaPreset(
        name="FL Mapping: Bell / Bandpass Window",
        category="FL Mapping Formula",
        mode="mapping",
        dialect="fl_studio",
        formula="Sin(Input * Pi)^sharpness",
        description="Peaks at 1.0 in the middle (0.5) and returns to 0 at both ends.",
        defaults={"sharpness": 2.0},
    ),
    FormulaPreset(
        name="FL Mapping: Sub-range Remap (Clamped)",
        category="FL Mapping Formula",
        mode="mapping",
        dialect="fl_studio",
        formula="Min(1, Max(0, (Input - lo) / (hi - lo)))",
        description="Maps only the [lo, hi] portion of the knob to the full 0..1 range.",
        defaults={"lo": 0.2, "hi": 0.8},
    ),
    FormulaPreset(
        name="FL Mapping: Inverted",
        category="FL Mapping Formula",
        mode="mapping",
        dialect="fl_studio",
        formula="1 - Input",
        description="Flips knob/automation direction.",
    ),

    # --- Fruity Formula Controller (a, b, c, SongTime) ---
    FormulaPreset(
        name="Formula Controller: Synced Sine LFO",
        category="Fruity Formula Controller",
        mode="controller",
        dialect="fl_studio",
        formula=(
            "// a = center offset, b = depth, c = speed multiplier\n"
            "a + (b - 0.5) * Sin(SongTime * Pi * (1 + Int(c * 7)))"
        ),
        description="Classic beat-synced sine LFO controlled by knobs a, b, c.",
        defaults={"a": 0.5, "b": 0.85, "c": 0.25},
    ),
    FormulaPreset(
        name="Formula Controller: Synced Sawtooth",
        category="Fruity Formula Controller",
        mode="controller",
        dialect="fl_studio",
        formula="a + (b - a) * Frac(SongTime * freq)",
        description="Beat-synced sawtooth sweeping between knob a and knob b.",
        defaults={"a": 0.1, "b": 0.9, "freq": 1.0},
    ),
    FormulaPreset(
        name="Formula Controller: Synced Triangle",
        category="Fruity Formula Controller",
        mode="controller",
        dialect="fl_studio",
        formula="1 - Abs(Frac(SongTime * freq) * 2 - 1)",
        description="Linear beat-synced triangle LFO.",
        defaults={"freq": 0.5},
    ),
    FormulaPreset(
        name="Formula Controller: PWM Gate / Square",
        category="Fruity Formula Controller",
        mode="controller",
        dialect="fl_studio",
        formula="IfL(Frac(SongTime * rate), b) * a",
        description="Beat-synced pulse/gate where a = height and b = pulse width.",
        defaults={"a": 0.9, "b": 0.35, "rate": 2.0},
    ),
    FormulaPreset(
        name="Formula Controller: Stepped Random (S&H)",
        category="Fruity Formula Controller",
        mode="controller",
        dialect="fl_studio",
        formula="a + b * (SeededRand(Int(SongTime * rate)) - 0.5)",
        description="Tempo-synced Sample & Hold stepped random modulation.",
        defaults={"a": 0.5, "b": 0.8, "rate": 4.0},
    ),
    FormulaPreset(
        name="Formula Controller: Bouncing Decay",
        category="Fruity Formula Controller",
        mode="controller",
        dialect="fl_studio",
        formula="Abs(Sin(SongTime * Pi * speed)) * (1 - Frac(SongTime / bars))^decay",
        description="Rhythmic decaying bounce over a bar loop.",
        defaults={"speed": 2.0, "bars": 4.0, "decay": 1.5},
    ),
    FormulaPreset(
        name="Original QFC Demo Formula",
        category="Fruity Formula Controller",
        mode="mapping",
        dialect="fl_studio",
        formula="0.7-(((2-(((Cos(x*Pi*c1))*Cos(x*Pi*c2)*amp)*mul))*scale)/norm)*(Sin(x*freq/Pi))",
        description="The original default formula from qfc.py v1.",
        defaults={"c1": 1.0, "c2": 1.0, "amp": 1.0, "mul": 1.0, "scale": 1.0, "norm": 1.0, "freq": 1.0},
    ),

    # --- Transfer Functions / Waveshapers (x in [-1, 1]) ---
    FormulaPreset(
        name="Waveshaper: Tanh Soft Clipper",
        category="Transfer Function / Waveshaper",
        mode="transfer",
        dialect="fl_studio",
        formula="TanH(x * drive) / TanH(drive)",
        description="Warm normalized tanh saturation (odd harmonics).",
        defaults={"drive": 3.5},
    ),
    FormulaPreset(
        name="Waveshaper: Asymmetric Tube Warmth",
        category="Transfer Function / Waveshaper",
        mode="transfer",
        dialect="fl_studio",
        formula=(
            "// Asymmetric saturation introduces even + odd harmonics\n"
            "Case(\n"
            "  IfGE(x, 0),\n"
            "  TanH(x * drive) / TanH(drive),\n"
            "  TanH(x * drive * asym) / TanH(drive * asym)\n"
            ")"
        ),
        description="Different positive/negative saturation knees for even harmonics.",
        defaults={"drive": 3.0, "asym": 0.55},
    ),
    FormulaPreset(
        name="Waveshaper: Sine Wavefolder",
        category="Transfer Function / Waveshaper",
        mode="transfer",
        dialect="fl_studio",
        formula="Sin(x * Pi * folds)",
        description="West-coast wavefolder creating rich upper harmonics.",
        defaults={"folds": 1.75},
    ),
    FormulaPreset(
        name="Waveshaper: Hard Clipper",
        category="Transfer Function / Waveshaper",
        mode="transfer",
        dialect="fl_studio",
        formula="Min(thresh, Max(-thresh, x)) / thresh",
        description="Symmetric hard clipper normalized back to [-1, 1].",
        defaults={"thresh": 0.5},
    ),
    FormulaPreset(
        name="Waveshaper: Bitcrusher / Quantizer",
        category="Transfer Function / Waveshaper",
        mode="transfer",
        dialect="fl_studio",
        formula="Round(x * steps) / steps",
        description="Amplitude quantization staircase.",
        defaults={"steps": 4.0},
    ),
    FormulaPreset(
        name="Waveshaper: Cubic Soft Clip",
        category="Transfer Function / Waveshaper",
        mode="transfer",
        dialect="fl_studio",
        formula="Case(IfGE(Abs(x * drive), 1), Sign(x) * 2 / 3, (x * drive) - ((x * drive)^3) / 3) * 1.5",
        description="Classic 3rd-order polynomial soft clipper.",
        defaults={"drive": 1.5},
    ),
    FormulaPreset(
        name="Waveshaper (GLSL): Smoothstep Knee",
        category="Transfer Function / Waveshaper",
        mode="transfer",
        dialect="dsp_glsl",
        formula="mix(x, tanh(x * drive), blend)",
        description="Parallel dry/wet tanh saturation using DSP/GLSL dialect.",
        defaults={"drive": 4.0, "blend": 0.75},
    ),
]
