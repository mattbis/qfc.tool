# tool-quick_formula_check (QFC)

**FL Studio Formula & DSP Transfer Function Lab** — a dark-themed PySide6 + PyQtGraph desktop app for writing, visualising, inspecting, and exporting FL Studio Mapping Formulas, Fruity Formula Controller expressions, and bipolar waveshaper transfer functions without fighting FL Studio's tiny single-line input box.

## Features

- **Three Dedicated Visualisation Modes**
  - **FL Mapping `[0..1]`**: Plots `Input` on `[0, 1]` with `y = x` unity reference, FL `[0, 1]` clip-region shading, and simultaneous Raw vs FL-Clipped curve overlay.
  - **Formula Controller**: Sweeps `SongTime` across a configurable beat window, includes dedicated `[0, 1]` knobs for FL's `a`, `b`, `c`, and a live BPM-synced **Play LFO** playhead & output meter.
  - **Transfer / Waveshaper `[-1..1]`**: Plots bipolar transfer functions with an integrated **Waveform Preview** (Sine / Triangle / Sawtooth + Pre-Gain) and **Harmonic Spectrum (FFT dB)** view.
- **Multi-Line Dialect-Aware Editor**
  - Supports multi-line formatting, `//` & `#` comments, syntax highlighting, and exact syntax-error span underlining.
  - Built-in **FL Studio** (case-insensitive `Tension`, `Case`, `IfL`/`IfLE`/`IfG`/`IfGE`/`IfE`, `Inter`, `InterNoSwap`, `Frac`, `Int`, `SeededRand`, `Rand`, `^`, `=`, `<>`, `and`/`or`/`xor`/`not`/`mod`/`div`, trig/hyperbolic/log), **Python / NumPy**, and **DSP / GLSL** dialects.
  - Drop custom `.json` dialect dictionaries into [`dialects/`](dialects/) to extend support for other DAWs or DSP languages.
  - Press **F1** (or **? Reference**) for a searchable syntax drawer with double-click-to-insert.
- **Inspection, Bounds Checks & Auto-Normalise**
  - Live hover crosshair `(x, raw y, FL clipped output)` and bounds bar (`Min/Max`, `Start → Mid → End`, `Mean/RMS`, `Clipped %`, `NaN/Inf`, monotonicity & jump detection).
  - One-click **Auto-Normalise** to remap a formula cleanly into `[0, 1]` or `[-1, 1]`.
  - **FL Studio Single-Line Export**: strips comments/newlines, translates non-FL functions (`clamp`, `mix`, `tanh`, `**`, `!=`, etc.) to FL Studio syntax, optionally bakes custom parameter sliders into numeric literals, clamps to `[0, 1]`, and copies straight to clipboard (**Ctrl+Shift+C**).

## Installation & Usage

```bash
pip install -r requirements.txt
python qfc.py
```

Headless CLI bounds check:

```bash
python qfc.py --check "Tension(Input, 40)"
python qfc.py --check "TanH(x * 3.5) / TanH(3.5)" --mode transfer
```

Run tests:

```bash
python -m pytest -v
```
