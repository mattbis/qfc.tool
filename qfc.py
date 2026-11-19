"""Quick Formula Check (QFC) — FL Studio Formula & DSP Transfer Function Lab.

Launch GUI:
    python qfc.py

Quick CLI bounds check (headless):
    python qfc.py --check "Tension(Input, 40)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def run_cli_check(formula: str, dialect_id: str, mode: str) -> int:
    from qfc.analysis import analyze_curve
    from qfc.dialects import DialectRegistry
    from qfc.exporter import export_to_fl_studio
    from qfc.parser import FormulaError, evaluate_ast, extract_custom_params, parse_formula

    repo_root = Path(__file__).resolve().parent
    registry = DialectRegistry(extra_dir=repo_root / "dialects")
    dialect = registry.get(dialect_id)

    try:
        ast = parse_formula(formula, dialect)
    except FormulaError as err:
        print(f"SYNTAX ERROR: {err.message}", file=sys.stderr)
        return 1

    x_lo, x_hi = (-1.0, 1.0) if mode == "transfer" else (0.0, 4.0 if mode == "controller" else 1.0)
    t_lo, t_hi = (-1.0, 1.0) if mode == "transfer" else (0.0, 1.0)

    xs = np.linspace(x_lo, x_hi, 4096)
    params = {p: 1.0 for p in extract_custom_params(ast, dialect)}
    env = {
        **params,
        "x": xs,
        "input": xs,
        "Input": xs,
        "songtime": xs,
        "SongTime": xs,
        "time": xs * (60000.0 / 128.0),
        "a": 0.5,
        "b": 0.5,
        "c": 0.5,
    }
    ys = evaluate_ast(ast, dialect, env, size=len(xs))
    report = analyze_curve(xs, ys, target_lo=t_lo, target_hi=t_hi)
    fl_out, warnings = export_to_fl_studio(ast, dialect, param_values=params, bake_custom_params=True)

    print(f"Dialect:    {dialect.name}  (mode={mode})")
    print(f"Range:      min={report.y_min:.5f}  max={report.y_max:.5f}  mean={report.y_mean:.5f}")
    print(f"Endpoints:  start={report.start_val:.5f}  mid={report.mid_val:.5f}  end={report.end_val:.5f}")
    print(f"Bounds:     [{t_lo:g}, {t_hi:g}] -> {'OK' if report.within_bounds else f'CLIPPED {report.total_clip_pct:.2f}%'}")
    print(f"Health:     NaN={report.nan_count}  Inf={report.inf_count}  Shape={report.monotonic}")
    print(f"FL Export:  {fl_out}")
    for w in warnings:
        print(f"Warning:    {w}")
    return 0


def run_gui() -> int:
    from PySide6.QtWidgets import QApplication

    from qfc.ui.main_window import MainWindow
    from qfc.ui.theme import DARK_QSS

    app = QApplication(sys.argv)
    app.setApplicationName("QFC — FL Studio Formula Lab")
    app.setStyleSheet(DARK_QSS)

    repo_root = Path(__file__).resolve().parent
    window = MainWindow(dialects_dir=repo_root / "dialects")
    window.show()
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser(description="QFC — FL Studio Formula & DSP Transfer Function Lab")
    parser.add_argument("--check", metavar="FORMULA", help="Headless CLI bounds check for a formula")
    parser.add_argument("--dialect", default="fl_studio", help="Dialect ID (fl_studio, python, dsp_glsl)")
    parser.add_argument("--mode", default="mapping", choices=["mapping", "controller", "transfer"])
    args = parser.parse_args()

    if args.check:
        return run_cli_check(args.check, args.dialect, args.mode)
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
