"""Comprehensive tests for QFC dialects, parser, FL Studio math, exporter, and analysis."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from qfc.analysis import analyze_curve, compute_harmonic_spectrum, generate_test_waveform
from qfc.dialects import DialectRegistry
from qfc.exporter import export_to_fl_studio, wrap_normalized_formula
from qfc.parser import FormulaError, evaluate_ast, extract_custom_params, parse_formula


@pytest.fixture
def registry():
    return DialectRegistry(extra_dir=Path(__file__).resolve().parent.parent / "dialects")


@pytest.fixture
def fl(registry):
    return registry.get("fl_studio")


def eval_fl(expr: str, fl, x=None, **kwargs):
    if x is None:
        x = np.linspace(0.0, 1.0, 101)
    ast = parse_formula(expr, fl)
    env = {"input": x, "x": x, "songtime": x, "a": 0.5, "b": 0.5, "c": 0.5, **kwargs}
    return evaluate_ast(ast, fl, env, size=len(x))


def test_fl_case_insensitivity_and_operators(fl):
    xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y1 = eval_fl("SIN(INPUT * PI) ^ 2", fl, xs)
    y2 = eval_fl("sin(input * pi) ^ 2", fl, xs)
    np.testing.assert_allclose(y1, y2)
    np.testing.assert_allclose(y1, np.sin(xs * np.pi) ** 2)


def test_fl_comparisons_and_logic(fl):
    xs = np.array([0.1, 0.5, 0.9])
    assert list(eval_fl("Input = 0.5", fl, xs)) == [0.0, 1.0, 0.0]
    assert list(eval_fl("Input <> 0.5", fl, xs)) == [1.0, 0.0, 1.0]
    assert list(eval_fl("(Input > 0.2) and (Input < 0.8)", fl, xs)) == [0.0, 1.0, 0.0]
    assert list(eval_fl("not (Input = 0.5)", fl, xs)) == [1.0, 0.0, 1.0]


def test_fl_tension_exact_symmetry(fl):
    xs = np.linspace(0.0, 1.0, 11)
    # Zero tension is linear
    np.testing.assert_allclose(eval_fl("Tension(Input, 0)", fl, xs), xs, atol=1e-9)
    # Endpoints always 0 and 1
    pos = eval_fl("Tension(Input, 40)", fl, xs)
    neg = eval_fl("Tension(Input, -40)", fl, xs)
    assert pos[0] == pytest.approx(0.0)
    assert pos[-1] == pytest.approx(1.0)
    assert pos[5] < 0.2  # concave / exponential slow start
    assert neg[5] > 0.8  # convex / logarithmic fast start
    # Exact mirror symmetry: Tension(x, -r) == 1 - Tension(1 - x, r)
    np.testing.assert_allclose(neg, 1.0 - pos[::-1], atol=1e-9)


def test_fl_case_if_and_inter(fl):
    xs = np.array([0.1, 0.5, 0.9])
    y_case = eval_fl("Case(IfL(Input, 0.5), 0.2, 0.8)", fl, xs)
    np.testing.assert_allclose(y_case, [0.2, 0.8, 0.8])

    # Inter auto-swaps bounds (0.8, 0.2) -> (0.2, 0.8)
    y_inter = eval_fl("Inter(Input, 0.8, 0.2)", fl, xs)
    np.testing.assert_allclose(y_inter, [0.0, 1.0, 0.0])

    # InterNoSwap does NOT swap -> always 0 when b > c
    y_noswap = eval_fl("InterNoSwap(Input, 0.8, 0.2)", fl, xs)
    np.testing.assert_allclose(y_noswap, [0.0, 0.0, 0.0])


def test_fl_int_frac_and_seeded_rand(fl):
    xs = np.array([-1.7, 0.0, 1.7])
    np.testing.assert_allclose(eval_fl("Int(Input)", fl, xs), [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(eval_fl("Frac(Input)", fl, xs), [-0.7, 0.0, 0.7])

    # SeededRand deterministic across repeated identical integer buckets
    st = np.array([0.1, 0.5, 0.9, 1.2, 1.8])
    sr = eval_fl("SeededRand(Int(SongTime))", fl, st)
    assert sr[0] == sr[1] == sr[2]
    assert sr[3] == sr[4]
    assert sr[0] != sr[3]
    assert np.all((sr >= 0.0) & (sr < 1.0))


def test_comments_and_multiline_parsing(fl):
    formula = """
    // FL S-curve test
    Case(
        IfL(Input, 0.5),   # first half
        Input * 2,
        1 /* constant top */
    )
    """
    xs = np.array([0.25, 0.75])
    np.testing.assert_allclose(eval_fl(formula, fl, xs), [0.5, 1.0])


def test_syntax_error_reports_span(fl):
    with pytest.raises(FormulaError) as exc:
        parse_formula("Sin(Input + )", fl)
    assert exc.value.start >= 0


def test_multi_dialect_and_json_loader(registry):
    ids = {d.id for d in registry.list_dialects()}
    assert {"fl_studio", "python", "dsp_glsl", "reaper_jsfx"}.issubset(ids)

    glsl = registry.get("dsp_glsl")
    xs = np.array([-1.0, 0.0, 1.0])
    ast = parse_formula("mix(x, tanh(x * 2), 0.5)", glsl)
    ys = evaluate_ast(ast, glsl, {"x": xs}, size=3)
    expected = xs * 0.5 + np.tanh(xs * 2.0) * 0.5
    np.testing.assert_allclose(ys, expected)


def test_exporter_bakes_params_and_translates_glsl_to_fl(registry):
    glsl = registry.get("dsp_glsl")
    ast = parse_formula("clamp(mix(x, tanh(x * drive), blend), 0, 1)", glsl)
    fl_str, warnings = export_to_fl_studio(
        ast,
        glsl,
        param_values={"drive": 3.0, "blend": 0.5},
        bake_custom_params=True,
        primary_var_target="Input",
    )
    assert not warnings
    assert "TanH" in fl_str
    assert "Min(" in fl_str and "Max(" in fl_str
    assert "Input" in fl_str
    assert "drive" not in fl_str

    # Verify exported FL formula parses & evaluates identically under FL Studio dialect!
    fl = registry.get("fl_studio")
    xs = np.linspace(0.0, 1.0, 50)
    y_src = evaluate_ast(ast, glsl, {"x": xs, "drive": 3.0, "blend": 0.5}, size=len(xs))
    y_fl = evaluate_ast(parse_formula(fl_str, fl), fl, {"input": xs}, size=len(xs))
    np.testing.assert_allclose(y_src, y_fl, atol=1e-6)


def test_auto_normalise_and_bounds_analysis(fl):
    xs = np.linspace(0.0, 1.0, 501)
    # Formula with range [-2, 6]
    ys = eval_fl("Input * 8 - 2", fl, xs)
    rep = analyze_curve(xs, ys, target_lo=0.0, target_hi=1.0)
    assert not rep.within_bounds
    assert rep.clip_low_pct > 20.0
    assert rep.clip_high_pct > 60.0
    assert rep.monotonic == "Increasing"

    norm_expr = wrap_normalized_formula("Input * 8 - 2", rep.y_min, rep.y_max, 0.0, 1.0)
    ys_norm = eval_fl(norm_expr, fl, xs)
    rep_norm = analyze_curve(xs, ys_norm, target_lo=0.0, target_hi=1.0)
    assert rep_norm.within_bounds
    assert rep_norm.y_min == pytest.approx(0.0, abs=1e-5)
    assert rep_norm.y_max == pytest.approx(1.0, abs=1e-5)


def test_waveshaper_harmonics_analysis(fl):
    # Pure sine -> fundamental dominant, tiny harmonics
    _, sine = generate_test_waveform("Sine", cycles=8.0, samples=4096)
    h, db = compute_harmonic_spectrum(sine, cycles=8, max_harmonic=8)
    assert db[1] == pytest.approx(0.0, abs=0.1)
    assert db[3] < -80.0

    # Tanh soft clip adds odd harmonics (3rd > 2nd)
    shaped = eval_fl("TanH(x * 4)", fl, sine)
    _, db_clip = compute_harmonic_spectrum(shaped, cycles=8, max_harmonic=8)
    assert db_clip[3] > -25.0
    assert db_clip[3] > db_clip[2] + 30.0
