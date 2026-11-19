"""Multi-dialect math & DSP dictionary registry.

Supports built-in dialects (FL Studio, Python / NumPy, DSP / GLSL) and
extensible JSON dialect definitions placed in the ``dialects/`` directory.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Vectorized math / FL Studio function implementations
# ---------------------------------------------------------------------------

def _as_arr(v: Any) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


def _broadcast(*args: Any) -> Tuple[np.ndarray, ...]:
    arrs = [_as_arr(a) for a in args]
    return tuple(np.broadcast_arrays(*arrs))


def fn_sin(x):
    return np.sin(x)


def fn_cos(x):
    return np.cos(x)


def fn_tan(x):
    return np.tan(x)


def fn_ctg(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.tan(x)


def fn_sec(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.cos(x)


def fn_cosec(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.sin(x)


def fn_arcsin(x):
    with np.errstate(invalid="ignore"):
        return np.arcsin(x)


def fn_arccos(x):
    with np.errstate(invalid="ignore"):
        return np.arccos(x)


def fn_arctan(x):
    return np.arctan(x)


def fn_atan2(y, x):
    return np.arctan2(y, x)


def fn_sinh(x):
    with np.errstate(over="ignore"):
        return np.sinh(x)


def fn_cosh(x):
    with np.errstate(over="ignore"):
        return np.cosh(x)


def fn_tanh(x):
    return np.tanh(x)


def fn_exp(x):
    with np.errstate(over="ignore"):
        return np.exp(x)


def fn_ln(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(x)


def fn_log2(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(x)


def fn_log10(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(x)


def fn_sqrt(x):
    with np.errstate(invalid="ignore"):
        return np.sqrt(x)


def fn_sqr(x):
    return np.square(x)


def fn_pow(x, y):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.power(_as_arr(x), _as_arr(y))


def fn_abs(x):
    return np.abs(x)


def fn_neg(x):
    return -_as_arr(x)


def fn_sign(x):
    return np.sign(x)


def fn_int(x):
    """FL Studio / Delphi Int(x): truncates toward zero."""
    return np.trunc(x)


def fn_frac(x):
    """FL Studio Frac(x): fractional part (x - trunc(x))."""
    xa = _as_arr(x)
    return xa - np.trunc(xa)


def fn_floor(x):
    return np.floor(x)


def fn_ceil(x):
    return np.ceil(x)


def fn_round(x):
    return np.round(x)


def fn_min(*args):
    if len(args) < 2:
        raise ValueError("Min requires at least 2 arguments")
    res = _as_arr(args[0])
    for a in args[1:]:
        res = np.minimum(res, _as_arr(a))
    return res


def fn_max(*args):
    if len(args) < 2:
        raise ValueError("Max requires at least 2 arguments")
    res = _as_arr(args[0])
    for a in args[1:]:
        res = np.maximum(res, _as_arr(a))
    return res


def fn_clamp(x, lo, hi):
    return np.clip(_as_arr(x), _as_arr(lo), _as_arr(hi))


def fn_saturate(x):
    return np.clip(_as_arr(x), 0.0, 1.0)


def fn_mod(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.mod(_as_arr(a), _as_arr(b))


def fn_fmod(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.fmod(_as_arr(a), _as_arr(b))


def fn_mix(a, b, t):
    aa, ba, ta = _broadcast(a, b, t)
    return aa + (ba - aa) * ta


def fn_step(edge, x):
    e, xa = _broadcast(edge, x)
    return np.where(xa < e, 0.0, 1.0)


def fn_smoothstep(edge0, edge1, x):
    e0, e1, xa = _broadcast(edge0, edge1, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip((xa - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fn_tension(value, rate):
    """FL Studio Tension(value, rate) curve shaping function.

    Matches FL Studio's exponential curve mapping:
      rate > 0: ((rate + 1)^x - 1) / rate   (concave / slow start, fast end)
      rate < 0: 1 - (((-rate + 1)^(1 - x) - 1) / (-rate))  (convex mirror)
      rate == 0: linear x
    """
    x, r = _broadcast(value, rate)
    out = np.array(x, dtype=np.float64, copy=True)
    pos = r > 1e-9
    neg = r < -1e-9

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if np.any(pos):
            rp = r[pos]
            xp = x[pos]
            out[pos] = (np.power(rp + 1.0, xp) - 1.0) / rp
        if np.any(neg):
            kn = -r[neg]
            xn = x[neg]
            out[neg] = 1.0 - (np.power(kn + 1.0, 1.0 - xn) - 1.0) / kn
    return out


def fn_case(cond, if_true, if_false):
    c, t, f = _broadcast(cond, if_true, if_false)
    return np.where(c != 0.0, t, f)


def _cond_op(a, b, pred, if_true=None, if_false=None):
    aa, ba = _broadcast(a, b)
    mask = pred(aa, ba)
    if if_true is not None and if_false is not None:
        _, _, t, f = _broadcast(aa, ba, if_true, if_false)
        return np.where(mask, t, f)
    return np.where(mask, 1.0, 0.0)


def fn_ife(a, b, *rest):
    if len(rest) == 2:
        return _cond_op(a, b, lambda x, y: np.isclose(x, y, atol=1e-12), rest[0], rest[1])
    if rest:
        raise ValueError("IfE expects 2 arguments (or 4 with true/false outputs)")
    return _cond_op(a, b, lambda x, y: np.isclose(x, y, atol=1e-12))


def fn_ifl(a, b, *rest):
    if len(rest) == 2:
        return _cond_op(a, b, lambda x, y: x < y, rest[0], rest[1])
    if rest:
        raise ValueError("IfL expects 2 arguments (or 4 with true/false outputs)")
    return _cond_op(a, b, lambda x, y: x < y)


def fn_ifle(a, b, *rest):
    if len(rest) == 2:
        return _cond_op(a, b, lambda x, y: x <= y, rest[0], rest[1])
    if rest:
        raise ValueError("IfLE expects 2 arguments (or 4 with true/false outputs)")
    return _cond_op(a, b, lambda x, y: x <= y)


def fn_ifg(a, b, *rest):
    if len(rest) == 2:
        return _cond_op(a, b, lambda x, y: x > y, rest[0], rest[1])
    if rest:
        raise ValueError("IfG expects 2 arguments (or 4 with true/false outputs)")
    return _cond_op(a, b, lambda x, y: x > y)


def fn_ifge(a, b, *rest):
    if len(rest) == 2:
        return _cond_op(a, b, lambda x, y: x >= y, rest[0], rest[1])
    if rest:
        raise ValueError("IfGE expects 2 arguments (or 4 with true/false outputs)")
    return _cond_op(a, b, lambda x, y: x >= y)


def fn_inter(val, lo, hi):
    """FL Studio Inter(a, b, c): 1 if a is strictly between b and c (auto-swaps b, c)."""
    v, b, c = _broadcast(val, lo, hi)
    mn = np.minimum(b, c)
    mx = np.maximum(b, c)
    return np.where((v > mn) & (v < mx), 1.0, 0.0)


def fn_internoswap(val, lo, hi):
    """FL Studio InterNoSwap(a, b, c): 1 if b < a < c without swapping b and c."""
    v, b, c = _broadcast(val, lo, hi)
    return np.where((v > b) & (v < c), 1.0, 0.0)


def fn_seeded_rand(seed):
    """Deterministic pseudo-random float in [0, 1) keyed by numeric seed."""
    s = _as_arr(seed)
    # Deterministic hash over float64 values (gives smooth stepped S&H with Int(SongTime*n))
    h = np.sin(s * 12.9898 + 78.233) * 43758.5453123
    return h - np.floor(h)


def fn_where(cond, a, b):
    return fn_case(cond, a, b)


# Core function catalog indexed by canonical ID (used by both built-in and JSON dialects)
CORE_FUNCTIONS: Dict[str, Tuple[Callable[..., Any], str, str, str]] = {
    # id: (callable, signature, doc, category)
    "sin": (fn_sin, "Sin(x)", "Sine of x (radians).", "Trig"),
    "cos": (fn_cos, "Cos(x)", "Cosine of x (radians).", "Trig"),
    "tan": (fn_tan, "Tan(x)", "Tangent of x (radians). FL alias: Tg(x).", "Trig"),
    "ctg": (fn_ctg, "Ctg(x)", "Cotangent 1 / tan(x).", "Trig"),
    "sec": (fn_sec, "Sec(x)", "Secant 1 / cos(x).", "Trig"),
    "cosec": (fn_cosec, "CoSec(x)", "Cosecant 1 / sin(x).", "Trig"),
    "arcsin": (fn_arcsin, "ArcSin(x)", "Inverse sine in radians (domain [-1, 1]).", "Trig"),
    "arccos": (fn_arccos, "ArcCos(x)", "Inverse cosine in radians (domain [-1, 1]).", "Trig"),
    "arctan": (fn_arctan, "ArcTg(x)", "Inverse tangent in radians.", "Trig"),
    "atan2": (fn_atan2, "Atan2(y, x)", "Two-argument arctangent.", "Trig"),
    "sinh": (fn_sinh, "SinH(x)", "Hyperbolic sine.", "Hyperbolic"),
    "cosh": (fn_cosh, "CosH(x)", "Hyperbolic cosine.", "Hyperbolic"),
    "tanh": (fn_tanh, "TanH(x)", "Hyperbolic tangent — classic soft-clipper.", "Hyperbolic"),
    "exp": (fn_exp, "Exp(x)", "Exponential e^x.", "Exp / Log"),
    "ln": (fn_ln, "Ln(x)", "Natural logarithm (base e).", "Exp / Log"),
    "log2": (fn_log2, "Log2(x)", "Base-2 logarithm.", "Exp / Log"),
    "log10": (fn_log10, "Log10(x)", "Base-10 logarithm.", "Exp / Log"),
    "sqrt": (fn_sqrt, "Sqrt(x)", "Square root (domain x >= 0).", "Exp / Log"),
    "sqr": (fn_sqr, "Sqr(x)", "Square x^2.", "Exp / Log"),
    "pow": (fn_pow, "Pow(x, y)", "Raises x to the power y.", "Exp / Log"),
    "abs": (fn_abs, "Abs(x)", "Absolute value |x|.", "Rounding / Sign"),
    "neg": (fn_neg, "Neg(x)", "Negates x (-x).", "Rounding / Sign"),
    "sign": (fn_sign, "Sign(x)", "Sign of x (-1, 0, or 1).", "Rounding / Sign"),
    "int": (fn_int, "Int(x)", "Integer part (truncates toward zero).", "Rounding / Sign"),
    "frac": (fn_frac, "Frac(x)", "Fractional part x - Int(x). Great for sawtooth waves.", "Rounding / Sign"),
    "floor": (fn_floor, "Floor(x)", "Largest integer <= x.", "Rounding / Sign"),
    "ceil": (fn_ceil, "Ceil(x)", "Smallest integer >= x.", "Rounding / Sign"),
    "round": (fn_round, "Round(x)", "Rounds x to nearest integer.", "Rounding / Sign"),
    "min": (fn_min, "Min(a, b)", "Smaller of two values.", "Range / Shaping"),
    "max": (fn_max, "Max(a, b)", "Larger of two values.", "Range / Shaping"),
    "clamp": (fn_clamp, "Clamp(x, lo, hi)", "Clamps x into [lo, hi].", "Range / Shaping"),
    "saturate": (fn_saturate, "Saturate(x)", "Clamps x into [0, 1].", "Range / Shaping"),
    "mod": (fn_mod, "Mod(a, b)", "Modulo remainder.", "Range / Shaping"),
    "fmod": (fn_fmod, "FMod(a, b)", "C-style floating-point remainder.", "Range / Shaping"),
    "mix": (fn_mix, "Mix(a, b, t)", "Linear interpolation a + (b - a) * t.", "Range / Shaping"),
    "step": (fn_step, "Step(edge, x)", "0 if x < edge else 1.", "Range / Shaping"),
    "smoothstep": (fn_smoothstep, "SmoothStep(e0, e1, x)", "Hermite smooth interpolation.", "Range / Shaping"),
    "tension": (fn_tension, "Tension(val, rate)", "FL Studio curve bend (+rate concave, -rate convex, 0 linear).", "FL Curves"),
    "case": (fn_case, "Case(cond, t, f)", "Returns t if cond != 0 else f.", "FL Logic"),
    "ife": (fn_ife, "IfE(a, b)", "Returns 1 if a == b else 0.", "FL Logic"),
    "ifl": (fn_ifl, "IfL(a, b)", "Returns 1 if a < b else 0.", "FL Logic"),
    "ifle": (fn_ifle, "IfLE(a, b)", "Returns 1 if a <= b else 0.", "FL Logic"),
    "ifg": (fn_ifg, "IfG(a, b)", "Returns 1 if a > b else 0.", "FL Logic"),
    "ifge": (fn_ifge, "IfGE(a, b)", "Returns 1 if a >= b else 0.", "FL Logic"),
    "inter": (fn_inter, "Inter(x, a, b)", "Returns 1 if x is strictly between a and b (auto-swaps).", "FL Logic"),
    "internoswap": (fn_internoswap, "InterNoSwap(x, a, b)", "Returns 1 if a < x < b (no swap).", "FL Logic"),
    "seededrand": (fn_seeded_rand, "SeededRand(seed)", "Deterministic pseudo-random value in [0, 1) from seed.", "FL Random"),
    "where": (fn_where, "where(cond, t, f)", "Vectorized ternary selection.", "Logic"),
}


@dataclass
class FunctionSpec:
    name: str
    canonical_id: str
    func: Callable[..., Any]
    signature: str
    doc: str
    category: str
    fl_name: Optional[str] = None  # Canonical FL Studio name for export


@dataclass
class Dialect:
    id: str
    name: str
    description: str
    case_insensitive: bool = True
    caret_is_power: bool = True
    single_eq_is_comparison: bool = True
    word_operators: Set[str] = field(default_factory=set)
    constants: Dict[str, float] = field(default_factory=dict)
    functions: Dict[str, FunctionSpec] = field(default_factory=dict)
    special_variables: Dict[str, str] = field(default_factory=dict)

    def normalize_key(self, name: str) -> str:
        return name.lower() if self.case_insensitive else name

    def lookup_function(self, name: str) -> Optional[FunctionSpec]:
        return self.functions.get(self.normalize_key(name))

    def lookup_constant(self, name: str) -> Optional[float]:
        return self.constants.get(self.normalize_key(name))

    def is_special_variable(self, name: str) -> bool:
        return self.normalize_key(name) in self.special_variables

    def is_known_symbol(self, name: str) -> bool:
        k = self.normalize_key(name)
        return (
            k in self.functions
            or k in self.constants
            or k in self.special_variables
            or k in self.word_operators
        )


def _add_fn(
    table: Dict[str, FunctionSpec],
    names: List[str],
    canonical_id: str,
    case_insensitive: bool,
    fl_name: Optional[str] = None,
    sig_override: Optional[str] = None,
    doc_override: Optional[str] = None,
) -> None:
    fn, sig, doc, cat = CORE_FUNCTIONS[canonical_id]
    primary = names[0]
    spec = FunctionSpec(
        name=primary,
        canonical_id=canonical_id,
        func=fn,
        signature=sig_override or sig,
        doc=doc_override or doc,
        category=cat,
        fl_name=fl_name,
    )
    for n in names:
        key = n.lower() if case_insensitive else n
        table[key] = spec


def build_fl_studio_dialect() -> Dialect:
    fns: Dict[str, FunctionSpec] = {}
    ci = True
    _add_fn(fns, ["Sin"], "sin", ci, "Sin")
    _add_fn(fns, ["Cos"], "cos", ci, "Cos")
    _add_fn(fns, ["Tan", "Tg"], "tan", ci, "Tan", "Tan(x) / Tg(x)")
    _add_fn(fns, ["Ctg", "Cot"], "ctg", ci, "Ctg")
    _add_fn(fns, ["Sec"], "sec", ci, "Sec")
    _add_fn(fns, ["CoSec", "Csc"], "cosec", ci, "CoSec")
    _add_fn(fns, ["ArcSin", "Asin"], "arcsin", ci, "ArcSin")
    _add_fn(fns, ["ArcCos", "Acos"], "arccos", ci, "ArcCos")
    _add_fn(fns, ["ArcTg", "ArcTan", "Atan"], "arctan", ci, "ArcTg", "ArcTg(x) / ArcTan(x)")
    _add_fn(fns, ["Atan2"], "atan2", ci, "Atan2")
    _add_fn(fns, ["SinH"], "sinh", ci, "SinH")
    _add_fn(fns, ["CosH"], "cosh", ci, "CosH")
    _add_fn(fns, ["TanH"], "tanh", ci, "TanH")
    _add_fn(fns, ["Exp"], "exp", ci, "Exp")
    _add_fn(fns, ["Ln"], "ln", ci, "Ln")
    _add_fn(fns, ["Log2"], "log2", ci, "Log2")
    _add_fn(fns, ["Log10"], "log10", ci, "Log10")
    _add_fn(fns, ["Sqrt"], "sqrt", ci, "Sqrt")
    _add_fn(fns, ["Sqr"], "sqr", ci, "Sqr")
    _add_fn(fns, ["Pow", "Power"], "pow", ci, "Pow")
    _add_fn(fns, ["Abs"], "abs", ci, "Abs")
    _add_fn(fns, ["Neg"], "neg", ci, "Neg")
    _add_fn(fns, ["Sign"], "sign", ci, "Sign")
    _add_fn(fns, ["Int"], "int", ci, "Int")
    _add_fn(fns, ["Frac"], "frac", ci, "Frac")
    _add_fn(fns, ["Round"], "round", ci, "Round")
    _add_fn(fns, ["Floor"], "floor", ci, "Int")
    _add_fn(fns, ["Ceil"], "ceil", ci, None)
    _add_fn(fns, ["Min"], "min", ci, "Min")
    _add_fn(fns, ["Max"], "max", ci, "Max")
    _add_fn(fns, ["Tension"], "tension", ci, "Tension")
    _add_fn(fns, ["Case"], "case", ci, "Case")
    _add_fn(fns, ["IfE"], "ife", ci, "IfE")
    _add_fn(fns, ["IfL"], "ifl", ci, "IfL")
    _add_fn(fns, ["IfLE"], "ifle", ci, "IfLE")
    _add_fn(fns, ["IfG"], "ifg", ci, "IfG")
    _add_fn(fns, ["IfGE"], "ifge", ci, "IfGE")
    _add_fn(fns, ["Inter"], "inter", ci, "Inter")
    _add_fn(fns, ["InterNoSwap"], "internoswap", ci, "InterNoSwap")
    _add_fn(fns, ["SeededRand"], "seededrand", ci, "SeededRand")

    return Dialect(
        id="fl_studio",
        name="FL Studio",
        description="FL Studio Mapping Formula, Fruity Formula Controller & X-Y-Z Controller syntax (case-insensitive).",
        case_insensitive=True,
        caret_is_power=True,
        single_eq_is_comparison=True,
        word_operators={"and", "or", "xor", "not", "mod", "div", "shl", "shr"},
        constants={
            "pi": math.pi,
            "e": math.e,
            "true": 1.0,
            "false": 0.0,
        },
        functions=fns,
        special_variables={
            "input": "Controller / automation input value (0.0 to 1.0).",
            "x": "Primary X-axis variable (alias for Input in Mapping/Transfer modes).",
            "a": "Fruity Formula Controller Knob A (0.0 to 1.0).",
            "b": "Fruity Formula Controller Knob B (0.0 to 1.0).",
            "c": "Fruity Formula Controller Knob C (0.0 to 1.0).",
            "songtime": "Current song position in quarter notes (beats).",
            "time": "Elapsed time in milliseconds (derived from SongTime & BPM).",
            "date": "Simulated time-of-day fraction (0.0 to 1.0).",
            "mousex": "Simulated Mouse X (0.0 to 1.0).",
            "mousey": "Simulated Mouse Y (0.0 to 1.0).",
            "rand": "Random number in [0.0, 1.0) — press Reseed to refresh.",
            "gettempo": "Current project tempo in BPM.",
        },
    )


def build_python_dialect() -> Dialect:
    fns: Dict[str, FunctionSpec] = {}
    ci = False
    _add_fn(fns, ["sin"], "sin", ci, "Sin", "sin(x)")
    _add_fn(fns, ["cos"], "cos", ci, "Cos", "cos(x)")
    _add_fn(fns, ["tan"], "tan", ci, "Tan", "tan(x)")
    _add_fn(fns, ["asin", "arcsin"], "arcsin", ci, "ArcSin", "asin(x)")
    _add_fn(fns, ["acos", "arccos"], "arccos", ci, "ArcCos", "acos(x)")
    _add_fn(fns, ["atan", "arctan"], "arctan", ci, "ArcTg", "atan(x)")
    _add_fn(fns, ["atan2", "arctan2"], "atan2", ci, "Atan2", "atan2(y, x)")
    _add_fn(fns, ["sinh"], "sinh", ci, "SinH", "sinh(x)")
    _add_fn(fns, ["cosh"], "cosh", ci, "CosH", "cosh(x)")
    _add_fn(fns, ["tanh"], "tanh", ci, "TanH", "tanh(x)")
    _add_fn(fns, ["exp"], "exp", ci, "Exp", "exp(x)")
    _add_fn(fns, ["log", "ln"], "ln", ci, "Ln", "log(x)")
    _add_fn(fns, ["log2"], "log2", ci, "Log2", "log2(x)")
    _add_fn(fns, ["log10"], "log10", ci, "Log10", "log10(x)")
    _add_fn(fns, ["sqrt"], "sqrt", ci, "Sqrt", "sqrt(x)")
    _add_fn(fns, ["pow"], "pow", ci, "Pow", "pow(x, y)")
    _add_fn(fns, ["abs", "fabs"], "abs", ci, "Abs", "abs(x)")
    _add_fn(fns, ["sign"], "sign", ci, "Sign", "sign(x)")
    _add_fn(fns, ["floor"], "floor", ci, "Int", "floor(x)")
    _add_fn(fns, ["ceil"], "ceil", ci, None, "ceil(x)")
    _add_fn(fns, ["trunc", "int"], "int", ci, "Int", "trunc(x)")
    _add_fn(fns, ["round"], "round", ci, "Round", "round(x)")
    _add_fn(fns, ["min", "minimum"], "min", ci, "Min", "min(a, b)")
    _add_fn(fns, ["max", "maximum"], "max", ci, "Max", "max(a, b)")
    _add_fn(fns, ["clip", "clamp"], "clamp", ci, None, "clip(x, lo, hi)")
    _add_fn(fns, ["where"], "where", ci, "Case", "where(cond, a, b)")
    _add_fn(fns, ["mod"], "mod", ci, None, "mod(a, b)")
    _add_fn(fns, ["tension"], "tension", ci, "Tension", "tension(x, rate)")

    return Dialect(
        id="python",
        name="Python / NumPy",
        description="Standard Python / NumPy mathematical expressions (** power, == equality, lowercase functions).",
        case_insensitive=False,
        caret_is_power=True,  # convenient in math tools; ** also supported
        single_eq_is_comparison=False,
        word_operators={"and", "or", "not"},
        constants={
            "pi": math.pi,
            "e": math.e,
            "tau": math.tau,
            "True": 1.0,
            "False": 0.0,
        },
        functions=fns,
        special_variables={
            "x": "Primary X-axis variable.",
            "Input": "Input variable alias.",
            "t": "Time axis alias (SongTime in Formula Controller mode).",
            "SongTime": "Song position in beats.",
        },
    )


def build_dsp_glsl_dialect() -> Dialect:
    fns: Dict[str, FunctionSpec] = {}
    ci = False
    _add_fn(fns, ["sin"], "sin", ci, "Sin", "sin(x)")
    _add_fn(fns, ["cos"], "cos", ci, "Cos", "cos(x)")
    _add_fn(fns, ["tan"], "tan", ci, "Tan", "tan(x)")
    _add_fn(fns, ["asin"], "arcsin", ci, "ArcSin", "asin(x)")
    _add_fn(fns, ["acos"], "arccos", ci, "ArcCos", "acos(x)")
    _add_fn(fns, ["atan"], "arctan", ci, "ArcTg", "atan(x)")
    _add_fn(fns, ["atan2"], "atan2", ci, "Atan2", "atan2(y, x)")
    _add_fn(fns, ["sinh"], "sinh", ci, "SinH", "sinh(x)")
    _add_fn(fns, ["cosh"], "cosh", ci, "CosH", "cosh(x)")
    _add_fn(fns, ["tanh"], "tanh", ci, "TanH", "tanh(x)")
    _add_fn(fns, ["exp"], "exp", ci, "Exp", "exp(x)")
    _add_fn(fns, ["log"], "ln", ci, "Ln", "log(x)")
    _add_fn(fns, ["log2"], "log2", ci, "Log2", "log2(x)")
    _add_fn(fns, ["log10"], "log10", ci, "Log10", "log10(x)")
    _add_fn(fns, ["sqrt"], "sqrt", ci, "Sqrt", "sqrt(x)")
    _add_fn(fns, ["pow"], "pow", ci, "Pow", "pow(x, y)")
    _add_fn(fns, ["abs", "fabs"], "abs", ci, "Abs", "abs(x)")
    _add_fn(fns, ["sign"], "sign", ci, "Sign", "sign(x)")
    _add_fn(fns, ["floor"], "floor", ci, "Int", "floor(x)")
    _add_fn(fns, ["ceil"], "ceil", ci, None, "ceil(x)")
    _add_fn(fns, ["trunc"], "int", ci, "Int", "trunc(x)")
    _add_fn(fns, ["fract", "frac"], "frac", ci, "Frac", "fract(x)")
    _add_fn(fns, ["round"], "round", ci, "Round", "round(x)")
    _add_fn(fns, ["min"], "min", ci, "Min", "min(a, b)")
    _add_fn(fns, ["max"], "max", ci, "Max", "max(a, b)")
    _add_fn(fns, ["clamp"], "clamp", ci, None, "clamp(x, lo, hi)")
    _add_fn(fns, ["saturate"], "saturate", ci, None, "saturate(x)")
    _add_fn(fns, ["mix", "lerp"], "mix", ci, None, "mix(a, b, t) / lerp(a, b, t)")
    _add_fn(fns, ["step"], "step", ci, None, "step(edge, x)")
    _add_fn(fns, ["smoothstep"], "smoothstep", ci, None, "smoothstep(e0, e1, x)")
    _add_fn(fns, ["mod"], "mod", ci, None, "mod(a, b)")
    _add_fn(fns, ["fmod"], "fmod", ci, None, "fmod(a, b)")

    return Dialect(
        id="dsp_glsl",
        name="DSP / GLSL / C",
        description="Shader & C/C++ DSP waveshaper syntax (mix/lerp, clamp, saturate, smoothstep, fract, tanh).",
        case_insensitive=False,
        caret_is_power=True,
        single_eq_is_comparison=False,
        word_operators=set(),
        constants={
            "PI": math.pi,
            "pi": math.pi,
            "TAU": math.tau,
            "E": math.e,
        },
        functions=fns,
        special_variables={
            "x": "Input sample (-1.0 to 1.0 in Transfer mode, or 0.0 to 1.0).",
            "in": "Input sample alias.",
            "Input": "Input sample alias.",
            "t": "Time / phase variable.",
        },
    )


class DialectRegistry:
    """Holds built-in dialects and loads user-defined JSON dialect dictionaries."""

    def __init__(self, extra_dir: Optional[Path] = None) -> None:
        self._dialects: Dict[str, Dialect] = {}
        for d in (
            build_fl_studio_dialect(),
            build_python_dialect(),
            build_dsp_glsl_dialect(),
        ):
            self._dialects[d.id] = d

        if extra_dir and extra_dir.is_dir():
            self.load_json_dir(extra_dir)

    def load_json_dir(self, directory: Path) -> List[str]:
        loaded: List[str] = []
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                dialect = self._dialect_from_dict(data)
                if dialect:
                    self._dialects[dialect.id] = dialect
                    loaded.append(dialect.id)
            except Exception:
                # Ignore malformed optional custom dialect files rather than crashing startup
                continue
        return loaded

    def _dialect_from_dict(self, data: Dict[str, Any]) -> Optional[Dialect]:
        did = data.get("id")
        name = data.get("name")
        if not did or not name:
            return None
        ci = bool(data.get("case_insensitive", True))
        fns: Dict[str, FunctionSpec] = {}
        # Optional inheritance from an existing dialect
        base_id = data.get("extends")
        if base_id and base_id in self._dialects:
            base = self._dialects[base_id]
            fns.update(base.functions)
            constants = dict(base.constants)
            special_vars = dict(base.special_variables)
            word_ops = set(base.word_operators)
        else:
            constants = {}
            special_vars = {}
            word_ops = set()

        for k, v in data.get("constants", {}).items():
            constants[k.lower() if ci else k] = float(v)
        for k, v in data.get("special_variables", {}).items():
            special_vars[k.lower() if ci else k] = str(v)
        for op in data.get("word_operators", []):
            word_ops.add(op.lower())
        for alias, target in data.get("function_aliases", {}).items():
            if target in CORE_FUNCTIONS:
                _add_fn(fns, [alias], target, ci, sig_override=f"{alias}(...)")

        return Dialect(
            id=did,
            name=name,
            description=str(data.get("description", "")),
            case_insensitive=ci,
            caret_is_power=bool(data.get("caret_is_power", True)),
            single_eq_is_comparison=bool(data.get("single_eq_is_comparison", False)),
            word_operators=word_ops,
            constants=constants,
            functions=fns,
            special_variables=special_vars,
        )

    def get(self, dialect_id: str) -> Dialect:
        return self._dialects[dialect_id]

    def list_dialects(self) -> List[Dialect]:
        return list(self._dialects.values())
