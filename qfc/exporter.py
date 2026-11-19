"""AST-to-FL-Studio transpiler, parameter baker, and auto-normalizer."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from qfc.dialects import Dialect
from qfc.parser import (
    ASTNode,
    BinOpNode,
    CallNode,
    IdentNode,
    NumberNode,
    UnaryOpNode,
)


def format_number(val: float) -> str:
    """Format a float cleanly without noisy trailing zeros or scientific artifacts."""
    if abs(val) < 1e-12:
        return "0"
    if abs(val - round(val)) < 1e-9 and abs(val) < 1e9:
        return str(int(round(val)))
    text = f"{val:.6g}"
    return text


_PREC = {
    "or": 1,
    "xor": 2,
    "and": 3,
    "==": 4,
    "!=": 4,
    "<": 4,
    "<=": 4,
    ">": 4,
    ">=": 4,
    "shl": 5,
    "shr": 5,
    "+": 6,
    "-": 6,
    "*": 7,
    "/": 7,
    "mod": 7,
    "div": 7,
    "^": 9,
}


def _fl_op(op: str) -> str:
    if op == "==":
        return "="
    if op == "!=":
        return "<>"
    return op


def _emit_fl_node(
    node: ASTNode,
    dialect: Dialect,
    bake_params: Dict[str, float],
    primary_var_target: Optional[str],
    warnings: List[str],
    parent_prec: int = 0,
    is_right_child: bool = False,
) -> str:
    if isinstance(node, NumberNode):
        return format_number(node.value)

    if isinstance(node, IdentNode):
        norm = dialect.normalize_key(node.name)
        if norm in bake_params:
            val = bake_params[norm]
            s = format_number(val)
            return f"({s})" if val < 0 and parent_prec >= 8 else s

        # Canonical FL Studio variable / constant casing
        fl_specials = {
            "pi": "Pi",
            "e": "Exp(1)",
            "tau": "(2*Pi)",
            "true": "1",
            "false": "0",
            "songtime": "SongTime",
            "time": "Time",
            "date": "Date",
            "mousex": "MouseX",
            "mousey": "MouseY",
            "rand": "Rand",
            "gettempo": "GetTempo()",
            "input": "Input",
            "a": "a",
            "b": "b",
            "c": "c",
        }
        if primary_var_target and norm in ("x", "in", "input"):
            return primary_var_target
        if norm in fl_specials:
            return fl_specials[norm]
        return node.name

    if isinstance(node, UnaryOpNode):
        inner = _emit_fl_node(node.operand, dialect, bake_params, primary_var_target, warnings, parent_prec=8)
        if node.op == "+":
            return inner
        if node.op == "-":
            res = f"-{inner}"
        elif node.op == "not":
            res = f"not ({inner})"
        else:
            res = f"{node.op}{inner}"
        return f"({res})" if parent_prec > 8 else res

    if isinstance(node, BinOpNode):
        prec = _PREC.get(node.op, 6)
        left_s = _emit_fl_node(node.left, dialect, bake_params, primary_var_target, warnings, parent_prec=prec)
        # Right child needs parens on equal precedence for non-associative ops (-, /, ^, mod, div)
        right_prec = prec + 1 if node.op in ("-", "/", "mod", "div", "==", "!=", "<", "<=", ">", ">=") else prec
        right_s = _emit_fl_node(node.right, dialect, bake_params, primary_var_target, warnings, parent_prec=right_prec, is_right_child=True)

        op_str = _fl_op(node.op)
        if op_str.isalpha():
            expr = f"{left_s} {op_str} {right_s}"
        elif op_str in ("^", "*", "/"):
            expr = f"{left_s}{op_str}{right_s}"
        else:
            expr = f"{left_s} {op_str} {right_s}"

        if prec < parent_prec or (is_right_child and prec == parent_prec and node.op in ("-", "/")):
            return f"({expr})"
        return expr

    if isinstance(node, CallNode):
        args_s = [
            _emit_fl_node(a, dialect, bake_params, primary_var_target, warnings, parent_prec=0)
            for a in node.args
        ]
        spec = dialect.lookup_function(node.name)
        if spec is None:
            return f"{node.name}({', '.join(args_s)})"

        cid = spec.canonical_id
        # Smart lowering for non-FL helper functions into pure FL Studio expressions!
        if cid == "clamp" and len(args_s) == 3:
            return f"Min({args_s[2]}, Max({args_s[1]}, {args_s[0]}))"
        if cid == "saturate" and len(args_s) == 1:
            return f"Min(1, Max(0, {args_s[0]}))"
        if cid == "mix" and len(args_s) == 3:
            return f"({args_s[0]} + ({args_s[1]} - {args_s[0]}) * ({args_s[2]}))"
        if cid == "step" and len(args_s) == 2:
            return f"IfGE({args_s[1]}, {args_s[0]})"
        if cid == "smoothstep" and len(args_s) == 3:
            t_expr = f"Min(1, Max(0, ({args_s[2]} - {args_s[0]}) / ({args_s[1]} - {args_s[0]})))"
            return f"(({t_expr})^2 * (3 - 2 * ({t_expr})))"
        if cid == "mod" and len(args_s) == 2:
            return f"({args_s[0]} mod {args_s[1]})"
        if cid in ("ife", "ifl", "ifle", "ifg", "ifge") and len(args_s) == 4:
            fl_fn = spec.fl_name or node.name
            return f"Case({fl_fn}({args_s[0]}, {args_s[1]}), {args_s[2]}, {args_s[3]})"
        if cid == "sign" and len(args_s) == 1:
            return f"(IfG({args_s[0]}, 0) - IfL({args_s[0]}, 0))"
        if cid == "ceil" and len(args_s) == 1:
            return f"(-Int(-({args_s[0]})))"
        if cid == "pow" and len(args_s) == 2:
            return f"(({args_s[0]})^({args_s[1]}))"

        if spec.fl_name:
            return f"{spec.fl_name}({', '.join(args_s)})"

        warnings.append(f"Function '{node.name}' has no direct FL Studio equivalent.")
        return f"{node.name}({', '.join(args_s)})"

    return ""


def export_to_fl_studio(
    ast: ASTNode,
    dialect: Dialect,
    param_values: Optional[Dict[str, float]] = None,
    bake_custom_params: bool = True,
    primary_var_target: Optional[str] = "Input",
    normalize_bounds: Optional[Tuple[float, float, float, float]] = None,
    clamp_range: Optional[Tuple[float, float]] = None,
) -> Tuple[str, List[str]]:
    """Convert an AST into a clean single-line FL Studio formula string.

    Args:
        ast: Parsed expression tree.
        dialect: Source dialect used to parse ``ast``.
        param_values: Mapping of custom slider names to numeric values.
        bake_custom_params: Replace custom variables with numeric literals.
        primary_var_target: Map ``x``/``Input`` to e.g. ``"Input"`` or ``"a"``.
        normalize_bounds: Optional ``(y_min, y_max, target_lo, target_hi)`` to wrap with auto-normalisation.
        clamp_range: Optional ``(lo, hi)`` to wrap with ``Min(hi, Max(lo, ...))``.
    """
    warnings: List[str] = []
    bake_map: Dict[str, float] = {}
    if bake_custom_params and param_values:
        for k, v in param_values.items():
            norm = dialect.normalize_key(k)
            # Never bake FL built-ins a, b, c, Input, SongTime
            if norm not in ("a", "b", "c", "input", "songtime", "time", "x"):
                bake_map[norm] = float(v)

    expr = _emit_fl_node(ast, dialect, bake_map, primary_var_target, warnings)

    if normalize_bounds is not None:
        y_min, y_max, t_lo, t_hi = normalize_bounds
        expr = wrap_normalized_formula(expr, y_min, y_max, t_lo, t_hi)

    if clamp_range is not None:
        lo_s = format_number(clamp_range[0])
        hi_s = format_number(clamp_range[1])
        expr = f"Min({hi_s}, Max({lo_s}, {expr}))"

    return expr, warnings


def wrap_normalized_formula(
    expr: str,
    y_min: float,
    y_max: float,
    target_lo: float = 0.0,
    target_hi: float = 1.0,
) -> str:
    """Wrap an expression so its [y_min, y_max] range maps cleanly into [target_lo, target_hi]."""
    span = y_max - y_min
    if not (span > 1e-12):
        return format_number((target_lo + target_hi) * 0.5)

    # Check if already normalized
    if abs(y_min - target_lo) < 1e-5 and abs(y_max - target_hi) < 1e-5:
        return expr

    # Shift to 0
    if abs(y_min) < 1e-6:
        shifted = f"({expr})"
    elif y_min > 0:
        shifted = f"(({expr}) - {format_number(y_min)})"
    else:
        shifted = f"(({expr}) + {format_number(-y_min)})"

    # Scale to [0, 1]
    if abs(span - 1.0) < 1e-6:
        unit = shifted
    else:
        unit = f"({shifted} / {format_number(span)})"

    # Map [0, 1] -> [target_lo, target_hi]
    t_span = target_hi - target_lo
    if abs(target_lo) < 1e-6 and abs(t_span - 1.0) < 1e-6:
        return unit
    if abs(target_lo) < 1e-6:
        return f"({unit} * {format_number(t_span)})"
    if target_lo < 0:
        return f"({unit} * {format_number(t_span)} - {format_number(-target_lo)})"
    return f"({unit} * {format_number(t_span)} + {format_number(target_lo)})"
