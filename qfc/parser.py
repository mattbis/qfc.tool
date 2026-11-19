"""Safe AST lexer, recursive-descent parser, and vectorized NumPy evaluator.

Supports FL Studio Formula Controller / Mapping Formula syntax as well as
Python/NumPy and DSP/GLSL dialects without using unsafe ``eval()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from qfc.dialects import Dialect


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class FormulaError(Exception):
    """Syntax or evaluation error with optional source span."""

    def __init__(self, message: str, start: int = -1, end: int = -1) -> None:
        super().__init__(message)
        self.message = message
        self.start = start
        self.end = end


# ---------------------------------------------------------------------------
# Tokens & Lexer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    kind: str  # NUMBER, IDENT, OP, LPAREN, RPAREN, COMMA, EOF
    value: str
    start: int
    end: int


_TWO_CHAR_OPS = {"**", "<=", ">=", "<>", "!=", "==", "&&", "||", "//", "<<", ">>"}
_ONE_CHAR_OPS = {"+", "-", "*", "/", "^", "%", "=", "<", ">", "!"}


def tokenize(source: str, dialect: Dialect) -> List[Token]:
    tokens: List[Token] = []
    n = len(source)
    i = 0

    while i < n:
        ch = source[i]

        # Whitespace
        if ch.isspace():
            i += 1
            continue

        # Line comments: // or #
        if (ch == "/" and i + 1 < n and source[i + 1] == "/") or ch == "#":
            while i < n and source[i] != "\n":
                i += 1
            continue

        # Block comments: /* ... */
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            end_idx = source.find("*/", i + 2)
            if end_idx == -1:
                raise FormulaError("Unterminated block comment '/*'", i, n)
            i = end_idx + 2
            continue

        # Numbers (including .5 and scientific notation 1e-3)
        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            start = i
            has_dot = False
            while i < n and (source[i].isdigit() or (source[i] == "." and not has_dot)):
                if source[i] == ".":
                    has_dot = True
                i += 1
            if i < n and source[i] in ("e", "E"):
                exp_start = i
                i += 1
                if i < n and source[i] in ("+", "-"):
                    i += 1
                if i < n and source[i].isdigit():
                    while i < n and source[i].isdigit():
                        i += 1
                else:
                    i = exp_start
            tokens.append(Token("NUMBER", source[start:i], start, i))
            continue

        # Identifiers & word operators
        if ch.isalpha() or ch == "_":
            start = i
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
            text = source[start:i]
            low = text.lower()
            if low in dialect.word_operators:
                tokens.append(Token("OP", low, start, i))
            else:
                tokens.append(Token("IDENT", text, start, i))
            continue

        # Punctuation
        if ch == "(":
            tokens.append(Token("LPAREN", "(", i, i + 1))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token("RPAREN", ")", i, i + 1))
            i += 1
            continue
        if ch == ",":
            tokens.append(Token("COMMA", ",", i, i + 1))
            i += 1
            continue

        # Two-character operators
        if i + 1 < n and source[i : i + 2] in _TWO_CHAR_OPS:
            op = source[i : i + 2]
            tokens.append(Token("OP", op, i, i + 2))
            i += 2
            continue

        # Single-character operators
        if ch in _ONE_CHAR_OPS:
            tokens.append(Token("OP", ch, i, i + 1))
            i += 1
            continue

        raise FormulaError(f"Unexpected character '{ch}'", i, i + 1)

    tokens.append(Token("EOF", "", n, n))
    return tokens


# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumberNode:
    value: float
    raw: str
    start: int
    end: int


@dataclass(frozen=True)
class IdentNode:
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class UnaryOpNode:
    op: str
    operand: ASTNode
    start: int
    end: int


@dataclass(frozen=True)
class BinOpNode:
    op: str
    left: ASTNode
    right: ASTNode
    start: int
    end: int


@dataclass(frozen=True)
class CallNode:
    name: str
    args: Tuple[ASTNode, ...]
    start: int
    end: int


ASTNode = Union[NumberNode, IdentNode, UnaryOpNode, BinOpNode, CallNode]


# ---------------------------------------------------------------------------
# Recursive-Descent Parser
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, tokens: List[Token], dialect: Dialect) -> None:
        self.tokens = tokens
        self.dialect = dialect
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.kind != "EOF":
            self.pos += 1
        return tok

    def match_op(self, *ops: str) -> Optional[Token]:
        tok = self.peek()
        if tok.kind == "OP" and tok.value in ops:
            return self.advance()
        return None

    def parse(self) -> ASTNode:
        if self.peek().kind == "EOF":
            raise FormulaError("Formula is empty", 0, 0)
        node = self.parse_or()
        if self.peek().kind != "EOF":
            tok = self.peek()
            raise FormulaError(f"Unexpected token '{tok.value}'", tok.start, tok.end)
        return node

    # Precedence (lowest -> highest):
    # or / || -> xor -> and / && -> comparisons -> shifts -> add/sub -> mul/div/mod -> unary -> power -> primary

    def parse_or(self) -> ASTNode:
        node = self.parse_xor()
        while tok := self.match_op("or", "||"):
            right = self.parse_xor()
            node = BinOpNode("or", node, right, node.start, right.end)
        return node

    def parse_xor(self) -> ASTNode:
        node = self.parse_and()
        while tok := self.match_op("xor"):
            right = self.parse_and()
            node = BinOpNode("xor", node, right, node.start, right.end)
        return node

    def parse_and(self) -> ASTNode:
        node = self.parse_comparison()
        while tok := self.match_op("and", "&&"):
            right = self.parse_comparison()
            node = BinOpNode("and", node, right, node.start, right.end)
        return node

    def parse_comparison(self) -> ASTNode:
        node = self.parse_shift()
        comp_ops = ("==", "!=", "<>", "<", "<=", ">", ">=")
        if self.dialect.single_eq_is_comparison:
            comp_ops = comp_ops + ("=",)
        while tok := self.match_op(*comp_ops):
            right = self.parse_shift()
            op = tok.value
            if op == "=":
                op = "=="
            elif op == "<>":
                op = "!="
            node = BinOpNode(op, node, right, node.start, right.end)
        return node

    def parse_shift(self) -> ASTNode:
        node = self.parse_additive()
        while tok := self.match_op("shl", "shr", "<<", ">>"):
            right = self.parse_additive()
            op = "shl" if tok.value in ("shl", "<<") else "shr"
            node = BinOpNode(op, node, right, node.start, right.end)
        return node

    def parse_additive(self) -> ASTNode:
        node = self.parse_multiplicative()
        while tok := self.match_op("+", "-"):
            right = self.parse_multiplicative()
            node = BinOpNode(tok.value, node, right, node.start, right.end)
        return node

    def parse_multiplicative(self) -> ASTNode:
        node = self.parse_unary()
        while tok := self.match_op("*", "/", "%", "mod", "div", "//"):
            right = self.parse_unary()
            op = tok.value
            if op == "%":
                op = "mod"
            elif op == "//":
                op = "div"
            node = BinOpNode(op, node, right, node.start, right.end)
        return node

    def parse_unary(self) -> ASTNode:
        if tok := self.match_op("+", "-", "not", "!"):
            operand = self.parse_unary()
            op = "not" if tok.value in ("not", "!") else tok.value
            return UnaryOpNode(op, operand, tok.start, operand.end)
        return self.parse_power()

    def parse_power(self) -> ASTNode:
        node = self.parse_primary()
        if tok := self.match_op("^", "**"):
            # Right-associative; allows unary minus in exponent like 2^-x
            right = self.parse_unary()
            return BinOpNode("^", node, right, node.start, right.end)
        return node

    def parse_primary(self) -> ASTNode:
        tok = self.peek()

        if tok.kind == "NUMBER":
            self.advance()
            return NumberNode(float(tok.value), tok.value, tok.start, tok.end)

        if tok.kind == "IDENT":
            self.advance()
            # Function call?
            if self.peek().kind == "LPAREN":
                self.advance()  # consume '('
                args: List[ASTNode] = []
                if self.peek().kind != "RPAREN":
                    args.append(self.parse_or())
                    while self.peek().kind == "COMMA":
                        self.advance()
                        args.append(self.parse_or())
                if self.peek().kind != "RPAREN":
                    err_tok = self.peek()
                    raise FormulaError(
                        f"Expected ')' to close call to '{tok.value}'",
                        err_tok.start,
                        err_tok.end,
                    )
                rparen = self.advance()
                return CallNode(tok.value, tuple(args), tok.start, rparen.end)
            return IdentNode(tok.value, tok.start, tok.end)

        if tok.kind == "LPAREN":
            self.advance()
            expr = self.parse_or()
            if self.peek().kind != "RPAREN":
                err_tok = self.peek()
                raise FormulaError("Unmatched '(' — expected ')'", tok.start, err_tok.end)
            self.advance()
            return expr

        raise FormulaError(
            f"Expected a number, variable, or '(' but found '{tok.value or 'end of formula'}'",
            tok.start,
            tok.end,
        )


def parse_formula(source: str, dialect: Dialect) -> ASTNode:
    tokens = tokenize(source, dialect)
    return _Parser(tokens, dialect).parse()


# ---------------------------------------------------------------------------
# AST Variable Extraction & Validation
# ---------------------------------------------------------------------------

def collect_identifiers(node: ASTNode) -> List[str]:
    """Return all non-call variable identifiers in order of appearance."""
    seen: List[str] = []

    def walk(n: ASTNode) -> None:
        if isinstance(n, IdentNode):
            seen.append(n.name)
        elif isinstance(n, UnaryOpNode):
            walk(n.operand)
        elif isinstance(n, BinOpNode):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, CallNode):
            for a in n.args:
                walk(a)

    walk(node)
    return seen


def extract_custom_params(
    node: ASTNode,
    dialect: Dialect,
    reserved_vars: Optional[Set[str]] = None,
) -> List[str]:
    """Extract user slider parameter names (excluding dialect constants/specials)."""
    reserved_norm = {dialect.normalize_key(v) for v in (reserved_vars or set())}
    result: List[str] = []
    seen_norm: Set[str] = set()

    for ident in collect_identifiers(node):
        norm = dialect.normalize_key(ident)
        if norm in seen_norm:
            continue
        if norm in reserved_norm:
            continue
        if dialect.lookup_constant(ident) is not None:
            continue
        if dialect.is_special_variable(ident):
            continue
        seen_norm.add(norm)
        result.append(ident)
    return result


# ---------------------------------------------------------------------------
# Vectorized AST Evaluator
# ---------------------------------------------------------------------------

def evaluate_ast(
    node: ASTNode,
    dialect: Dialect,
    variables: Dict[str, Any],
    size: int = 1,
    rng_seed: int = 42,
    bpm: float = 128.0,
) -> np.ndarray:
    """Evaluate AST into a float64 NumPy array of shape ``(size,)``."""
    norm_vars = {dialect.normalize_key(k): v for k, v in variables.items()}
    rng = np.random.default_rng(rng_seed)

    def _eval(n: ASTNode) -> Any:
        if isinstance(n, NumberNode):
            return np.float64(n.value)

        if isinstance(n, IdentNode):
            key = dialect.normalize_key(n.name)
            # Special zero-arg dynamic functions in FL Studio (Rand, GetTempo)
            if key == "rand":
                return rng.random(size)
            if key == "gettempo":
                return np.float64(bpm)
            if key in norm_vars:
                return norm_vars[key]
            const = dialect.lookup_constant(n.name)
            if const is not None:
                return np.float64(const)
            raise FormulaError(f"Unknown variable '{n.name}'", n.start, n.end)

        if isinstance(n, UnaryOpNode):
            val = _eval(n.operand)
            if n.op == "+":
                return +np.asarray(val, dtype=np.float64)
            if n.op == "-":
                return -np.asarray(val, dtype=np.float64)
            if n.op == "not":
                return np.where(np.asarray(val, dtype=np.float64) == 0.0, 1.0, 0.0)
            raise FormulaError(f"Unknown unary operator '{n.op}'", n.start, n.end)

        if isinstance(n, BinOpNode):
            left = np.asarray(_eval(n.left), dtype=np.float64)
            right = np.asarray(_eval(n.right), dtype=np.float64)
            op = n.op
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                if op == "+":
                    return left + right
                if op == "-":
                    return left - right
                if op == "*":
                    return left * right
                if op == "/":
                    return left / right
                if op == "^":
                    return np.power(left, right)
                if op == "mod":
                    return np.mod(left, right)
                if op == "div":
                    return np.trunc(left / right)
                if op == "==":
                    return np.where(np.isclose(left, right, atol=1e-12), 1.0, 0.0)
                if op == "!=":
                    return np.where(~np.isclose(left, right, atol=1e-12), 1.0, 0.0)
                if op == "<":
                    return np.where(left < right, 1.0, 0.0)
                if op == "<=":
                    return np.where(left <= right, 1.0, 0.0)
                if op == ">":
                    return np.where(left > right, 1.0, 0.0)
                if op == ">=":
                    return np.where(left >= right, 1.0, 0.0)
                if op == "and":
                    return np.where((left != 0.0) & (right != 0.0), 1.0, 0.0)
                if op == "or":
                    return np.where((left != 0.0) | (right != 0.0), 1.0, 0.0)
                if op == "xor":
                    return np.where((left != 0.0) ^ (right != 0.0), 1.0, 0.0)
                if op == "shl":
                    return np.left_shift(left.astype(np.int64), np.clip(right.astype(np.int64), 0, 60)).astype(np.float64)
                if op == "shr":
                    return np.right_shift(left.astype(np.int64), np.clip(right.astype(np.int64), 0, 60)).astype(np.float64)
            raise FormulaError(f"Unknown binary operator '{op}'", n.start, n.end)

        if isinstance(n, CallNode):
            key = dialect.normalize_key(n.name)
            if key == "rand":
                if n.args:
                    raise FormulaError("Rand takes no arguments (use SeededRand(seed) for seeded random)", n.start, n.end)
                return rng.random(size)
            if key == "gettempo":
                if n.args:
                    raise FormulaError("GetTempo() takes no arguments", n.start, n.end)
                return np.float64(bpm)

            spec = dialect.lookup_function(n.name)
            if spec is None:
                raise FormulaError(
                    f"Unknown function '{n.name}' in dialect '{dialect.name}'",
                    n.start,
                    n.end,
                )
            eval_args = [_eval(a) for a in n.args]
            try:
                return spec.func(*eval_args)
            except TypeError as exc:
                raise FormulaError(
                    f"Wrong number of arguments for {spec.signature}: {exc}",
                    n.start,
                    n.end,
                ) from exc
            except ValueError as exc:
                raise FormulaError(str(exc), n.start, n.end) from exc

        raise FormulaError(f"Unsupported AST node: {type(n).__name__}")

    result = _eval(node)
    arr = np.asarray(result, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(size, float(arr), dtype=np.float64)
    elif arr.shape != (size,):
        arr = np.broadcast_to(arr, (size,)).astype(np.float64, copy=True)
    return arr
