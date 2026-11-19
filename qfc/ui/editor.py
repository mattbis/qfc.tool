"""Multi-line dialect-aware formula editor with syntax highlighting & error underlines."""

from __future__ import annotations

import re
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from PySide6.QtWidgets import QPlainTextEdit, QTextEdit

from qfc.dialects import Dialect
from qfc.ui.theme import PALETTE


class FormulaHighlighter(QSyntaxHighlighter):
    """Highlights functions, FL variables, constants, numbers, operators, and comments."""

    _TOKEN_RE = re.compile(
        r"(?P<comment>//[^\n]*|#[^\n]*)"
        r"|(?P<number>\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"
        r"|(?P<ident>\b[A-Za-z_][A-Za-z0-9_]*\b)"
        r"|(?P<op>\*\*|<=|>=|<>|!=|==|&&|\|\||[+\-*/^%<>=!])"
    )

    def __init__(self, document: QTextDocument, dialect: Dialect) -> None:
        super().__init__(document)
        self._dialect = dialect

        self.fmt_comment = self._make_fmt("#64748b", italic=True)
        self.fmt_number = self._make_fmt("#fbbf24")
        self.fmt_function = self._make_fmt("#c084fc", bold=True)
        self.fmt_special = self._make_fmt("#38bdf8", bold=True)
        self.fmt_const = self._make_fmt("#4ade80", bold=True)
        self.fmt_keyword = self._make_fmt("#fb7185", bold=True)
        self.fmt_param = self._make_fmt("#ff8c32")
        self.fmt_op = self._make_fmt("#94a3b8")

    @staticmethod
    def _make_fmt(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        if bold:
            fmt.setFontWeight(QFont.Weight.Bold)
        if italic:
            fmt.setFontItalic(True)
        return fmt

    def set_dialect(self, dialect: Dialect) -> None:
        self._dialect = dialect
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:
        d = self._dialect
        for m in self._TOKEN_RE.finditer(text):
            start, end = m.start(), m.end()
            length = end - start
            kind = m.lastgroup
            val = m.group()

            if kind == "comment":
                self.setFormat(start, length, self.fmt_comment)
            elif kind == "number":
                self.setFormat(start, length, self.fmt_number)
            elif kind == "op":
                self.setFormat(start, length, self.fmt_op)
            elif kind == "ident":
                low = val.lower()
                if low in d.word_operators:
                    self.setFormat(start, length, self.fmt_keyword)
                elif d.lookup_function(val) is not None:
                    self.setFormat(start, length, self.fmt_function)
                elif d.lookup_constant(val) is not None:
                    self.setFormat(start, length, self.fmt_const)
                elif d.is_special_variable(val):
                    self.setFormat(start, length, self.fmt_special)
                else:
                    self.setFormat(start, length, self.fmt_param)


class FormulaEditor(QPlainTextEdit):
    """Comfortable multi-line formula editor (replacing FL Studio's tiny box!)."""

    def __init__(self, dialect: Dialect, parent=None) -> None:
        super().__init__(parent)
        font = QFont("Cascadia Code", 11)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 2)
        self.setPlaceholderText(
            "Enter FL Studio / DSP formula (multi-line & // comments supported)...\n"
            "e.g. Tension(Input, 25)  or  a + b * 0.5 * Sin(SongTime * Pi * 2)"
        )
        self.setMinimumHeight(88)
        self.highlighter = FormulaHighlighter(self.document(), dialect)

    def set_dialect(self, dialect: Dialect) -> None:
        self.highlighter.set_dialect(dialect)

    def set_error_span(self, start: int, end: int) -> None:
        if start < 0:
            self.setExtraSelections([])
            return
        doc_len = len(self.toPlainText())
        s = max(0, min(start, doc_len))
        e = max(s + 1, min(end, doc_len)) if doc_len > 0 else 0

        sel = QTextEdit.ExtraSelection()
        sel.format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)
        sel.format.setUnderlineColor(QColor(PALETTE["red"]))
        sel.format.setBackground(QColor("#f43f5e26"))
        cursor = self.textCursor()
        cursor.setPosition(s)
        cursor.setPosition(e, QTextCursor.MoveMode.KeepAnchor)
        sel.cursor = cursor
        self.setExtraSelections([sel])

    def clear_error_span(self) -> None:
        self.setExtraSelections([])

    def insert_snippet(self, snippet: str) -> None:
        cursor = self.textCursor()
        selected = cursor.selectedText()
        if selected and "(" in snippet and ")" in snippet:
            # Wrap selection inside function call if user highlighted something
            fn_prefix = snippet.split("(", 1)[0]
            cursor.insertText(f"{fn_prefix}({selected})")
        else:
            cursor.insertText(snippet)
            # Place cursor inside parentheses if empty parens
            if snippet.endswith("()") or "(" in snippet:
                idx = snippet.find("(")
                if idx != -1:
                    move_back = len(snippet) - idx - 1
                    cursor.movePosition(
                        QTextCursor.MoveOperation.Left,
                        QTextCursor.MoveMode.MoveAnchor,
                        move_back,
                    )
        self.setTextCursor(cursor)
        self.setFocus(Qt.FocusReason.OtherFocusReason)
