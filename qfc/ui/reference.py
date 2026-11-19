"""Searchable Dialect Reference drawer ('?') with double-click-to-insert."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from qfc.dialects import Dialect
from qfc.ui.theme import PALETTE


class ReferenceDrawer(QFrame):
    """Side panel listing all functions, variables, constants, and operators for the active dialect."""

    snippetSelected = Signal(str)

    def __init__(self, dialect: Dialect, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setMinimumWidth(250)
        self.setMaximumWidth(340)
        self._dialect = dialect

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.lbl_title = QLabel("Dialect Reference")
        self.lbl_title.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {PALETTE['accent']}; background: transparent;"
        )
        layout.addWidget(self.lbl_title)

        self.lbl_hint = QLabel("Double-click any item to insert into the editor.")
        self.lbl_hint.setStyleSheet(f"font-size: 11px; color: {PALETTE['text_dim']}; background: transparent;")
        self.lbl_hint.setWordWrap(True)
        layout.addWidget(self.lbl_hint)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter functions, vars, ops...")
        self.search_box.textChanged.connect(self._rebuild_tree)
        layout.addWidget(self.search_box)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Syntax", "Description"])
        self.tree.setColumnWidth(0, 130)
        self.tree.setRootIsDecorated(True)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.tree, stretch=1)

        self._rebuild_tree()

    def set_dialect(self, dialect: Dialect) -> None:
        self._dialect = dialect
        self.lbl_title.setText(f"{dialect.name} Reference")
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        query = self.search_box.text().strip().lower()
        self.tree.clear()
        d = self._dialect

        # Group functions by category, deduplicating aliases pointing to same FunctionSpec
        groups: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

        # Special variables
        for var, doc in d.special_variables.items():
            display = var if not d.case_insensitive else _pretty_var(var)
            groups["Variables"].append((display, doc, display))

        # Constants
        for c_name, c_val in d.constants.items():
            display = "Pi" if c_name.lower() == "pi" and d.case_insensitive else c_name
            groups["Constants"].append((display, f"= {c_val:.6g}", display))

        # Functions (deduplicated by canonical_id + name)
        seen_specs = set()
        for spec in d.functions.values():
            key = (spec.name, spec.canonical_id)
            if key in seen_specs:
                continue
            seen_specs.add(key)
            insert_txt = f"{spec.name}()"
            groups[spec.category].append((spec.signature, spec.doc, insert_txt))

        # Operators
        ops_list = [
            ("+ - * /", "Basic arithmetic", " + "),
            ("^", "Power (exponentiation)", "^"),
            ("= <> < <= > >=", "Comparisons (return 1 or 0)" if d.single_eq_is_comparison else "Comparisons", " < "),
        ]
        if d.word_operators:
            ops_list.append((" ".join(sorted(d.word_operators)), "Word logic / integer operators", " and "))
        groups["Operators"].extend(ops_list)

        for cat, items in groups.items():
            filtered = [
                (sig, doc, ins)
                for sig, doc, ins in items
                if not query or query in sig.lower() or query in doc.lower() or query in cat.lower()
            ]
            if not filtered:
                continue
            parent_item = QTreeWidgetItem([cat, ""])
            parent_item.setFlags(parent_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            for sig, doc, ins in filtered:
                child = QTreeWidgetItem([sig, doc])
                child.setToolTip(0, f"{sig}\n{doc}")
                child.setToolTip(1, doc)
                child.setData(0, Qt.ItemDataRole.UserRole, ins)
                parent_item.addChild(child)
            self.tree.addTopLevelItem(parent_item)
            parent_item.setExpanded(True)

    def _on_double_click(self, item: QTreeWidgetItem, _col: int) -> None:
        snippet = item.data(0, Qt.ItemDataRole.UserRole)
        if snippet:
            self.snippetSelected.emit(str(snippet))


def _pretty_var(v: str) -> str:
    mapping = {
        "input": "Input",
        "songtime": "SongTime",
        "time": "Time",
        "date": "Date",
        "mousex": "MouseX",
        "mousey": "MouseY",
        "rand": "Rand",
        "gettempo": "GetTempo()",
    }
    return mapping.get(v.lower(), v)
