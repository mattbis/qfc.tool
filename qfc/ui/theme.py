"""FL-Studio-inspired sleek dark theme palette and Qt stylesheet."""

from __future__ import annotations

PALETTE = {
    "bg_dark": "#14161b",
    "bg_panel": "#1c1f26",
    "bg_elevated": "#242833",
    "bg_input": "#12141a",
    "border": "#2e3442",
    "border_focus": "#ff8c32",
    "text": "#e6e9ef",
    "text_muted": "#9aa3b2",
    "text_dim": "#6b7485",
    "accent": "#ff8c32",       # FL warm orange
    "accent_hover": "#ffa052",
    "cyan": "#38bdf8",         # Secondary wave / reference
    "green": "#22c55e",        # Compiled OK / in-bounds
    "amber": "#f59e0b",        # Warning / raw curve
    "red": "#f43f5e",          # Error / clipping
    "purple": "#c084fc",       # Functions / keywords
}


DARK_QSS = f"""
QMainWindow, QWidget {{
    background-color: {PALETTE["bg_dark"]};
    color: {PALETTE["text"]};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}

QFrame#Card, QGroupBox {{
    background-color: {PALETTE["bg_panel"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 8px;
}}

QGroupBox {{
    margin-top: 12px;
    padding-top: 10px;
    font-weight: 600;
    color: {PALETTE["text_muted"]};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    background-color: {PALETTE["bg_panel"]};
    color: {PALETTE["text_muted"]};
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}

QPlainTextEdit, QLineEdit {{
    background-color: {PALETTE["bg_input"]};
    color: {PALETTE["text"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #ff8c3255;
}}

QPlainTextEdit:focus, QLineEdit:focus {{
    border: 1px solid {PALETTE["border_focus"]};
}}

QComboBox {{
    background-color: {PALETTE["bg_elevated"]};
    color: {PALETTE["text"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 24px;
}}

QDoubleSpinBox, QSpinBox {{
    background-color: {PALETTE["bg_elevated"]};
    color: {PALETTE["text"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 5px;
    padding: 2px 4px;
    min-height: 22px;
}}

QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{
    border-color: #464f63;
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

QComboBox QAbstractItemView {{
    background-color: {PALETTE["bg_elevated"]};
    color: {PALETTE["text"]};
    border: 1px solid {PALETTE["border"]};
    selection-background-color: {PALETTE["accent"]};
    selection-color: #111;
}}

QPushButton {{
    background-color: {PALETTE["bg_elevated"]};
    color: {PALETTE["text"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
    padding: 5px 12px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: #2d3240;
    border-color: #485266;
}}

QPushButton:pressed {{
    background-color: #1a1d24;
}}

QPushButton#PrimaryBtn {{
    background-color: {PALETTE["accent"]};
    color: #14161b;
    border: 1px solid {PALETTE["accent"]};
    font-weight: 700;
}}

QPushButton#PrimaryBtn:hover {{
    background-color: {PALETTE["accent_hover"]};
}}

QPushButton#ModeBtn {{
    background-color: {PALETTE["bg_panel"]};
    color: {PALETTE["text_muted"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
}}

QPushButton#ModeBtn:checked {{
    background-color: #ff8c3224;
    color: {PALETTE["accent"]};
    border: 1px solid {PALETTE["accent"]};
}}

QPushButton#SmallBtn {{
    padding: 2px 8px;
    font-size: 11px;
    border-radius: 4px;
}}

QCheckBox {{
    spacing: 6px;
    color: {PALETTE["text_muted"]};
}}

QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1px solid {PALETTE["border"]};
    background-color: {PALETTE["bg_input"]};
}}

QCheckBox::indicator:checked {{
    background-color: {PALETTE["accent"]};
    border-color: {PALETTE["accent"]};
}}

QSlider::groove:horizontal {{
    border: none;
    height: 6px;
    background: #2b303d;
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: {PALETTE["accent"]};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: #f1f5f9;
    border: 2px solid {PALETTE["accent"]};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: #ffffff;
    border-color: {PALETTE["accent_hover"]};
}}

QScrollArea {{
    border: none;
    background: transparent;
}}

QScrollBar:vertical {{
    background: {PALETTE["bg_dark"]};
    width: 8px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: #353b4a;
    border-radius: 4px;
    min-height: 24px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QSplitter::handle {{
    background-color: {PALETTE["border"]};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

QProgressBar {{
    background-color: {PALETTE["bg_input"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 4px;
    text-align: center;
    color: {PALETTE["text"]};
    font-size: 11px;
}}

QProgressBar::chunk {{
    background-color: {PALETTE["accent"]};
    border-radius: 3px;
}}

QTreeWidget, QListWidget {{
    background-color: {PALETTE["bg_input"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
    outline: none;
}}

QTreeWidget::item:selected, QListWidget::item:selected {{
    background-color: #ff8c3233;
    color: {PALETTE["text"]};
}}

QHeaderView::section {{
    background-color: {PALETTE["bg_panel"]};
    color: {PALETTE["text_muted"]};
    border: none;
    padding: 4px 6px;
    font-size: 11px;
}}
"""
