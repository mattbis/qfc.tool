"""Reusable PySide6 widgets: parameter sliders, FL a/b/c knobs, inspection badges."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from qfc.analysis import BoundsReport
from qfc.ui.theme import PALETTE


class ParameterSliderRow(QFrame):
    """Smooth high-resolution float slider with spinbox and editable min/max bounds."""

    valueChanged = Signal(str, float)

    _STEPS = 2000

    def __init__(
        self,
        name: str,
        value: float = 1.0,
        min_val: float = -10.0,
        max_val: float = 10.0,
        accent_color: str = PALETTE["accent"],
        locked_range: bool = False,
        default_reset: float = 1.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.param_name = name
        self._min = float(min_val)
        self._max = float(max_val)
        self._default_reset = float(default_reset)
        self._updating = False

        self.setObjectName("Card")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(6)

        self.lbl_name = QLabel(name)
        self.lbl_name.setFixedWidth(62)
        self.lbl_name.setStyleSheet(f"font-weight: 700; color: {accent_color}; background: transparent;")
        layout.addWidget(self.lbl_name)

        self.spin_min = QDoubleSpinBox()
        self.spin_min.setRange(-10000.0, 10000.0)
        self.spin_min.setDecimals(2)
        self.spin_min.setValue(self._min)
        self.spin_min.setFixedWidth(58)
        self.spin_min.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.spin_min.setToolTip("Slider minimum")
        self.spin_min.setVisible(not locked_range)
        layout.addWidget(self.spin_min)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._STEPS)
        layout.addWidget(self.slider, stretch=1)

        self.spin_max = QDoubleSpinBox()
        self.spin_max.setRange(-10000.0, 10000.0)
        self.spin_max.setDecimals(2)
        self.spin_max.setValue(self._max)
        self.spin_max.setFixedWidth(58)
        self.spin_max.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.spin_max.setToolTip("Slider maximum")
        self.spin_max.setVisible(not locked_range)
        layout.addWidget(self.spin_max)

        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(-100000.0, 100000.0)
        self.spin_val.setDecimals(3)
        self.spin_val.setSingleStep(0.05 if locked_range else 0.1)
        self.spin_val.setFixedWidth(74)
        self.spin_val.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.spin_val.setToolTip(f"Exact value for {name} (type or scroll wheel)")
        layout.addWidget(self.spin_val)

        self.btn_reset = QPushButton("↺")
        self.btn_reset.setObjectName("SmallBtn")
        self.btn_reset.setFixedWidth(26)
        self.btn_reset.setToolTip(f"Reset {name} to {self._default_reset:g}")
        layout.addWidget(self.btn_reset)

        self.slider.valueChanged.connect(self._on_slider_moved)
        self.spin_val.valueChanged.connect(self._on_spin_changed)
        self.spin_min.valueChanged.connect(self._on_bounds_changed)
        self.spin_max.valueChanged.connect(self._on_bounds_changed)
        self.btn_reset.clicked.connect(lambda: self.set_value(self._default_reset))

        self.set_value(value, emit=False)

    def value(self) -> float:
        return float(self.spin_val.value())

    def set_value(self, val: float, emit: bool = True) -> None:
        self._updating = True
        # Expand range smoothly if an initial value is outside unlocked range
        if val < self._min and self.spin_min.isVisible():
            self._min = min(val, self._min)
            self.spin_min.setValue(self._min)
        if val > self._max and self.spin_max.isVisible():
            self._max = max(val, self._max)
            self.spin_max.setValue(self._max)

        self.spin_val.setValue(val)
        self._sync_slider_from_val(val)
        self._updating = False
        if emit:
            self.valueChanged.emit(self.param_name, self.value())

    def _sync_slider_from_val(self, val: float) -> None:
        span = self._max - self._min
        if span <= 1e-12:
            pos = 0
        else:
            frac = max(0.0, min(1.0, (val - self._min) / span))
            pos = int(round(frac * self._STEPS))
        self.slider.setValue(pos)

    def _on_slider_moved(self, pos: int) -> None:
        if self._updating:
            return
        self._updating = True
        val = self._min + (pos / self._STEPS) * (self._max - self._min)
        self.spin_val.setValue(val)
        self._updating = False
        self.valueChanged.emit(self.param_name, val)

    def _on_spin_changed(self, val: float) -> None:
        if self._updating:
            return
        self._updating = True
        self._sync_slider_from_val(val)
        self._updating = False
        self.valueChanged.emit(self.param_name, float(val))

    def _on_bounds_changed(self) -> None:
        if self._updating:
            return
        lo = self.spin_min.value()
        hi = self.spin_max.value()
        if hi <= lo:
            hi = lo + 1.0
            self._updating = True
            self.spin_max.setValue(hi)
            self._updating = False
        self._min = lo
        self._max = hi
        self._updating = True
        self._sync_slider_from_val(self.spin_val.value())
        self._updating = False


class MetricBadge(QFrame):
    """Compact key-value status pill for the Inspection & Bounds bar."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(1)

        self.lbl_title = QLabel(title.upper())
        self.lbl_title.setStyleSheet(
            f"font-size: 10px; font-weight: 600; color: {PALETTE['text_dim']}; background: transparent;"
        )
        self.lbl_val = QLabel("—")
        self.lbl_val.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {PALETTE['text']}; background: transparent;"
        )
        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_val)

    def set_value(self, text: str, color: Optional[str] = None) -> None:
        c = color or PALETTE["text"]
        self.lbl_val.setText(text)
        self.lbl_val.setStyleSheet(f"font-size: 12px; font-weight: 700; color: {c}; background: transparent;")


class InspectionBar(QWidget):
    """Horizontal bar of live bounds, clipping, and curve-shape diagnostics."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.b_range = MetricBadge("Min / Max")
        self.b_endpoints = MetricBadge("Start → Mid → End")
        self.b_dc = MetricBadge("Mean / RMS")
        self.b_clip = MetricBadge("FL Bounds Check")
        self.b_health = MetricBadge("NaN / Inf")
        self.b_shape = MetricBadge("Shape")

        for b in (self.b_range, self.b_endpoints, self.b_dc, self.b_clip, self.b_health, self.b_shape):
            layout.addWidget(b, stretch=1)

    def update_report(self, r: BoundsReport) -> None:
        def _fmt(v: float) -> str:
            return "NaN" if v != v else f"{v:.4f}"

        self.b_range.set_value(f"{_fmt(r.y_min)}  ..  {_fmt(r.y_max)}")
        self.b_endpoints.set_value(f"{_fmt(r.start_val)} → {_fmt(r.mid_val)} → {_fmt(r.end_val)}")
        self.b_dc.set_value(f"{_fmt(r.y_mean)} / {_fmt(r.y_rms)}")

        bounds_str = f"[{r.target_lo:g}, {r.target_hi:g}]"
        if r.total_clip_pct <= 0.0:
            self.b_clip.set_value(f"OK in {bounds_str}", PALETTE["green"])
        else:
            parts = []
            if r.clip_low_pct > 0:
                parts.append(f"↓{r.clip_low_pct:.1f}%")
            if r.clip_high_pct > 0:
                parts.append(f"↑{r.clip_high_pct:.1f}%")
            self.b_clip.set_value(f"Clipped {' '.join(parts)}", PALETTE["red"])

        if r.nan_count == 0 and r.inf_count == 0:
            self.b_health.set_value("Clean (0)", PALETTE["green"])
        else:
            self.b_health.set_value(f"NaN:{r.nan_count} Inf:{r.inf_count}", PALETTE["red"])

        shape = r.monotonic
        if r.discontinuities > 0:
            shape += f" ({r.discontinuities} jump{'s' if r.discontinuities != 1 else ''})"
        self.b_shape.set_value(shape, PALETTE["cyan"])
