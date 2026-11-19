"""Main PySide6 + PyQtGraph application window for Quick Formula Check."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from qfc.analysis import (
    BoundsReport,
    analyze_curve,
    compute_harmonic_spectrum,
    generate_test_waveform,
)
from qfc.dialects import Dialect, DialectRegistry
from qfc.exporter import export_to_fl_studio, wrap_normalized_formula
from qfc.parser import ASTNode, FormulaError, evaluate_ast, extract_custom_params, parse_formula
from qfc.presets import PRESETS, FormulaPreset
from qfc.ui.editor import FormulaEditor
from qfc.ui.reference import ReferenceDrawer
from qfc.ui.theme import PALETTE
from qfc.ui.widgets import InspectionBar, ParameterSliderRow


pg.setConfigOptions(antialias=True, background=PALETTE["bg_input"], foreground=PALETTE["text_muted"])


class MainWindow(QMainWindow):
    """FL Studio Formula & DSP Transfer Function Lab."""

    MODE_MAPPING = "mapping"
    MODE_CONTROLLER = "controller"
    MODE_TRANSFER = "transfer"

    def __init__(self, dialects_dir: Optional[Path] = None) -> None:
        super().__init__()
        self.setWindowTitle("QFC — FL Studio Formula & Transfer Function Lab")
        self.resize(1380, 860)

        self.registry = DialectRegistry(extra_dir=dialects_dir)
        self.current_dialect: Dialect = self.registry.get("fl_studio")
        self.current_mode: str = self.MODE_MAPPING

        self._ast: Optional[ASTNode] = None
        self._last_xs: Optional[np.ndarray] = None
        self._last_ys_raw: Optional[np.ndarray] = None
        self._last_ys_clipped: Optional[np.ndarray] = None
        self._last_report: Optional[BoundsReport] = None
        self._rng_seed: int = 42

        # Preserved parameter values & slider widgets
        self._param_values: Dict[str, float] = {}
        self._param_rows: Dict[str, ParameterSliderRow] = {}
        self._fl_knob_rows: Dict[str, ParameterSliderRow] = {}

        # Live Formula Controller playhead state
        self._playing: bool = False
        self._play_beat: float = 0.0
        self._last_tick_time: float = time.perf_counter()

        self._eval_debounce = QTimer(self)
        self._eval_debounce.setSingleShot(True)
        self._eval_debounce.setInterval(45)
        self._eval_debounce.timeout.connect(self._on_formula_changed)

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(16)  # ~60fps playhead
        self._play_timer.timeout.connect(self._on_play_tick)

        self._build_ui()
        self._setup_shortcuts()
        self._load_preset(PRESETS[0])

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # 1. Top Toolbar
        root.addWidget(self._build_toolbar())

        # 2. Main Horizontal Splitter: Left (Editor + Controls) | Center (Plots) | Right (Reference Drawer)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self.main_splitter, stretch=1)

        self.left_pane = self._build_left_pane()
        self.center_pane = self._build_center_pane()
        self.ref_drawer = ReferenceDrawer(self.current_dialect)
        self.ref_drawer.snippetSelected.connect(self.editor.insert_snippet)
        self.ref_drawer.setVisible(False)

        self.main_splitter.addWidget(self.left_pane)
        self.main_splitter.addWidget(self.center_pane)
        self.main_splitter.addWidget(self.ref_drawer)
        self.main_splitter.setStretchFactor(0, 4)
        self.main_splitter.setStretchFactor(1, 7)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([460, 860, 280])

    def _build_toolbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("Card")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        # Mode switcher buttons
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.btn_mode_map = QPushButton("FL Mapping [0..1]")
        self.btn_mode_ctrl = QPushButton("Formula Controller")
        self.btn_mode_xfer = QPushButton("Transfer / Waveshaper [-1..1]")

        for btn, mode in (
            (self.btn_mode_map, self.MODE_MAPPING),
            (self.btn_mode_ctrl, self.MODE_CONTROLLER),
            (self.btn_mode_xfer, self.MODE_TRANSFER),
        ):
            btn.setObjectName("ModeBtn")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked, m=mode: self._set_mode(m))
            self.mode_group.addButton(btn)
            layout.addWidget(btn)
        self.btn_mode_map.setChecked(True)

        layout.addSpacing(8)

        # Dialect selector
        lbl_d = QLabel("Dialect:")
        lbl_d.setStyleSheet(f"color: {PALETTE['text_muted']}; background: transparent;")
        layout.addWidget(lbl_d)
        self.combo_dialect = QComboBox()
        for d in self.registry.list_dialects():
            self.combo_dialect.addItem(d.name, d.id)
        self.combo_dialect.setMinimumWidth(145)
        self.combo_dialect.currentIndexChanged.connect(self._on_dialect_changed)
        layout.addWidget(self.combo_dialect)

        # Preset picker
        lbl_p = QLabel("Preset:")
        lbl_p.setStyleSheet(f"color: {PALETTE['text_muted']}; background: transparent;")
        layout.addWidget(lbl_p)
        self.combo_preset = QComboBox()
        self.combo_preset.setMinimumWidth(210)
        for i, p in enumerate(PRESETS):
            self.combo_preset.addItem(p.name, i)
        self.combo_preset.activated.connect(self._on_preset_selected)
        layout.addWidget(self.combo_preset)

        layout.addStretch(1)

        # Quick actions
        self.btn_reseed = QPushButton("Reseed Rand")
        self.btn_reseed.setToolTip("Generate a fresh seed for Rand")
        self.btn_reseed.clicked.connect(self._on_reseed)
        layout.addWidget(self.btn_reseed)

        self.btn_ref = QPushButton("? Reference")
        self.btn_ref.setCheckable(True)
        self.btn_ref.setToolTip("Toggle searchable dialect syntax reference (F1)")
        self.btn_ref.toggled.connect(lambda on: self.ref_drawer.setVisible(on))
        layout.addWidget(self.btn_ref)

        return bar

    def _build_left_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # --- Formula Editor Card ---
        editor_box = QGroupBox("Formula Editor  (multi-line && // comments supported)")
        ed_lay = QVBoxLayout(editor_box)
        ed_lay.setContentsMargins(10, 12, 10, 10)
        ed_lay.setSpacing(6)

        self.editor = FormulaEditor(self.current_dialect)
        self.editor.textChanged.connect(self._eval_debounce.start)
        ed_lay.addWidget(self.editor)

        status_row = QHBoxLayout()
        self.lbl_status = QLabel("Compiled OK")
        self.lbl_status.setStyleSheet(
            f"font-weight: 700; color: {PALETTE['green']}; background: transparent;"
        )
        self.lbl_status.setWordWrap(True)
        status_row.addWidget(self.lbl_status, stretch=1)

        self.btn_apply_norm = QPushButton("Auto-Normalise")
        self.btn_apply_norm.setObjectName("SmallBtn")
        self.btn_apply_norm.setToolTip("Wrap formula in editor so its min..max maps cleanly into target bounds")
        self.btn_apply_norm.clicked.connect(self._apply_auto_normalise)
        status_row.addWidget(self.btn_apply_norm)
        ed_lay.addLayout(status_row)

        layout.addWidget(editor_box)

        # --- FL Studio Export Strip ---
        export_box = QGroupBox("FL Studio Single-Line Export")
        ex_lay = QVBoxLayout(export_box)
        ex_lay.setContentsMargins(10, 12, 10, 10)
        ex_lay.setSpacing(6)

        self.export_preview = QLineEdit()
        self.export_preview.setReadOnly(True)
        self.export_preview.setPlaceholderText("Single-line FL Studio formula appears here...")
        ex_lay.addWidget(self.export_preview)

        opts_row = QHBoxLayout()
        self.chk_bake = QCheckBox("Bake sliders")
        self.chk_bake.setChecked(True)
        self.chk_bake.setToolTip("Replace custom slider variables with their numeric values for FL Studio")
        self.chk_bake.toggled.connect(self._update_export_preview)

        self.chk_export_norm = QCheckBox("Normalise")
        self.chk_export_norm.setToolTip("Auto-scale exported formula to target bounds")
        self.chk_export_norm.toggled.connect(self._update_export_preview)

        self.chk_export_clamp = QCheckBox("Clamp [0,1]")
        self.chk_export_clamp.setToolTip("Wrap exported formula in Min(1, Max(0, ...))")
        self.chk_export_clamp.toggled.connect(self._update_export_preview)

        opts_row.addWidget(self.chk_bake)
        opts_row.addWidget(self.chk_export_norm)
        opts_row.addWidget(self.chk_export_clamp)
        opts_row.addStretch(1)

        self.btn_copy_fl = QPushButton("Copy for FL")
        self.btn_copy_fl.setObjectName("PrimaryBtn")
        self.btn_copy_fl.setToolTip("Copy single-line FL Studio formula to clipboard (Ctrl+Shift+C)")
        self.btn_copy_fl.clicked.connect(self._copy_fl_formula)
        opts_row.addWidget(self.btn_copy_fl)
        ex_lay.addLayout(opts_row)

        self.lbl_export_warn = QLabel("")
        self.lbl_export_warn.setStyleSheet(
            f"font-size: 11px; color: {PALETTE['amber']}; background: transparent;"
        )
        self.lbl_export_warn.setWordWrap(True)
        self.lbl_export_warn.setVisible(False)
        ex_lay.addWidget(self.lbl_export_warn)

        layout.addWidget(export_box)

        # --- Fruity Formula Controller Transport & a/b/c Knobs (shown in Controller mode) ---
        self.ctrl_box = QGroupBox("Fruity Formula Controller  (a / b / c && SongTime)")
        ctrl_lay = QVBoxLayout(self.ctrl_box)
        ctrl_lay.setContentsMargins(10, 12, 10, 10)
        ctrl_lay.setSpacing(6)

        transport = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play LFO")
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._toggle_play)
        transport.addWidget(self.btn_play)

        lbl_bpm = QLabel("BPM:")
        lbl_bpm.setStyleSheet("background: transparent;")
        transport.addWidget(lbl_bpm)
        self.spin_bpm = QDoubleSpinBox()
        self.spin_bpm.setRange(20.0, 300.0)
        self.spin_bpm.setDecimals(1)
        self.spin_bpm.setValue(128.0)
        self.spin_bpm.setFixedWidth(72)
        self.spin_bpm.valueChanged.connect(self._evaluate_and_plot)
        transport.addWidget(self.spin_bpm)

        lbl_beats = QLabel("Window (beats):")
        lbl_beats.setStyleSheet("background: transparent;")
        transport.addWidget(lbl_beats)
        self.spin_beats = QDoubleSpinBox()
        self.spin_beats.setRange(0.25, 64.0)
        self.spin_beats.setDecimals(2)
        self.spin_beats.setValue(4.0)
        self.spin_beats.setFixedWidth(70)
        self.spin_beats.valueChanged.connect(self._evaluate_and_plot)
        transport.addWidget(self.spin_beats)
        transport.addStretch(1)
        ctrl_lay.addLayout(transport)

        # Live controller output bar
        meter_row = QHBoxLayout()
        lbl_out = QLabel("Live Out:")
        lbl_out.setStyleSheet(f"font-size: 11px; color: {PALETTE['text_muted']}; background: transparent;")
        meter_row.addWidget(lbl_out)
        self.live_meter = QProgressBar()
        self.live_meter.setRange(0, 1000)
        self.live_meter.setValue(500)
        self.live_meter.setFormat("0.500")
        self.live_meter.setFixedHeight(18)
        meter_row.addWidget(self.live_meter, stretch=1)
        ctrl_lay.addLayout(meter_row)

        # Dedicated FL knobs a, b, c locked to [0, 1]
        for knob_name in ("a", "b", "c"):
            row = ParameterSliderRow(
                name=f"Knob {knob_name}",
                value=0.5,
                min_val=0.0,
                max_val=1.0,
                accent_color=PALETTE["cyan"],
                locked_range=True,
                default_reset=0.5,
            )
            row.param_name = knob_name
            row.valueChanged.connect(self._on_param_changed)
            self._fl_knob_rows[knob_name] = row
            self._param_values[knob_name] = 0.5
            ctrl_lay.addWidget(row)

        self.ctrl_box.setVisible(False)
        layout.addWidget(self.ctrl_box)

        # --- Auto-Detected Custom Parameter Sliders ---
        params_box = QGroupBox("Formula Parameters  (auto-detected)")
        pb_lay = QVBoxLayout(params_box)
        pb_lay.setContentsMargins(6, 12, 6, 6)

        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setContentsMargins(2, 2, 2, 2)
        self.params_layout.setSpacing(6)
        self.params_layout.addStretch(1)
        self.params_scroll.setWidget(self.params_container)
        pb_lay.addWidget(self.params_scroll)

        self.lbl_no_params = QLabel("No custom variables detected — type any name (e.g. drive, freq) to spawn a slider.")
        self.lbl_no_params.setStyleSheet(
            f"font-size: 11px; color: {PALETTE['text_dim']}; padding: 6px; background: transparent;"
        )
        self.lbl_no_params.setWordWrap(True)
        pb_lay.addWidget(self.lbl_no_params)

        layout.addWidget(params_box, stretch=1)
        return pane

    def _build_center_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Plot Controls Strip
        ctrl_bar = QFrame()
        ctrl_bar.setObjectName("Card")
        cb_lay = QHBoxLayout(ctrl_bar)
        cb_lay.setContentsMargins(10, 6, 10, 6)
        cb_lay.setSpacing(10)

        self.chk_show_raw = QCheckBox("Raw Curve")
        self.chk_show_raw.setChecked(True)
        self.chk_show_raw.toggled.connect(self._update_plot_visibility)

        self.chk_show_clip = QCheckBox("FL Clipped Output")
        self.chk_show_clip.setChecked(True)
        self.chk_show_clip.toggled.connect(self._update_plot_visibility)

        self.chk_show_unity = QCheckBox("y = x Ref")
        self.chk_show_unity.setChecked(True)
        self.chk_show_unity.toggled.connect(self._update_plot_visibility)

        self.chk_show_wave = QCheckBox("Wave / FFT")
        self.chk_show_wave.setChecked(False)
        self.chk_show_wave.toggled.connect(self._on_wave_toggle)

        cb_lay.addWidget(self.chk_show_raw)
        cb_lay.addWidget(self.chk_show_clip)
        cb_lay.addWidget(self.chk_show_unity)
        cb_lay.addWidget(self.chk_show_wave)
        cb_lay.addStretch(1)

        self.lbl_cursor = QLabel("Hover to inspect")
        self.lbl_cursor.setStyleSheet(
            f"font-family: 'Cascadia Code', monospace; font-size: 12px; color: {PALETTE['text_muted']}; background: transparent;"
        )
        cb_lay.addWidget(self.lbl_cursor)

        self.btn_reset_view = QPushButton("Reset View")
        self.btn_reset_view.setObjectName("SmallBtn")
        self.btn_reset_view.clicked.connect(self._reset_plot_view)
        cb_lay.addWidget(self.btn_reset_view)

        layout.addWidget(ctrl_bar)

        # Vertical Splitter: Main Transfer/Formula Plot + Optional Waveshaper/FFT Preview Plot
        self.plot_splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(self.plot_splitter, stretch=1)

        # --- Primary PyQtGraph Plot ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.22)
        self.plot_widget.setLabel("bottom", "Input")
        self.plot_widget.setLabel("left", "Output")
        self.plot_widget.getAxis("bottom").setPen(PALETTE["border"])
        self.plot_widget.getAxis("left").setPen(PALETTE["border"])

        # Out-of-bounds danger shading (above target_hi and below target_lo)
        danger_brush = pg.mkBrush(244, 63, 94, 24)
        self.region_high = pg.LinearRegionItem(
            values=(1.0, 100.0), orientation="horizontal", brush=danger_brush, movable=False
        )
        self.region_low = pg.LinearRegionItem(
            values=(-100.0, 0.0), orientation="horizontal", brush=danger_brush, movable=False
        )
        for reg in (self.region_high, self.region_low):
            for line in reg.lines:
                line.setPen(pg.mkPen(None))
            reg.setZValue(-20)
            self.plot_widget.addItem(reg)

        # Bounds lines
        bound_pen = pg.mkPen("#f43f5e88", width=1, style=Qt.PenStyle.DashLine)
        self.line_hi = pg.InfiniteLine(pos=1.0, angle=0, pen=bound_pen)
        self.line_lo = pg.InfiniteLine(pos=0.0, angle=0, pen=bound_pen)
        self.line_zero_y = pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen("#47556966", width=1))
        self.line_zero_x = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#47556966", width=1))
        for ln in (self.line_hi, self.line_lo, self.line_zero_y, self.line_zero_x):
            ln.setZValue(-10)
            self.plot_widget.addItem(ln)

        # Unity diagonal y = x
        self.curve_unity = self.plot_widget.plot(
            pen=pg.mkPen("#38bdf844", width=1.4, style=Qt.PenStyle.DashLine)
        )
        self.curve_unity.setZValue(-5)

        # Raw mathematical curve (amber dashed)
        self.curve_raw = self.plot_widget.plot(
            pen=pg.mkPen("#f59e0baa", width=1.8, style=Qt.PenStyle.DashLine),
            connect="finite",
        )
        self.curve_raw.setZValue(5)

        # FL Clipped Output curve (bright glowing FL orange)
        self.curve_clipped = self.plot_widget.plot(
            pen=pg.mkPen(PALETTE["accent"], width=2.8),
            connect="finite",
        )
        self.curve_clipped.setZValue(10)

        # Live Formula Controller playhead vertical line + dot
        self.playhead_line = pg.InfiniteLine(
            pos=0.0, angle=90, pen=pg.mkPen(PALETTE["cyan"], width=1.6)
        )
        self.playhead_dot = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(PALETTE["cyan"]), pen=pg.mkPen("#ffffff", width=1.5)
        )
        self.playhead_line.setVisible(False)
        self.playhead_dot.setVisible(False)
        self.playhead_line.setZValue(15)
        self.playhead_dot.setZValue(16)
        self.plot_widget.addItem(self.playhead_line)
        self.plot_widget.addItem(self.playhead_dot)

        # Hover crosshair
        cross_pen = pg.mkPen("#94a3b866", width=1, style=Qt.PenStyle.DotLine)
        self.cross_v = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self.cross_h = pg.InfiniteLine(angle=0, movable=False, pen=cross_pen)
        self.cross_dot = pg.ScatterPlotItem(
            size=8, brush=pg.mkBrush(PALETTE["accent"]), pen=pg.mkPen("#ffffff", width=1.2)
        )
        for item in (self.cross_v, self.cross_h, self.cross_dot):
            item.setVisible(False)
            item.setZValue(20)
            self.plot_widget.addItem(item)

        self.plot_widget.scene().sigMouseMoved.connect(self._on_plot_mouse_moved)
        self.plot_splitter.addWidget(self.plot_widget)

        # --- Secondary Waveshaper / Harmonic Spectrum Panel ---
        self.wave_panel = self._build_wave_panel()
        self.wave_panel.setVisible(False)
        self.plot_splitter.addWidget(self.wave_panel)
        self.plot_splitter.setStretchFactor(0, 3)
        self.plot_splitter.setStretchFactor(1, 2)

        # Inspection & Bounds Diagnostics Bar
        self.inspection_bar = InspectionBar()
        layout.addWidget(self.inspection_bar)

        return pane

    def _build_wave_panel(self) -> QWidget:
        box = QFrame()
        box.setObjectName("Card")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 6, 8, 8)
        lay.setSpacing(6)

        hdr = QHBoxLayout()
        lbl = QLabel("Waveshaper Signal Preview")
        lbl.setStyleSheet(f"font-weight: 700; color: {PALETTE['cyan']}; background: transparent;")
        hdr.addWidget(lbl)

        self.combo_wave_type = QComboBox()
        self.combo_wave_type.addItems(["Sine", "Triangle", "Sawtooth"])
        self.combo_wave_type.currentIndexChanged.connect(self._update_wave_preview)
        hdr.addWidget(self.combo_wave_type)

        lbl_g = QLabel("Pre-Gain:")
        lbl_g.setStyleSheet("background: transparent;")
        hdr.addWidget(lbl_g)
        self.spin_pregain = QDoubleSpinBox()
        self.spin_pregain.setRange(0.1, 4.0)
        self.spin_pregain.setSingleStep(0.1)
        self.spin_pregain.setDecimals(2)
        self.spin_pregain.setValue(1.0)
        self.spin_pregain.setFixedWidth(68)
        self.spin_pregain.valueChanged.connect(self._update_wave_preview)
        hdr.addWidget(self.spin_pregain)

        self.combo_wave_view = QComboBox()
        self.combo_wave_view.addItems(["Time Waveform", "Harmonic Spectrum (FFT dB)"])
        self.combo_wave_view.currentIndexChanged.connect(self._update_wave_preview)
        hdr.addWidget(self.combo_wave_view)

        hdr.addStretch(1)
        lay.addLayout(hdr)

        self.wave_plot = pg.PlotWidget()
        self.wave_plot.setMinimumHeight(180)
        self.wave_plot.showGrid(x=True, y=True, alpha=0.2)
        self.wave_curve_in = self.wave_plot.plot(
            pen=pg.mkPen("#38bdf866", width=1.5, style=Qt.PenStyle.DashLine)
        )
        self.wave_curve_out = self.wave_plot.plot(
            pen=pg.mkPen(PALETTE["accent"], width=2.2),
            connect="finite",
        )
        self.fft_bars = pg.BarGraphItem(
            x=[], height=[], width=0.55, brush=pg.mkBrush("#ff8c32cc"), pen=pg.mkPen(PALETTE["accent"])
        )
        self.wave_plot.addItem(self.fft_bars)
        self.fft_bars.setVisible(False)
        lay.addWidget(self.wave_plot, stretch=1)
        return box

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("F1"), self, activated=lambda: self.btn_ref.toggle())
        QShortcut(QKeySequence("Ctrl+Shift+C"), self, activated=self._copy_fl_formula)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._on_formula_changed)

    # ------------------------------------------------------------------
    # Mode, Dialect, Preset Switching
    # ------------------------------------------------------------------

    def _target_bounds(self) -> tuple[float, float]:
        if self.current_mode == self.MODE_TRANSFER:
            return (-1.0, 1.0)
        return (0.0, 1.0)

    def _x_domain(self) -> tuple[float, float]:
        if self.current_mode == self.MODE_TRANSFER:
            return (-1.0, 1.0)
        if self.current_mode == self.MODE_CONTROLLER:
            return (0.0, float(self.spin_beats.value()))
        return (0.0, 1.0)

    def _set_mode(self, mode: str) -> None:
        self.current_mode = mode
        self.btn_mode_map.setChecked(mode == self.MODE_MAPPING)
        self.btn_mode_ctrl.setChecked(mode == self.MODE_CONTROLLER)
        self.btn_mode_xfer.setChecked(mode == self.MODE_TRANSFER)

        is_ctrl = mode == self.MODE_CONTROLLER
        is_xfer = mode == self.MODE_TRANSFER

        self.ctrl_box.setVisible(is_ctrl)
        if not is_ctrl and self._playing:
            self.btn_play.setChecked(False)
        self.playhead_line.setVisible(is_ctrl)
        self.playhead_dot.setVisible(is_ctrl)

        # Auto-show waveshaper preview in Transfer mode
        self.chk_show_wave.setChecked(is_xfer)
        self.wave_panel.setVisible(is_xfer)

        # Axis labels & clamp label
        if is_ctrl:
            self.plot_widget.setLabel("bottom", "SongTime (beats)")
            self.plot_widget.setLabel("left", "Controller Output [0..1]")
            self.chk_show_unity.setChecked(False)
            self.chk_export_clamp.setText("Clamp [0,1]")
        elif is_xfer:
            self.plot_widget.setLabel("bottom", "Input Amplitude (x)")
            self.plot_widget.setLabel("left", "Shaped Output f(x)")
            self.chk_show_unity.setChecked(True)
            self.chk_export_clamp.setText("Clamp [-1,1]")
        else:
            self.plot_widget.setLabel("bottom", "Input [0..1]")
            self.plot_widget.setLabel("left", "Mapped Output [0..1]")
            self.chk_show_unity.setChecked(True)
            self.chk_export_clamp.setText("Clamp [0,1]")

        self._sync_bounds_visuals()
        self._on_formula_changed()
        self._reset_plot_view()

    def _sync_bounds_visuals(self) -> None:
        lo, hi = self._target_bounds()
        self.region_high.setRegion((hi, hi + 200.0))
        self.region_low.setRegion((lo - 200.0, lo))
        self.line_hi.setPos(hi)
        self.line_lo.setPos(lo)

    def _on_dialect_changed(self, idx: int) -> None:
        did = self.combo_dialect.itemData(idx)
        if not did:
            return
        self.current_dialect = self.registry.get(did)
        self.editor.set_dialect(self.current_dialect)
        self.ref_drawer.set_dialect(self.current_dialect)
        self._on_formula_changed()

    def _on_preset_selected(self, idx: int) -> None:
        preset_idx = self.combo_preset.itemData(idx)
        if preset_idx is not None and 0 <= preset_idx < len(PRESETS):
            self._load_preset(PRESETS[preset_idx])

    def _load_preset(self, preset: FormulaPreset) -> None:
        # Sync preset dropdown display
        p_idx = self.combo_preset.findText(preset.name)
        if p_idx >= 0:
            self.combo_preset.blockSignals(True)
            self.combo_preset.setCurrentIndex(p_idx)
            self.combo_preset.blockSignals(False)

        # Select dialect first
        d_idx = self.combo_dialect.findData(preset.dialect)
        if d_idx >= 0:
            self.combo_dialect.blockSignals(True)
            self.combo_dialect.setCurrentIndex(d_idx)
            self.combo_dialect.blockSignals(False)
            self.current_dialect = self.registry.get(preset.dialect)
            self.editor.set_dialect(self.current_dialect)
            self.ref_drawer.set_dialect(self.current_dialect)

        # Pre-seed defaults
        for k, v in preset.defaults.items():
            self._param_values[k] = float(v)
            if k in self._fl_knob_rows:
                self._fl_knob_rows[k].set_value(float(v), emit=False)

        self.editor.blockSignals(True)
        self.editor.setPlainText(preset.formula)
        self.editor.blockSignals(False)

        self._set_mode(preset.mode)

        # Apply preset defaults onto newly created sliders too
        for k, v in preset.defaults.items():
            if k in self._param_rows:
                self._param_rows[k].set_value(float(v), emit=False)
        self._evaluate_and_plot()
        self._reset_plot_view()

    # ------------------------------------------------------------------
    # Formula Parsing, Parameter Sync & Evaluation
    # ------------------------------------------------------------------

    def _on_formula_changed(self) -> None:
        src = self.editor.toPlainText()
        try:
            ast = parse_formula(src, self.current_dialect)
        except FormulaError as err:
            self._ast = None
            self.editor.set_error_span(err.start, err.end)
            self.lbl_status.setText(f"Syntax Error: {err.message}")
            self.lbl_status.setStyleSheet(
                f"font-weight: 700; color: {PALETTE['red']}; background: transparent;"
            )
            self.export_preview.setText("")
            return

        self._ast = ast
        self.editor.clear_error_span()
        self._sync_parameter_sliders(ast)
        self._evaluate_and_plot()

    def _reserved_vars_for_mode(self) -> Set[str]:
        base = {"x", "input", "in", "songtime", "time", "date", "mousex", "mousey", "rand", "gettempo", "t"}
        if self.current_mode == self.MODE_CONTROLLER:
            base.update({"a", "b", "c"})
        return base

    def _sync_parameter_sliders(self, ast: ASTNode) -> None:
        # In Mapping/Transfer mode, a/b/c are allowed as custom sliders if the user writes them!
        reserved = self._reserved_vars_for_mode()
        # Temporarily ignore a/b/c from dialect special_variables when NOT in controller mode
        if self.current_mode != self.MODE_CONTROLLER:
            raw_params = []
            seen = set()
            from qfc.parser import collect_identifiers

            for ident in collect_identifiers(ast):
                norm = self.current_dialect.normalize_key(ident)
                if norm in seen or norm in reserved:
                    continue
                if self.current_dialect.lookup_constant(ident) is not None:
                    continue
                if norm not in ("a", "b", "c") and self.current_dialect.is_special_variable(ident):
                    continue
                seen.add(norm)
                raw_params.append(ident)
            found = raw_params
        else:
            found = extract_custom_params(ast, self.current_dialect, reserved_vars=reserved)

        # Remove sliders no longer present
        found_set = set(found)
        for existing in list(self._param_rows.keys()):
            if existing not in found_set:
                widget = self._param_rows.pop(existing)
                self.params_layout.removeWidget(widget)
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

        # Add sliders for newly discovered variables
        for name in found:
            if name not in self._param_rows:
                low = name.lower()
                default_val = self._param_values.get(name, self._smart_default(low))
                lo, hi = self._smart_range(low, default_val)
                row = ParameterSliderRow(
                    name=name,
                    value=default_val,
                    min_val=lo,
                    max_val=hi,
                    default_reset=default_val,
                )
                row.valueChanged.connect(self._on_param_changed)
                self._param_rows[name] = row
                self._param_values[name] = default_val
                # Insert before trailing stretch
                self.params_layout.insertWidget(self.params_layout.count() - 1, row)

        self.lbl_no_params.setVisible(len(self._param_rows) == 0)

    @staticmethod
    def _smart_default(low_name: str) -> float:
        if low_name in ("a", "b", "c", "blend", "mix", "duty", "pw", "thresh", "asym"):
            return 0.5
        if low_name in ("rate", "bend", "tension"):
            return 10.0
        if low_name in ("steps", "levels"):
            return 8.0
        if low_name in ("drive", "gain", "folds"):
            return 2.0
        return 1.0

    @staticmethod
    def _smart_range(low_name: str, val: float) -> tuple[float, float]:
        if low_name in ("a", "b", "c", "blend", "mix", "duty", "pw", "lo", "hi", "dz"):
            return (0.0, 1.0)
        if low_name in ("rate", "bend", "tension"):
            return (-100.0, 100.0)
        if low_name in ("steps", "levels", "bars"):
            return (1.0, 32.0)
        if low_name in ("drive", "gain", "folds", "sharpness", "decay", "freq", "speed"):
            return (0.01, max(10.0, val * 2.0))
        lo = min(-10.0, math.floor(val) - 5.0)
        hi = max(10.0, math.ceil(val) + 5.0)
        return (lo, hi)

    def _on_param_changed(self, name: str, val: float) -> None:
        self._param_values[name] = val
        self._evaluate_and_plot()

    def _build_eval_env(self, xs: np.ndarray) -> Dict[str, object]:
        bpm = float(self.spin_bpm.value())
        env: Dict[str, object] = dict(self._param_values)

        # FL Formula Controller built-in knobs
        for k, row in self._fl_knob_rows.items():
            if self.current_mode == self.MODE_CONTROLLER or k not in self._param_rows:
                env[k] = row.value()

        # Override with custom slider values
        for k, row in self._param_rows.items():
            env[k] = row.value()

        if self.current_mode == self.MODE_CONTROLLER:
            env["songtime"] = xs
            env["SongTime"] = xs
            env["t"] = xs
            env["time"] = xs * (60000.0 / max(bpm, 1.0))
            env["Time"] = env["time"]
            env["x"] = xs
            env["input"] = xs
            env["Input"] = xs
        else:
            env["x"] = xs
            env["input"] = xs
            env["Input"] = xs
            env["in"] = xs
            env["t"] = xs
            env["songtime"] = xs
            env["SongTime"] = xs
            env["time"] = xs * 1000.0
            env["Time"] = xs * 1000.0

        env["date"] = 0.5
        env["mousex"] = 0.5
        env["mousey"] = 0.5
        return env

    def _evaluate_and_plot(self) -> None:
        if self._ast is None:
            return

        x_lo, x_hi = self._x_domain()
        xs = np.linspace(x_lo, x_hi, 4096)
        env = self._build_eval_env(xs)
        bpm = float(self.spin_bpm.value())

        try:
            ys_raw = evaluate_ast(
                self._ast,
                self.current_dialect,
                env,
                size=len(xs),
                rng_seed=self._rng_seed,
                bpm=bpm,
            )
        except FormulaError as err:
            self.editor.set_error_span(err.start, err.end)
            self.lbl_status.setText(f"Eval Error: {err.message}")
            self.lbl_status.setStyleSheet(
                f"font-weight: 700; color: {PALETTE['red']}; background: transparent;"
            )
            return

        t_lo, t_hi = self._target_bounds()
        ys_clipped = np.clip(ys_raw, t_lo, t_hi)
        report = analyze_curve(xs, ys_raw, target_lo=t_lo, target_hi=t_hi)

        self._last_xs = xs
        self._last_ys_raw = ys_raw
        self._last_ys_clipped = ys_clipped
        self._last_report = report

        # Status badge
        if report.within_bounds:
            self.lbl_status.setText("Compiled OK  •  In bounds")
            self.lbl_status.setStyleSheet(
                f"font-weight: 700; color: {PALETTE['green']}; background: transparent;"
            )
        elif report.nan_count > 0 or report.inf_count > 0:
            self.lbl_status.setText(
                f"Compiled OK  •  Warning: {report.nan_count} NaN / {report.inf_count} Inf"
            )
            self.lbl_status.setStyleSheet(
                f"font-weight: 700; color: {PALETTE['amber']}; background: transparent;"
            )
        else:
            self.lbl_status.setText(
                f"Compiled OK  •  FL clips {report.total_clip_pct:.1f}% outside [{t_lo:g}, {t_hi:g}]"
            )
            self.lbl_status.setStyleSheet(
                f"font-weight: 700; color: {PALETTE['amber']}; background: transparent;"
            )

        # Plot curves
        self.curve_unity.setData([x_lo, x_hi], [x_lo, x_hi])
        self.curve_raw.setData(xs, ys_raw)
        self.curve_clipped.setData(xs, ys_clipped)
        self._update_plot_visibility()

        self.inspection_bar.update_report(report)
        self._update_export_preview()
        if self.wave_panel.isVisible():
            self._update_wave_preview()
        if self.current_mode == self.MODE_CONTROLLER:
            self._update_playhead_graphics()

    def _update_plot_visibility(self) -> None:
        self.curve_raw.setVisible(self.chk_show_raw.isChecked())
        self.curve_clipped.setVisible(self.chk_show_clip.isChecked())
        self.curve_unity.setVisible(self.chk_show_unity.isChecked())

    def _reset_plot_view(self) -> None:
        x_lo, x_hi = self._x_domain()
        t_lo, t_hi = self._target_bounds()
        pad_x = (x_hi - x_lo) * 0.03
        pad_y = (t_hi - t_lo) * 0.14
        y_lo = t_lo - pad_y
        y_hi = t_hi + pad_y
        if self._last_report is not None and math.isfinite(self._last_report.y_min) and math.isfinite(self._last_report.y_max):
            # Include slight overshoot if curve goes moderately beyond bounds
            y_lo = min(y_lo, max(t_lo - 2.0, self._last_report.y_min - pad_y * 0.5))
            y_hi = max(y_hi, min(t_hi + 2.0, self._last_report.y_max + pad_y * 0.5))
        self.plot_widget.setXRange(x_lo - pad_x, x_hi + pad_x, padding=0)
        self.plot_widget.setYRange(y_lo, y_hi, padding=0)

    # ------------------------------------------------------------------
    # Hover Crosshair Inspection
    # ------------------------------------------------------------------

    def _on_plot_mouse_moved(self, pos) -> None:
        if self._last_xs is None or self._last_ys_raw is None or self._last_ys_clipped is None:
            return
        vb = self.plot_widget.getViewBox()
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            self.cross_v.setVisible(False)
            self.cross_h.setVisible(False)
            self.cross_dot.setVisible(False)
            return

        pt = vb.mapSceneToView(pos)
        x_val = float(pt.x())
        xs = self._last_xs
        if x_val < xs[0] or x_val > xs[-1]:
            self.cross_v.setVisible(False)
            self.cross_h.setVisible(False)
            self.cross_dot.setVisible(False)
            return

        idx = int(np.searchsorted(xs, x_val))
        idx = max(0, min(len(xs) - 1, idx))
        xv = float(xs[idx])
        yr = float(self._last_ys_raw[idx])
        yc = float(self._last_ys_clipped[idx])

        self.cross_v.setPos(xv)
        self.cross_v.setVisible(True)
        if math.isfinite(yc):
            self.cross_h.setPos(yc)
            self.cross_h.setVisible(True)
            self.cross_dot.setData([xv], [yc])
            self.cross_dot.setVisible(True)
        else:
            self.cross_h.setVisible(False)
            self.cross_dot.setVisible(False)

        x_name = "SongTime" if self.current_mode == self.MODE_CONTROLLER else ("x" if self.current_mode == self.MODE_TRANSFER else "Input")
        yr_str = "NaN" if math.isnan(yr) else ("Inf" if math.isinf(yr) else f"{yr:.4f}")
        yc_str = "NaN" if math.isnan(yc) else f"{yc:.4f}"
        clipped = math.isfinite(yr) and abs(yr - yc) > 1e-6
        color = PALETTE["red"] if clipped else PALETTE["text"]
        self.lbl_cursor.setText(f"{x_name}={xv:.4f}  |  Raw={yr_str}  |  Out={yc_str}")
        self.lbl_cursor.setStyleSheet(
            f"font-family: 'Cascadia Code', monospace; font-size: 12px; color: {color}; background: transparent;"
        )

    # ------------------------------------------------------------------
    # Waveshaper Waveform & Harmonic Spectrum Preview
    # ------------------------------------------------------------------

    def _on_wave_toggle(self, checked: bool) -> None:
        self.wave_panel.setVisible(checked)
        if checked:
            self._update_wave_preview()

    def _update_wave_preview(self) -> None:
        if self._ast is None or not self.wave_panel.isVisible():
            return

        wave_kind = self.combo_wave_type.currentText()
        pre_gain = float(self.spin_pregain.value())
        is_fft = self.combo_wave_view.currentIndex() == 1

        cycles = 8 if is_fft else 2
        t, wave_in = generate_test_waveform(kind=wave_kind, cycles=float(cycles), samples=4096, pre_gain=pre_gain)

        # If in [0, 1] mapping mode, map bipolar test wave into [0, 1] for shaping then back for display
        if self.current_mode != self.MODE_TRANSFER:
            eval_in = np.clip(wave_in * 0.5 + 0.5, 0.0, 1.0)
        else:
            eval_in = wave_in

        env = self._build_eval_env(eval_in)
        try:
            shaped = evaluate_ast(
                self._ast,
                self.current_dialect,
                env,
                size=len(eval_in),
                rng_seed=self._rng_seed,
                bpm=float(self.spin_bpm.value()),
            )
        except FormulaError:
            return

        t_lo, t_hi = self._target_bounds()
        shaped = np.clip(shaped, t_lo, t_hi)

        if is_fft:
            self.wave_curve_in.setVisible(False)
            self.wave_curve_out.setVisible(False)
            self.fft_bars.setVisible(True)
            # Center [0, 1] curves before FFT so DC doesn't swamp harmonics
            sig = shaped * 2.0 - 1.0 if self.current_mode != self.MODE_TRANSFER else shaped
            harmonics, db = compute_harmonic_spectrum(sig, cycles=cycles, max_harmonic=16)
            heights = db + 96.0  # floor at -96 dB
            self.fft_bars.setOpts(x=harmonics, height=heights, y0=-96.0)
            self.wave_plot.setLabel("bottom", "Harmonic Number (0=DC, 1=Fund, 2=2nd...)")
            self.wave_plot.setLabel("left", "Level (dB)")
            self.wave_plot.setXRange(-0.6, 16.6, padding=0)
            self.wave_plot.setYRange(-96.0, 6.0, padding=0)
        else:
            self.fft_bars.setVisible(False)
            self.wave_curve_in.setVisible(True)
            self.wave_curve_out.setVisible(True)
            self.wave_curve_in.setData(t, eval_in)
            self.wave_curve_out.setData(t, shaped)
            self.wave_plot.setLabel("bottom", "Cycles")
            self.wave_plot.setLabel("left", "Amplitude")
            self.wave_plot.setXRange(0.0, float(cycles), padding=0)
            if self.current_mode == self.MODE_TRANSFER:
                self.wave_plot.setYRange(-1.15, 1.15, padding=0)
            else:
                self.wave_plot.setYRange(-0.1, 1.1, padding=0)

    # ------------------------------------------------------------------
    # Formula Controller Live Playhead
    # ------------------------------------------------------------------

    def _toggle_play(self, playing: bool) -> None:
        self._playing = playing
        if playing:
            self.btn_play.setText("⏸ Pause LFO")
            self._last_tick_time = time.perf_counter()
            self._play_timer.start()
        else:
            self.btn_play.setText("▶ Play LFO")
            self._play_timer.stop()

    def _on_play_tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_tick_time
        self._last_tick_time = now
        beats_per_sec = float(self.spin_bpm.value()) / 60.0
        window = max(0.25, float(self.spin_beats.value()))
        self._play_beat = (self._play_beat + dt * beats_per_sec) % window
        self._update_playhead_graphics()

    def _update_playhead_graphics(self) -> None:
        if self._last_xs is None or self._last_ys_clipped is None:
            return
        xs = self._last_xs
        window = max(0.25, float(self.spin_beats.value()))
        beat = self._play_beat % window
        idx = int(np.searchsorted(xs, beat))
        idx = max(0, min(len(xs) - 1, idx))
        y_val = float(self._last_ys_clipped[idx])
        if not math.isfinite(y_val):
            y_val = 0.0

        self.playhead_line.setPos(beat)
        self.playhead_dot.setData([beat], [y_val])
        self.live_meter.setValue(int(round(max(0.0, min(1.0, y_val)) * 1000)))
        self.live_meter.setFormat(f"{y_val:.3f}  (beat {beat:.2f})")

    # ------------------------------------------------------------------
    # FL Studio Export & Auto-Normalisation
    # ------------------------------------------------------------------

    def _update_export_preview(self) -> None:
        if self._ast is None:
            self.export_preview.setText("")
            self.lbl_export_warn.setVisible(False)
            return

        t_lo, t_hi = self._target_bounds()
        norm_bounds = None
        if (
            self.chk_export_norm.isChecked()
            and self._last_report is not None
            and math.isfinite(self._last_report.y_min)
            and math.isfinite(self._last_report.y_max)
        ):
            norm_bounds = (self._last_report.y_min, self._last_report.y_max, t_lo, t_hi)

        clamp_range = (t_lo, t_hi) if self.chk_export_clamp.isChecked() else None
        primary_target = "Input" if self.current_mode == self.MODE_MAPPING else (None if self.current_mode == self.MODE_CONTROLLER else "x")

        exported, warnings = export_to_fl_studio(
            self._ast,
            self.current_dialect,
            param_values=self._param_values,
            bake_custom_params=self.chk_bake.isChecked(),
            primary_var_target=primary_target,
            normalize_bounds=norm_bounds,
            clamp_range=clamp_range,
        )
        self.export_preview.setText(exported)
        if warnings:
            self.lbl_export_warn.setText("Note: " + " ".join(warnings))
            self.lbl_export_warn.setVisible(True)
        else:
            self.lbl_export_warn.setVisible(False)

    def _apply_auto_normalise(self) -> None:
        if self._ast is None or self._last_report is None:
            return
        r = self._last_report
        if not (math.isfinite(r.y_min) and math.isfinite(r.y_max)) or r.span <= 1e-12:
            return
        t_lo, t_hi = self._target_bounds()
        # Keep the user's variable names intact when applying directly into editor!
        clean_expr, _ = export_to_fl_studio(
            self._ast,
            self.current_dialect,
            param_values=self._param_values,
            bake_custom_params=False,
            primary_var_target=None,
        )
        wrapped = wrap_normalized_formula(clean_expr, r.y_min, r.y_max, t_lo, t_hi)
        self.editor.setPlainText(wrapped)

    def _copy_fl_formula(self) -> None:
        text = self.export_preview.text().strip()
        if not text:
            return
        QGuiApplication.clipboard().setText(text)
        orig = self.btn_copy_fl.text()
        self.btn_copy_fl.setText("✓ Copied!")
        QTimer.singleShot(1400, lambda: self.btn_copy_fl.setText(orig))

    def _on_reseed(self) -> None:
        self._rng_seed = (self._rng_seed * 1664525 + 1013904223) & 0x7FFFFFFF
        self._evaluate_and_plot()
