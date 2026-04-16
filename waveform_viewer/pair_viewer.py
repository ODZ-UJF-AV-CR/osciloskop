#!/usr/bin/env python3
"""
pair_viewer.py — Zobrazení páru H5 + CSV souborů.
Horní graf: data z CSV souboru (scatter/line plot).
Spodní graf: waveform z H5 souboru pro vybraný vzorek.
Kliknutí na bod synchronizuje zvýraznění v obou grafech.
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import sys
from pathlib import Path

import numpy as np

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
except ImportError:
    print("Chyba: vyžaduje pyqtgraph a PyQt5/PyQt6.")
    sys.exit(1)

from waveform_viewer import (
    H5WaveformDataset,
    centered_moving_average,
    compute_waveform_measurements,
    SMOOTHING_WINDOW,
    CURRENT_COLORS,
)
from view_fw import parse_csv_log, CEvent

pg.setConfigOptions(antialias=False)

CSV_LINE_COLOR = "#4cc9f0"
CSV_HIGHLIGHT_COLOR = "#ef476f"
WF_TRACE_COLOR = "#ffd166"
WF_SMOOTH_COLOR = "#95d67b"


class PairViewer(QtWidgets.QMainWindow):
    """Viewer pro pár H5 + CSV souborů se synchronizovaným zvýrazněním."""

    def __init__(self, h5_path: str, csv_path: str) -> None:
        super().__init__()
        self.setWindowTitle("H5 + CSV Pair Viewer")
        self.resize(1400, 900)

        # Načíst data
        self.dataset = H5WaveformDataset(h5_path)
        self.events, self.device_info = parse_csv_log(csv_path)

        # Přeskočit první vzorek v H5
        self.h5_sample_ids = self.dataset.sample_ids[1:]

        # Počet spárovaných vzorků (minimum z obou)
        self.pair_count = min(len(self.events), len(self.h5_sample_ids))

        self.selected_index: int = -1
        self.skipped: set[int] = set()  # indexy přeskočených bodů
        self._skipped_path = Path(h5_path).parent / f"{Path(h5_path).stem}_{Path(csv_path).stem}_skipped.json"
        self._load_skipped()
        self.csv_scatter: pg.ScatterPlotItem | None = None
        self.skip_scatter: pg.ScatterPlotItem | None = None
        self.highlight_items: list = []
        self.wf_highlight_items: list = []
        self.wf_pending_autorange: bool = True
        self._csv_norm_min: float = 0.0
        self._csv_norm_range: float = 1.0

        # Předpočítáme amplitudy z CH2 pro horní graf
        self.ch2_amplitudes = self._compute_ch2_amplitudes()

        # Načíst io_markers
        self.marker_csv_indices: set[int] = set()

        self._build_ui()
        self._draw_csv_plot()
        self._draw_histogram()

    @staticmethod
    def _is_ch2(spec) -> bool:
        """Rozpoznání CH2 série (CH2, CHAN2, ...)."""
        name = spec.name.upper()
        return name in ("CH2", "CHAN2") or spec.title.upper() in ("CH2", "CHAN2")

    def _find_ch2_name(self) -> str | None:
        """Najdi název CH2 série."""
        for spec in self.dataset.series_specs:
            if self._is_ch2(spec):
                return spec.name
        return None

    def _h5_index_for_csv(self, csv_idx: int) -> int | None:
        """Vrátí index do h5_sample_ids pro daný CSV index.

        Přeskočené body vkládají mezeru — následné H5 vzorky se posouvají.
        """
        if csv_idx in self.skipped:
            return None
        skip_count = sum(1 for s in self.skipped if s < csv_idx)
        h5_idx = csv_idx - skip_count
        if 0 <= h5_idx < len(self.h5_sample_ids):
            return h5_idx
        return None

    def _compute_ch2_amplitudes(self) -> np.ndarray:
        """Vypočítat amplitudy z CH2 waveformů s ohledem na přeskočené body."""
        ch2_name = self._find_ch2_name()
        amps = np.zeros(self.pair_count, dtype=np.float64)
        if ch2_name is None:
            return amps

        for i in range(self.pair_count):
            h5_idx = self._h5_index_for_csv(i)
            if h5_idx is None:
                amps[i] = 0.0
                continue
            h5_sid = self.h5_sample_ids[h5_idx]
            wf = self.dataset.waveform(h5_sid, ch2_name)
            if wf is not None:
                amp, _area, _delay = compute_waveform_measurements(wf)
                amps[i] = amp
        return amps

    def load_io_markers(self, markers_path: str | Path | None = None, h5_path: str | None = None) -> None:
        """Načíst io_markers CSV s waveform_id.

        Pokud markers_path není zadán, hledá {h5_stem}_io_markers.csv.
        Mapuje waveform_id z H5 na CSV indexy a uloží do self.marker_csv_indices.
        """
        if markers_path is None and h5_path is not None:
            markers_path = Path(h5_path).parent / f"{Path(h5_path).stem}_io_markers.csv"
            if not Path(markers_path).exists():
                print(f"IO markers: soubor {markers_path} neexistuje")
                return
        elif markers_path is not None:
            markers_path = Path(markers_path)
            if not markers_path.exists():
                print(f"IO markers: soubor {markers_path} neexistuje")
                return
        else:
            return

        # Načíst waveform_id z prvního sloupce
        marker_h5_ids: set[int] = set()
        try:
            with open(markers_path, "r") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    wid = row.get("waveform_id", "")
                    if wid.strip().isdigit():
                        marker_h5_ids.add(int(wid))
        except (OSError, KeyError):
            return

        if not marker_h5_ids:
            return

        print(f"IO markers: načteno {len(marker_h5_ids)} waveform_id z {markers_path}")

        # Mapovat H5 sample_id → CSV index (inverzní k _h5_index_for_csv)
        h5_sid_to_csv: dict[int, int] = {}
        for csv_idx in range(self.pair_count):
            h5_idx = self._h5_index_for_csv(csv_idx)
            if h5_idx is not None:
                h5_sid_to_csv[self.h5_sample_ids[h5_idx]] = csv_idx

        matched = 0
        for wid in marker_h5_ids:
            if wid in h5_sid_to_csv:
                self.marker_csv_indices.add(h5_sid_to_csv[wid])
                matched += 1

        print(f"IO markers: {matched}/{len(marker_h5_ids)} namapováno na CSV indexy")

    # -- Sestavení UI ----------------------------------------------------------

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # -- Ovládací prvky --
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QtWidgets.QLabel("Osa Y (CSV):"))
        self.y_column_combo = QtWidgets.QComboBox()
        self.y_column_combo.addItem("val3", "val3")
        self.y_column_combo.addItem("val4", "val4")
        self.y_column_combo.addItem("machine_time", "machine_time")
        self.y_column_combo.currentIndexChanged.connect(self._on_column_changed)
        controls.addWidget(self.y_column_combo)

        controls.addWidget(QtWidgets.QLabel("Osa X:"))
        self.x_axis_combo = QtWidgets.QComboBox()
        self.x_axis_combo.addItem("Index", "index")
        self.x_axis_combo.addItem("Machine time", "machine_time")
        self.x_axis_combo.currentIndexChanged.connect(self._on_column_changed)
        controls.addWidget(self.x_axis_combo)

        controls.addWidget(QtWidgets.QLabel("Série H5:"))
        self.series_combo = QtWidgets.QComboBox()
        self.series_combo.addItem("Všechny", None)
        default_idx = 0
        for i, spec in enumerate(self.dataset.series_specs):
            self.series_combo.addItem(spec.title, spec.name)
            if self._is_ch2(spec):
                default_idx = i + 1  # +1 kvůli "Všechny"
        self.series_combo.setCurrentIndex(default_idx)
        self.series_combo.currentIndexChanged.connect(self._refresh_waveform)
        controls.addWidget(self.series_combo)

        controls.addWidget(QtWidgets.QLabel("Vzorek:"))
        self.sample_spin = QtWidgets.QSpinBox()
        self.sample_spin.setRange(0, max(0, self.pair_count - 1))
        self.sample_spin.setKeyboardTracking(False)
        self.sample_spin.valueChanged.connect(self._on_spin_changed)
        controls.addWidget(self.sample_spin)

        self.pos_label = QtWidgets.QLabel(f"/ {self.pair_count}")
        controls.addWidget(self.pos_label)

        self.info_label = QtWidgets.QLabel(
            f"CSV: {len(self.events)} $C událostí | "
            f"H5: {len(self.dataset.sample_ids)} vzorků (1. přeskočen) | "
            f"Párů: {self.pair_count}"
        )
        controls.addWidget(self.info_label)

        self.skip_button = QtWidgets.QPushButton("Přeskočit (S)")
        self.skip_button.setToolTip("Označit/odoznačit aktuální bod jako přeskočený")
        self.skip_button.clicked.connect(self._toggle_skip_current)
        controls.addWidget(self.skip_button)

        self.skip_count_label = QtWidgets.QLabel("Přeskočeno: 0")
        controls.addWidget(self.skip_count_label)

        controls.addWidget(QtWidgets.QLabel("Škála CH2:"))
        self.ch2_scale_spin = QtWidgets.QDoubleSpinBox()
        self.ch2_scale_spin.setRange(0.001, 100.0)
        self.ch2_scale_spin.setValue(0.1)
        self.ch2_scale_spin.setSingleStep(0.01)
        self.ch2_scale_spin.setDecimals(3)
        self.ch2_scale_spin.setToolTip("Faktor škálování oranžového CH2 grafu")
        self.ch2_scale_spin.valueChanged.connect(self._on_column_changed)
        controls.addWidget(self.ch2_scale_spin)

        controls.addStretch()
        layout.addLayout(controls)

        # -- Hlavní splitter: levý (grafy) + pravý (histogram) --
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # -- Levá strana: dva grafy přes sebe --
        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        self.csv_plot = pg.PlotWidget()
        self.csv_plot.setBackground("#101418")
        self.csv_plot.showGrid(x=True, y=True, alpha=0.25)
        self.csv_plot.setLabel("bottom", "Vzorek (CSV index)")
        self.csv_plot.setTitle("CSV data")
        left_splitter.addWidget(self.csv_plot)

        self.wf_plot = pg.PlotWidget()
        self.wf_plot.setBackground("#101418")
        self.wf_plot.showGrid(x=True, y=True, alpha=0.25)
        self.wf_plot.setLabel("left", "Amplituda [V]")
        self.wf_plot.setLabel("bottom", "Čas [µs]")
        self.wf_plot.setTitle("Waveform z H5 — vyber bod v CSV grafu")
        left_splitter.addWidget(self.wf_plot)

        left_splitter.setSizes([400, 500])
        main_splitter.addWidget(left_splitter)

        # -- Pravá strana: 2D histogram val3 vs val4 --
        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setBackground("#101418")
        self.hist_plot.showGrid(x=True, y=True, alpha=0.25)
        self.hist_plot.setLabel("left", "val4")
        self.hist_plot.setLabel("bottom", "val3")
        self.hist_plot.setTitle("2D Histogram: val3 vs val4")
        self.hist_scatter: pg.ScatterPlotItem | None = None
        self.hist_crosshair_items: list = []
        main_splitter.addWidget(self.hist_plot)

        main_splitter.setSizes([900, 500])
        layout.addWidget(main_splitter, stretch=1)

        self.statusBar().showMessage(
            "Klikni na bod v CSV grafu nebo použij šipky ←/→ pro navigaci."
        )

    # -- Vykreslení CSV grafu --------------------------------------------------

    def _get_event_values(self, field: str) -> np.ndarray:
        """Vrátit pole hodnot z $C událostí pro dané pole."""
        if field == "val3":
            return np.array([self.events[i].val3 for i in range(self.pair_count)], dtype=np.float64)
        elif field == "val4":
            return np.array([self.events[i].val4 for i in range(self.pair_count)], dtype=np.float64)
        else:
            return np.array([self.events[i].machine_time for i in range(self.pair_count)], dtype=np.float64)

    def _get_x_values(self) -> np.ndarray:
        """Vrátit X hodnoty pro horní graf dle vybraného režimu."""
        mode = self.x_axis_combo.currentData() if hasattr(self, 'x_axis_combo') else "index"
        if mode == "machine_time":
            mt = np.array([self.events[i].machine_time for i in range(self.pair_count)], dtype=np.float64)
            return mt * 128e-6  # 1 tik = 128 µs → sekundy
        return np.arange(self.pair_count, dtype=np.float64)

    def _draw_csv_plot(self) -> None:
        """Vykreslit $C data do horního grafu."""
        # Zachovat přiblížení horního grafu
        csv_prev_range = None
        if hasattr(self, '_csv_plot_initialized') and self._csv_plot_initialized:
            csv_prev_range = self.csv_plot.plotItem.vb.viewRange()

        self.csv_plot.clear()
        self.csv_scatter = None
        self.skip_scatter = None
        self.highlight_items.clear()

        if self.pair_count == 0:
            return

        field = self.y_column_combo.currentData()
        if field is None:
            field = "val3"

        x = self._get_x_values()
        y = self._get_event_values(field)

        # Normalizujeme na 0–1
        y_min, y_max = float(np.min(y)), float(np.max(y))
        y_range = y_max - y_min if y_max > y_min else 1.0
        y_norm = (y - y_min) / y_range
        self._csv_norm_min = y_min
        self._csv_norm_range = y_range

        # CSV čárové grafy — val3 + val4 (nebo vybraný field)
        if field == "val3":
            # Zobrazit oba: val3 a val4
            y_val4 = self._get_event_values("val4")
            y4_norm = (y_val4 - y_min) / y_range  # stejná škála jako val3
            self.csv_plot.plot(
                x, y_norm,
                pen=pg.mkPen(CSV_LINE_COLOR, width=1.2),
                name="val3 (norm)",
            )
            self.csv_plot.plot(
                x, y4_norm,
                pen=pg.mkPen("#95d67b", width=1.0),
                name="val4 (norm)",
            )
        elif field == "val4":
            y_val3 = self._get_event_values("val3")
            y3_norm = (y_val3 - y_min) / y_range
            self.csv_plot.plot(
                x, y_norm,
                pen=pg.mkPen(CSV_LINE_COLOR, width=1.2),
                name="val4 (norm)",
            )
            self.csv_plot.plot(
                x, y3_norm,
                pen=pg.mkPen("#95d67b", width=1.0),
                name="val3 (norm)",
            )
        else:
            self.csv_plot.plot(
                x, y_norm,
                pen=pg.mkPen(CSV_LINE_COLOR, width=1.2),
                name=f"{field} (norm)",
            )

        # Scatter body (normální) pro klikání
        normal_mask = np.array([i not in self.skipped for i in range(self.pair_count)])
        skip_mask = ~normal_mask

        if np.any(normal_mask):
            self.csv_scatter = pg.ScatterPlotItem(
                x=x[normal_mask], y=y_norm[normal_mask],
                pen=pg.mkPen(CSV_LINE_COLOR, width=0.5),
                brush=pg.mkBrush(CSV_LINE_COLOR + "80"),
                size=6,
            )
            self._scatter_to_global = np.where(normal_mask)[0].tolist()
            self.csv_scatter.sigClicked.connect(self._on_csv_clicked)
            self.csv_plot.addItem(self.csv_scatter)

        # Přeskočené body — červené křížky
        if np.any(skip_mask):
            self.skip_scatter = pg.ScatterPlotItem(
                x=x[skip_mask], y=y_norm[skip_mask],
                pen=pg.mkPen("#ff2222", width=1.5),
                brush=pg.mkBrush("#ff222240"),
                size=10,
                symbol="x",
            )
            self._skip_scatter_to_global = np.where(skip_mask)[0].tolist()
            self.skip_scatter.sigClicked.connect(self._on_skip_scatter_clicked)
            self.csv_plot.addItem(self.skip_scatter)

        # IO marker body — žluté tečky na označených waveformech
        if self.marker_csv_indices:
            marker_mask = np.array([i in self.marker_csv_indices for i in range(self.pair_count)])
            if np.any(marker_mask):
                marker_scatter = pg.ScatterPlotItem(
                    x=x[marker_mask], y=y_norm[marker_mask],
                    pen=pg.mkPen("#ffd700", width=2),
                    brush=pg.mkBrush("#ffd700c0"),
                    size=16,
                    symbol="t",
                )
                marker_scatter.setZValue(50)
                self.csv_plot.addItem(marker_scatter)

        # CH2 amplitudy normalizované (oranžově) — přeskočené = mezera (0)
        amp_color = "#ff8833"
        amps = self.ch2_amplitudes
        # Normalizace z nepřeskočených hodnot
        valid = amps[amps > 0] if np.any(amps > 0) else amps
        a_min = float(np.min(valid)) if len(valid) > 0 else 0.0
        a_max = float(np.max(valid)) if len(valid) > 0 else 1.0
        a_range = a_max - a_min if a_max > a_min else 1.0
        ch2_scale = self.ch2_scale_spin.value()
        amp_norm = (amps - a_min) / a_range * ch2_scale
        # Přeskočené explicitně na 0
        for si in self.skipped:
            if 0 <= si < len(amp_norm):
                amp_norm[si] = 0.0

        self.csv_plot.plot(
            x, amp_norm,
            pen=pg.mkPen(amp_color, width=1.8),
            name="CH2 max (norm)",
        )

        self.csv_plot.setLabel("left", "Normalizovaná hodnota")
        x_label = "Čas [s]" if (hasattr(self, 'x_axis_combo') and self.x_axis_combo.currentData() == "machine_time") else "Vzorek (CSV index)"
        self.csv_plot.setLabel("bottom", x_label)
        self.csv_plot.setTitle(f"$C {field} + CH2 amplituda | přeskočeno: {len(self.skipped)}")
        self.csv_plot.addLegend(offset=(10, 10))

        # Obnovit přiblížení nebo autorange
        if csv_prev_range is not None:
            self.csv_plot.setXRange(*csv_prev_range[0], padding=0)
            self.csv_plot.setYRange(*csv_prev_range[1], padding=0)
        else:
            self.csv_plot.autoRange()
            self._csv_plot_initialized = True

        self.skip_count_label.setText(f"Přeskočeno: {len(self.skipped)}")

        # Obnovit zvýraznění pokud je vybraný vzorek
        if 0 <= self.selected_index < self.pair_count:
            self._highlight_csv(self.selected_index)

    def _on_column_changed(self) -> None:
        self._draw_csv_plot()

    # -- 2D Histogram val3 vs val4 ---------------------------------------------

    def _draw_histogram(self) -> None:
        """Vykreslit 2D scatter histogram val3 vs val4."""
        self.hist_plot.clear()
        self.hist_scatter = None
        self.hist_crosshair_items.clear()

        if self.pair_count == 0:
            return

        v3 = self._get_event_values("val3")
        v4 = self._get_event_values("val4")

        # 2D histogram heatmapa na pozadí (rozmezí 0–3000, bin 20×20)
        hist_range = [[0, 3000], [0, 3000]]
        bin_size = 20
        n_bins = int(3000 / bin_size)
        if len(v3) > 10:
            hist, xedges, yedges = np.histogram2d(v3, v4, bins=n_bins, range=hist_range)

            img = pg.ImageItem()
            positions = np.array([0.0, 0.01, 0.05, 0.15, 0.4, 0.7, 1.0])
            colors_rgba = np.array([
                [0, 0, 0, 255],
                [10, 10, 40, 255],
                [30, 50, 150, 255],
                [60, 180, 100, 255],
                [200, 220, 50, 255],
                [255, 120, 30, 255],
                [255, 255, 255, 255],
            ], dtype=np.ubyte)
            cmap = pg.ColorMap(positions, colors_rgba)
            lut = cmap.getLookupTable(nPts=256)

            log_hist = np.log1p(hist)
            img.setImage(log_hist)
            img.setLookupTable(lut)
            img.setRect(QtCore.QRectF(
                xedges[0], yedges[0],
                xedges[-1] - xedges[0], yedges[-1] - yedges[0],
            ))
            img.setZValue(-10)
            img.setOpacity(0.8)
            self.hist_plot.addItem(img)

        # Scatter body
        self.hist_scatter = pg.ScatterPlotItem(
            x=v3, y=v4,
            pen=pg.mkPen(CSV_LINE_COLOR, width=0.3),
            brush=pg.mkBrush(CSV_LINE_COLOR + "60"),
            size=4,
        )
        self.hist_scatter.sigClicked.connect(self._on_hist_clicked)
        self.hist_plot.addItem(self.hist_scatter)

        self.hist_plot.setXRange(0, 3000, padding=0.02)
        self.hist_plot.setYRange(0, 3000, padding=0.02)

    def _on_hist_clicked(self, scatter_item, points, ev) -> None:
        """Kliknutí na bod v histogramu."""
        if len(points) == 0:
            return
        idx = points[0].index()
        if 0 <= idx < self.pair_count:
            self._select_index(idx)

    def _highlight_histogram(self, index: int) -> None:
        """Zvýraznit vybraný bod v histogramu crosshairem."""
        for item in self.hist_crosshair_items:
            self.hist_plot.removeItem(item)
        self.hist_crosshair_items.clear()

        if index < 0 or index >= self.pair_count:
            return

        ev = self.events[index]
        vx = float(ev.val3)
        vy = float(ev.val4)

        pen = pg.mkPen(CSV_HIGHLIGHT_COLOR, width=1.5, style=QtCore.Qt.PenStyle.DashLine)
        vline = pg.InfiniteLine(pos=vx, angle=90, pen=pen)
        hline = pg.InfiniteLine(pos=vy, angle=0, pen=pen)
        self.hist_plot.addItem(vline)
        self.hist_plot.addItem(hline)
        self.hist_crosshair_items.extend([vline, hline])

        marker = pg.ScatterPlotItem(
            x=[vx], y=[vy],
            pen=pg.mkPen("#ffffff", width=2),
            brush=pg.mkBrush(CSV_HIGHLIGHT_COLOR),
            size=14,
            symbol="crosshair",
        )
        marker.setZValue(100)
        self.hist_plot.addItem(marker)
        self.hist_crosshair_items.append(marker)

    # -- Interakce a synchronizace ---------------------------------------------

    def _on_csv_clicked(self, scatter_item, points, ev) -> None:
        """Kliknutí na bod v CSV scatter plotu (normální body)."""
        if len(points) == 0:
            return
        local_idx = points[0].index()
        if hasattr(self, '_scatter_to_global') and 0 <= local_idx < len(self._scatter_to_global):
            self._select_index(self._scatter_to_global[local_idx])

    def _on_skip_scatter_clicked(self, scatter_item, points, ev) -> None:
        """Kliknutí na přeskočený bod."""
        if len(points) == 0:
            return
        local_idx = points[0].index()
        if hasattr(self, '_skip_scatter_to_global') and 0 <= local_idx < len(self._skip_scatter_to_global):
            self._select_index(self._skip_scatter_to_global[local_idx])

    def _on_spin_changed(self, value: int) -> None:
        if 0 <= value < self.pair_count:
            self._select_index(value)

    def _select_index(self, index: int) -> None:
        """Vybrat vzorek a synchronizovat oba grafy."""
        self.selected_index = index

        self.sample_spin.blockSignals(True)
        self.sample_spin.setValue(index)
        self.sample_spin.blockSignals(False)

        self._update_skip_button_text()
        self._highlight_csv(index)
        self._highlight_histogram(index)
        self._refresh_waveform()

        h5_idx = self._h5_index_for_csv(index)
        ev = self.events[index]
        if h5_idx is not None:
            h5_sid = self.h5_sample_ids[h5_idx]
            skip_str = ""
        else:
            h5_sid = "---"
            skip_str = " [PŘESKOČEN]"
        marker_str = " ★MARKER" if index in self.marker_csv_indices else ""
        self.statusBar().showMessage(
            f"Index {index} | H5 wf #{h5_sid}{skip_str}{marker_str} | "
            f"$C mt={ev.machine_time} val3={ev.val3} val4={ev.val4} | "
            f"{ev.timestamp}"
        )

    def _highlight_csv(self, index: int) -> None:
        """Zvýraznit vybraný bod v CSV grafu (crosshair + marker)."""
        for item in self.highlight_items:
            self.csv_plot.removeItem(item)
        self.highlight_items.clear()

        if index < 0 or index >= self.pair_count:
            return

        field = self.y_column_combo.currentData() or "val3"
        ev = self.events[index]
        val = ev.val3 if field == "val3" else (ev.val4 if field == "val4" else ev.machine_time)

        x_mode = self.x_axis_combo.currentData() if hasattr(self, 'x_axis_combo') else "index"
        if x_mode == "machine_time":
            x = float(self.events[index].machine_time) * 128e-6
        else:
            x = float(index)
        y = (float(val) - self._csv_norm_min) / self._csv_norm_range

        # Vertikální a horizontální čára
        vline = pg.InfiniteLine(
            pos=x, angle=90,
            pen=pg.mkPen(CSV_HIGHLIGHT_COLOR, width=1.5, style=QtCore.Qt.PenStyle.DashLine),
        )
        hline = pg.InfiniteLine(
            pos=y, angle=0,
            pen=pg.mkPen(CSV_HIGHLIGHT_COLOR, width=1.0, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.csv_plot.addItem(vline)
        self.csv_plot.addItem(hline)
        self.highlight_items.extend([vline, hline])

        # Marker
        marker = pg.ScatterPlotItem(
            x=[x], y=[y],
            pen=pg.mkPen("#ffffff", width=2),
            brush=pg.mkBrush(CSV_HIGHLIGHT_COLOR),
            size=14,
            symbol="crosshair",
        )
        marker.setZValue(100)
        self.csv_plot.addItem(marker)
        self.highlight_items.append(marker)

    # -- Přeskočení bodů -------------------------------------------------------

    def _save_skipped(self) -> None:
        """Uložit přeskočené body do JSON souboru."""
        data = {"skipped": sorted(self.skipped)}
        try:
            with open(self._skipped_path, "w") as f:
                json.dump(data, f)
        except OSError:
            pass

    def _load_skipped(self) -> None:
        """Načíst přeskočené body z JSON souboru."""
        if self._skipped_path.exists():
            try:
                with open(self._skipped_path, "r") as f:
                    data = json.load(f)
                self.skipped = set(data.get("skipped", []))
            except (json.JSONDecodeError, OSError):
                pass

    def _toggle_skip_current(self) -> None:
        """Přepnout přeskočení aktuálního bodu."""
        if self.selected_index < 0:
            return
        if self.selected_index in self.skipped:
            self.skipped.discard(self.selected_index)
        else:
            self.skipped.add(self.selected_index)
        self._save_skipped()
        # Přepočítat CH2 amplitudy s novým mapováním
        self.ch2_amplitudes = self._compute_ch2_amplitudes()
        self._update_skip_button_text()
        self._draw_csv_plot()
        # Posunout na další nepřeskočený
        self._navigate_skip(1)

    def _update_skip_button_text(self) -> None:
        if self.selected_index in self.skipped:
            self.skip_button.setText("Odoznačit (S)")
        else:
            self.skip_button.setText("Přeskočit (S)")

    def _navigate_skip(self, delta: int) -> None:
        """Posunout se na další/předchozí nepřeskočený bod."""
        idx = self.selected_index + delta
        while 0 <= idx < self.pair_count and idx in self.skipped:
            idx += delta
        if 0 <= idx < self.pair_count:
            self._select_index(idx)

    # -- Vykreslení waveformu --------------------------------------------------

    def _refresh_waveform(self) -> None:
        """Vykreslit waveform ze spárovaného H5 vzorku."""
        # Zachovat škálu pokud už něco bylo zobrazeno
        previous_range = None
        if not self.wf_pending_autorange:
            previous_range = self.wf_plot.plotItem.vb.viewRange()

        self.wf_plot.clear()
        self.wf_highlight_items.clear()

        if self.selected_index < 0 or self.selected_index >= self.pair_count:
            self.wf_plot.setTitle("Waveform z H5 — vyber bod v CSV grafu")
            return

        if self.selected_index in self.skipped:
            self.wf_plot.setTitle(f"Vzorek {self.selected_index} — PŘESKOČEN")
            return

        h5_idx = self._h5_index_for_csv(self.selected_index)
        if h5_idx is None:
            self.wf_plot.setTitle(f"Vzorek {self.selected_index} — mimo rozsah H5")
            return
        h5_sid = self.h5_sample_ids[h5_idx]
        selected_series = self.series_combo.currentData()

        # Které série zobrazit
        if selected_series is None:
            specs = self.dataset.series_specs
        else:
            specs = [s for s in self.dataset.series_specs if s.name == selected_series]

        has_data = False
        for idx, spec in enumerate(specs):
            wf = self.dataset.waveform(h5_sid, spec.name)
            if wf is None:
                continue
            has_data = True

            color = CURRENT_COLORS[idx % len(CURRENT_COLORS)]

            # Škálování CHAN2 200×
            y_data = wf.y_v if self._is_ch2(spec) else wf.y_v / 300
            label = f"{spec.title} (÷200)" if self._is_ch2(spec) else spec.title

            # Surový signál
            self.wf_plot.plot(
                wf.x_us, y_data,
                pen=pg.mkPen(color, width=2),
                name=label,
            )

            # Vyhlazená křivka
            smooth = centered_moving_average(y_data, SMOOTHING_WINDOW)
            self.wf_plot.plot(
                wf.x_us, smooth,
                pen=pg.mkPen(color, width=1.2, style=QtCore.Qt.PenStyle.DashLine),
            )

            # Marker na peaku
            peak_idx = int(np.argmax(smooth))
            peak_x = float(wf.x_us[peak_idx])
            peak_y = float(smooth[peak_idx])
            peak_pen = pg.mkPen(color, width=1.0, style=QtCore.Qt.PenStyle.DashLine)
            vl = pg.InfiniteLine(pos=peak_x, angle=90, pen=peak_pen)
            hl = pg.InfiniteLine(pos=peak_y, angle=0, pen=peak_pen)
            self.wf_plot.addItem(vl)
            self.wf_plot.addItem(hl)
            self.wf_highlight_items.extend([vl, hl])

        if not has_data:
            self.wf_plot.setTitle(f"H5 sample #{h5_sid} — žádná data")
            return

        self.wf_plot.addLegend(offset=(10, 10))
        marker_str = " ★" if self.selected_index in self.marker_csv_indices else ""
        self.wf_plot.setTitle(
            f"H5 waveform #{h5_sid} (CSV index {self.selected_index}){marker_str}"
        )

        if previous_range is not None:
            self.wf_plot.setXRange(*previous_range[0], padding=0)
            self.wf_plot.setYRange(*previous_range[1], padding=0)
        else:
            self.wf_plot.autoRange()
            self.wf_pending_autorange = False

    # -- Navigace klávesnicí ---------------------------------------------------

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Left:
            if self.selected_index > 0:
                self._select_index(self.selected_index - 1)
            return
        elif event.key() == QtCore.Qt.Key.Key_Right:
            if self.selected_index < self.pair_count - 1:
                self._select_index(self.selected_index + 1)
            return
        elif event.key() == QtCore.Qt.Key.Key_S:
            self._toggle_skip_current()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.dataset is not None:
            self.dataset.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Zobrazení páru H5 + CSV souborů se synchronizovaným zvýrazněním.",
    )
    parser.add_argument("h5_file", help="Cesta k HDF5 souboru s waveformy.")
    parser.add_argument("csv_file", help="Cesta k CSV souboru s daty.")
    parser.add_argument("--markers", type=str, default=None,
                        help="Cesta k io_markers CSV souboru (výchozí: {h5_stem}_io_markers.csv).")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    viewer = PairViewer(args.h5_file, args.csv_file)
    viewer.load_io_markers(markers_path=args.markers, h5_path=args.h5_file)
    viewer._draw_csv_plot()  # překreslit s markery
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
