#!/usr/bin/env python3
"""
view_fw.py — Firmware log viewer.
Vizualizace $C zpráv ze dvou CSV logů detektorů v jednom scatter plotu.
Párování částic podle strojového času v rámci stejné sekundy ($TIME).
ROI výběr, crosshair na oba body páru.
"""

from __future__ import annotations

import argparse
import bisect
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
except ImportError:
    print("Chyba: vyžaduje pyqtgraph a PyQt5/PyQt6.")
    sys.exit(1)

pg.setConfigOptions(antialias=False)

DETECTOR_COLORS = ["#4488ff", "#ff8833"]  # modrá, oranžová
DEFAULT_PAIR_THRESHOLD = 5
DEFAULT_MIN_VALUE = 0


# ---------------------------------------------------------------------------
#  Datové struktury a parsování
# ---------------------------------------------------------------------------

@dataclass
class CEvent:
    """Jedna $C událost z logu detektoru."""
    timestamp: str       # $TIME kontext (ISO řetězec)
    machine_time: int    # strojový čas (2. pole v $C zprávě)
    val3: int            # 3. pole – vykreslí se na ose X
    val4: int            # 4. pole – vykreslí se na ose Y
    index: int           # pořadí události v souboru
    line_no: int = 0     # řádek v souboru (1-indexed)
    pair_idx: int = -1   # index párové události v druhém souboru (-1 = bez páru)


def parse_csv_log(path: str | Path) -> tuple[list[CEvent], str]:
    """Parsování CSV logu, extrakce $C událostí.

    Vrací (seznam událostí, informační řetězec o zařízení).
    """
    events: list[CEvent] = []
    current_time = ""
    idx = 0
    device_info = Path(path).name

    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith("$TIME,"):
                current_time = line.split(",", 1)[1]
            elif line.startswith("$C,"):
                parts = line.split(",")
                if len(parts) >= 4:
                    try:
                        events.append(CEvent(
                            timestamp=current_time,
                            machine_time=int(parts[1]),
                            val3=int(parts[2]),
                            val4=int(parts[3]),
                            index=idx,
                            line_no=line_no,
                        ))
                        idx += 1
                    except ValueError:
                        pass
            elif line.startswith("$DOS,"):
                device_info = line

    return events, device_info


# ---------------------------------------------------------------------------
#  Párování událostí mezi detektory
# ---------------------------------------------------------------------------

def pair_events(events_a: list[CEvent], events_b: list[CEvent],
                threshold: int = DEFAULT_PAIR_THRESHOLD) -> int:
    """Spárování událostí ze dvou detektorů.

    Algoritmus: v rámci stejné $TIME sekundy páruje události s nejbližším
    strojovým časem (greedy, od nejmenšího Δt). Typický Δt je 1–2 tiky.

    Vrací počet spárovaných dvojic.
    """
    for ev in events_a:
        ev.pair_idx = -1
    for ev in events_b:
        ev.pair_idx = -1

    # Seskupení podle $TIME timestampu
    by_ts_a: dict[str, list[int]] = {}
    for i, ev in enumerate(events_a):
        by_ts_a.setdefault(ev.timestamp, []).append(i)

    by_ts_b: dict[str, list[int]] = {}
    for i, ev in enumerate(events_b):
        by_ts_b.setdefault(ev.timestamp, []).append(i)

    paired = 0
    used_b: set[int] = set()

    for ts, indices_a in by_ts_a.items():
        if ts not in by_ts_b:
            continue

        indices_b = by_ts_b[ts]
        sorted_b = sorted(indices_b, key=lambda i: events_b[i].machine_time)
        b_mt = [events_b[i].machine_time for i in sorted_b]

        for idx_a in sorted(indices_a, key=lambda i: events_a[i].machine_time):
            mt_a = events_a[idx_a].machine_time
            pos = bisect.bisect_left(b_mt, mt_a)

            best_b = -1
            best_diff = threshold + 1

            for c in (pos - 1, pos):
                if 0 <= c < len(sorted_b):
                    real_b = sorted_b[c]
                    if real_b in used_b:
                        continue
                    d = abs(b_mt[c] - mt_a)
                    if d < best_diff:
                        best_diff = d
                        best_b = real_b

            if best_b >= 0 and best_diff <= threshold:
                events_a[idx_a].pair_idx = best_b
                events_b[best_b].pair_idx = idx_a
                used_b.add(best_b)
                paired += 1

    return paired


# ---------------------------------------------------------------------------
#  RawLogWidget – surové logy dvou souborů vedle sebe
# ---------------------------------------------------------------------------

class RawLogWidget(QtWidgets.QWidget):
    """Tab se surovými logy dvou souborů vedle sebe (lazy loading)."""

    def __init__(self, path_a: str, path_b: str | None,
                 name_a: str, name_b: str,
                 events: list[list[CEvent]],
                 parent=None) -> None:
        super().__init__(parent)
        self._path_a = path_a
        self._path_b = path_b
        self._name_a = name_a
        self._name_b = name_b
        self.events = events
        self.lists: list[QtWidgets.QListWidget] = []
        self._highlight_brushes = [
            QtGui.QBrush(QtGui.QColor(DETECTOR_COLORS[0]).darker(200)),
            QtGui.QBrush(QtGui.QColor(DETECTOR_COLORS[1]).darker(200)),
        ]
        self._prev_highlighted: list[tuple[int, int]] = []  # (det_idx, row)
        # timestamp → řádek (0-indexed) pro $TIME řádky v každém souboru
        self.time_line_map: list[dict[str, int]] = [{}, {}]
        # line_no → event index pro $C řádky v každém souboru
        self.line_to_event: list[dict[int, int]] = [{}, {}]
        self._event_callback = None
        self._loaded = False
        # Prázdný layout připravený pro lazy loading
        self._layout = QtWidgets.QHBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(8)
        self._placeholder = QtWidgets.QLabel("Načítání logů proběhne při přepnutí na tento tab...")
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._placeholder)

    def set_event_callback(self, callback) -> None:
        """Callback(det_idx, ev_idx) volaný při kliknutí na $C řádek."""
        self._event_callback = callback

    def ensure_loaded(self) -> None:
        """Načíst logy (volá se lazy při prvním zobrazení tabu)."""
        if self._loaded:
            return
        self._loaded = True
        if self._placeholder is not None:
            self._placeholder.deleteLater()
            self._placeholder = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = self._layout

        file_specs = [(self._path_a, self._name_a, DETECTOR_COLORS[0])]
        if self._path_b is not None:
            file_specs.append((self._path_b, self._name_b, DETECTOR_COLORS[1]))

        for det_idx, (path, name, color) in enumerate(file_specs):
            col = QtWidgets.QVBoxLayout()
            label = QtWidgets.QLabel(name)
            label.setStyleSheet(f"color: {color}; font-weight: bold;")
            col.addWidget(label)

            lw = QtWidgets.QListWidget()
            lw.setFont(QtGui.QFont("monospace", 9))
            lw.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

            with open(path, "r") as f:
                for line_no, line in enumerate(f, 1):
                    raw = line.rstrip()
                    item = QtWidgets.QListWidgetItem(raw)
                    if line.startswith("$C,"):
                        item.setForeground(QtGui.QBrush(QtGui.QColor(color)))
                    elif line.startswith("$TIME,"):
                        ts = line.strip().split(",", 1)[1]
                        self.time_line_map[det_idx][ts] = line_no - 1
                    lw.addItem(item)

            # Mapování line_no → event index
            for ev in self.events[det_idx]:
                self.line_to_event[det_idx][ev.line_no] = ev.index

            lw.itemClicked.connect(
                lambda item, di=det_idx: self._on_line_clicked(di, item)
            )

            col.addWidget(lw, stretch=1)
            layout.addLayout(col)
            self.lists.append(lw)

    def _on_line_clicked(self, det_idx: int, item: QtWidgets.QListWidgetItem) -> None:
        """Kliknutí na řádek v logu – pokud je to $C, vyber událost."""
        lw = self.lists[det_idx]
        row = lw.row(item)
        line_no = row + 1
        if line_no in self.line_to_event[det_idx]:
            ev_idx = self.line_to_event[det_idx][line_no]
            if self._event_callback:
                self._event_callback(det_idx, ev_idx)

    def highlight_event(self, det_idx: int, line_no: int,
                        timestamp: str = "",
                        pair_det_idx: int = -1, pair_line_no: int = -1) -> None:
        if not self._loaded:
            return  # logy ještě není načtené
        """Zvýraznit řádky události (a jejího páru) a vycentrovat.

        Pokud pár neexistuje ale timestamp je zadán, scrolluje druhý soubor
        na odpovídající $TIME blok.
        """
        # Smazat předchozí zvýraznění
        for di, row in self._prev_highlighted:
            if 0 <= di < len(self.lists):
                it = self.lists[di].item(row)
                if it is not None:
                    it.setBackground(QtGui.QBrush())
        self._prev_highlighted.clear()

        # Zvýraznit a scrollovat vybranou událost
        if 0 <= det_idx < len(self.lists) and line_no > 0:
            lw = self.lists[det_idx]
            row = line_no - 1
            if 0 <= row < lw.count():
                item = lw.item(row)
                item.setBackground(self._highlight_brushes[det_idx])
                lw.scrollToItem(
                    item,
                    QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter,
                )
                self._prev_highlighted.append((det_idx, row))

        # Zvýraznit a scrollovat pár
        if pair_det_idx >= 0 and pair_line_no > 0 and pair_det_idx < len(self.lists):
            lw = self.lists[pair_det_idx]
            row = pair_line_no - 1
            if 0 <= row < lw.count():
                item = lw.item(row)
                item.setBackground(self._highlight_brushes[pair_det_idx])
                lw.scrollToItem(
                    item,
                    QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter,
                )
                self._prev_highlighted.append((pair_det_idx, row))
        elif timestamp and 0 <= det_idx < len(self.lists):
            # Bez páru – scrollovat druhý soubor na stejný $TIME blok
            other_idx = 1 - det_idx
            if other_idx < len(self.lists) and timestamp in self.time_line_map[other_idx]:
                row = self.time_line_map[other_idx][timestamp]
                lw = self.lists[other_idx]
                if 0 <= row < lw.count():
                    lw.scrollToItem(
                        lw.item(row),
                        QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter,
                    )


# ---------------------------------------------------------------------------
#  FwLogViewer – hlavní okno (jeden společný graf)
# ---------------------------------------------------------------------------

class FwLogViewer(QtWidgets.QMainWindow):
    """Hlavní okno firmware log vieweru – jeden scatter plot pro oba detektory."""

    def __init__(self, path_a: str, path_b: str | None = None,
                 threshold: int = DEFAULT_PAIR_THRESHOLD,
                 min_value: int = DEFAULT_MIN_VALUE) -> None:
        super().__init__()
        self.setWindowTitle("Firmware Log Viewer")
        self.resize(1400, 900)
        self.threshold = threshold
        self.min_value = min_value
        self.path_a = path_a
        self.path_b = path_b
        self.single_mode = path_b is None

        # Parsování logů
        self.events: list[list[CEvent]] = [[], []]
        self.names: list[str] = ["", ""]
        self.events[0], _ = parse_csv_log(path_a)
        self.names[0] = Path(path_a).stem
        if path_b is not None:
            self.events[1], _ = parse_csv_log(path_b)
            self.names[1] = Path(path_b).stem

        # Párování
        self.pair_count = 0
        if not self.single_mode:
            self.pair_count = pair_events(self.events[0], self.events[1], threshold)

        # Scatter a crosshair items
        self.scatters: list[pg.ScatterPlotItem | None] = [None, None]
        self.crosshair_items: list = []
        self.roi: pg.PolyLineROI | None = None

        # Debounce timer pro ROI změny
        self._roi_timer = QtCore.QTimer()
        self._roi_timer.setSingleShot(True)
        self._roi_timer.setInterval(150)  # ms
        self._roi_timer.timeout.connect(self._apply_filters)

        self._build_ui()
        self._draw_scatter()
        self._apply_filters()
        self._update_status()

    # -- UI --

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # -- Ovládací prvky --
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QtWidgets.QLabel("Zobrazení:"))
        self.display_combo = QtWidgets.QComboBox()
        if self.single_mode:
            self.display_combo.addItem(self.names[0], "a")
        else:
            self.display_combo.addItem("Oba detektory", "both")
            self.display_combo.addItem(self.names[0], "a")
            self.display_combo.addItem(self.names[1], "b")
        self.display_combo.currentIndexChanged.connect(self._on_display_changed)
        controls.addWidget(self.display_combo)

        controls.addWidget(QtWidgets.QLabel("Filtr:"))
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItem("Všechny", "all")
        self.filter_combo.addItem("Spárované", "paired")
        self.filter_combo.addItem("Nespárované", "unpaired")
        self.filter_combo.currentIndexChanged.connect(self._apply_filters)
        controls.addWidget(self.filter_combo)

        self.roi_check = QtWidgets.QCheckBox("ROI výběr")
        self.roi_check.toggled.connect(self._toggle_roi)
        controls.addWidget(self.roi_check)

        self.roi_reset_button = QtWidgets.QPushButton("Reset ROI")
        self.roi_reset_button.setEnabled(False)
        self.roi_reset_button.clicked.connect(self._reset_roi)
        controls.addWidget(self.roi_reset_button)

        if not self.single_mode:
            controls.addWidget(QtWidgets.QLabel("ROI kanál:"))
            self.roi_channel_combo = QtWidgets.QComboBox()
            self.roi_channel_combo.addItem("Oba", "both")
            self.roi_channel_combo.addItem(self.names[0], "a")
            self.roi_channel_combo.addItem(self.names[1], "b")
            self.roi_channel_combo.currentIndexChanged.connect(self._apply_filters)
            controls.addWidget(self.roi_channel_combo)

            controls.addWidget(QtWidgets.QLabel("Práh [tiky]:"))
            self.threshold_spin = QtWidgets.QSpinBox()
            self.threshold_spin.setRange(1, 200)
            self.threshold_spin.setValue(self.threshold)
            self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
            controls.addWidget(self.threshold_spin)
        else:
            self.roi_channel_combo = None
            self.threshold_spin = None

        controls.addWidget(QtWidgets.QLabel("Min hodnota:"))
        self.min_value_spin = QtWidgets.QSpinBox()
        self.min_value_spin.setRange(0, 100000)
        self.min_value_spin.setValue(self.min_value)
        self.min_value_spin.setSingleStep(10)
        self.min_value_spin.setToolTip("Odfiltrovat události s max(val3, val4) < této hodnoty")
        self.min_value_spin.valueChanged.connect(self._on_min_value_changed)
        controls.addWidget(self.min_value_spin)

        self.info_label = QtWidgets.QLabel()
        controls.addWidget(self.info_label)
        controls.addStretch()
        layout.addLayout(controls)

        # -- QTabWidget --
        self.tab_widget = QtWidgets.QTabWidget()
        layout.addWidget(self.tab_widget, stretch=1)

        # -- Tab "Scatter" --
        scatter_page = QtWidgets.QWidget()
        scatter_layout = QtWidgets.QVBoxLayout(scatter_page)
        scatter_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#101418")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("left", "Hodnota 4")
        self.plot_widget.setLabel("bottom", "Hodnota 3")
        self.plot_widget.setTitle("$C události – scatter val3 vs val4")
        splitter.addWidget(self.plot_widget)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(4)

        self.count_label = QtWidgets.QLabel("Události: 0")
        side_layout.addWidget(self.count_label)

        nav = QtWidgets.QHBoxLayout()
        self.prev_button = QtWidgets.QPushButton("◀")
        self.prev_button.clicked.connect(self._prev_event)
        nav.addWidget(self.prev_button)
        self.next_button = QtWidgets.QPushButton("▶")
        self.next_button.clicked.connect(self._next_event)
        nav.addWidget(self.next_button)
        side_layout.addLayout(nav)

        self.event_list = QtWidgets.QListWidget()
        self.event_list.currentItemChanged.connect(self._on_event_selected)
        side_layout.addWidget(self.event_list, stretch=1)

        self.detail_label = QtWidgets.QLabel()
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #f0d060;"
        )
        self.detail_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        side_layout.addWidget(self.detail_label)

        splitter.addWidget(side)
        splitter.setSizes([1000, 350])
        scatter_layout.addWidget(splitter)

        self.tab_widget.addTab(scatter_page, "Scatter")

        # -- Tab "Logy" --
        if self.single_mode:
            self.raw_log_widget = RawLogWidget(
                self.path_a, None,
                self.names[0], "",
                self.events,
            )
        else:
            self.raw_log_widget = RawLogWidget(
                self.path_a, self.path_b,
                self.names[0], self.names[1],
                self.events,
            )
        self.raw_log_widget.set_event_callback(self._on_raw_log_event_clicked)
        self.tab_widget.addTab(self.raw_log_widget, "Logy")

        # Lazy loading raw logů při přepnutí na tab
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int) -> None:
        widget = self.tab_widget.widget(index)
        if isinstance(widget, RawLogWidget):
            widget.ensure_loaded()

    # -- Scatter plot --

    def _draw_scatter(self) -> None:
        """Překreslit scatter – body obou detektorů v jednom grafu.

        ● tečka  = spárovaná událost (v obou diodách)
        + křížek = nespárovaná událost (jen v jedné diodě)
        """
        # Zapamatovat ROI
        roi_active = self.roi_check.isChecked() and self.roi is not None

        self.plot_widget.clear()
        self._clear_crosshairs()
        self.scatters = [None, None]

        mode = self.display_combo.currentData()

        for det_idx in range(2):
            if mode == "a" and det_idx == 1:
                continue
            if mode == "b" and det_idx == 0:
                continue

            events = self.events[det_idx]
            color = DETECTOR_COLORS[det_idx]
            if not events:
                continue

            v3 = np.array([ev.val3 for ev in events], dtype=np.float64)
            v4 = np.array([ev.val4 for ev in events], dtype=np.float64)

            # Symbol na bod: tečka (●) pro spárované, křížek (+) pro nespárované
            symbols = ["o" if ev.pair_idx >= 0 else "+" for ev in events]
            sizes = np.array([5 if ev.pair_idx >= 0 else 7 for ev in events])

            scatter = pg.ScatterPlotItem(
                x=v3, y=v4,
                symbol=symbols,
                size=sizes,
                pen=pg.mkPen(color, width=0.5),
                brush=pg.mkBrush(color + "80"),
                name=self.names[det_idx],
            )
            scatter.sigClicked.connect(
                lambda item, pts, ev, di=det_idx: self._on_scatter_clicked(di, item, pts, ev)
            )
            self.plot_widget.addItem(scatter)
            self.scatters[det_idx] = scatter

        self.plot_widget.addLegend(offset=(10, 10))

        # Obnovit ROI
        if roi_active and self.roi is not None:
            self.plot_widget.addItem(self.roi)

        self.plot_widget.autoRange()

    # -- Filtry --

    def _on_display_changed(self) -> None:
        self._draw_scatter()
        self._apply_filters()

    def _on_threshold_changed(self, value: int) -> None:
        self.threshold = value
        self.pair_count = pair_events(self.events[0], self.events[1], value)
        self._draw_scatter()
        self._apply_filters()
        self._update_status()

    def _on_min_value_changed(self, value: int) -> None:
        self.min_value = value
        self._draw_scatter()
        self._apply_filters()

    def _on_roi_changed(self) -> None:
        """ROI se změnila – restartovat debounce timer."""
        self._roi_timer.start()

    def _apply_filters(self) -> None:
        """Aplikovat filtr (spárované/nespárované) + ROI, aktualizovat seznam a viditelnost."""
        mode = self.display_combo.currentData()
        filter_mode = self.filter_combo.currentData()
        roi_channel = self.roi_channel_combo.currentData() if self.roi_channel_combo else "both"

        polygon = None
        if self.roi_check.isChecked() and self.roi is not None:
            polygon = self._get_roi_polygon()

        # Shromáždit viditelné události: (det_idx, ev_index, event)
        visible: list[tuple[int, int, CEvent]] = []
        visible_set: set[tuple[int, int]] = set()

        has_roi = polygon is not None and len(polygon) >= 3
        # Který kanál ROI přímo filtruje (0, 1 nebo -1 = oba)
        roi_primary = -1
        if has_roi and roi_channel == "a":
            roi_primary = 0
        elif has_roi and roi_channel == "b":
            roi_primary = 1

        for det_idx in range(2):
            if mode == "a" and det_idx == 1:
                continue
            if mode == "b" and det_idx == 0:
                continue
            # Když ROI filtruje jen jeden kanál, druhý přeskočit
            # (páry se doplní níže)
            if roi_primary >= 0 and roi_primary != det_idx:
                continue

            apply_roi = has_roi and (roi_primary == -1 or roi_primary == det_idx)

            for ev in self.events[det_idx]:
                if self.min_value > 0 and max(ev.val3, ev.val4) < self.min_value:
                    continue
                if filter_mode == "paired" and ev.pair_idx < 0:
                    continue
                if filter_mode == "unpaired" and ev.pair_idx >= 0:
                    continue
                if apply_roi:
                    if not self._point_in_polygon(ev.val3, ev.val4, polygon):
                        continue
                visible.append((det_idx, ev.index, ev))
                visible_set.add((det_idx, ev.index))

        # Když ROI filtruje jeden kanál, přidat komplementární páry z druhého
        if has_roi and roi_primary >= 0:
            other_idx = 1 - roi_primary
            # Zkontrolovat, že druhý kanál je zobrazený
            if not (mode == "a" and other_idx == 1) and not (mode == "b" and other_idx == 0):
                for di, ei, ev in list(visible):
                    if di == roi_primary and ev.pair_idx >= 0:
                        pev = self.events[other_idx][ev.pair_idx]
                        key = (other_idx, pev.index)
                        if key not in visible_set:
                            # Párová událost projde jen filtr paired/unpaired
                            if filter_mode == "paired" and pev.pair_idx < 0:
                                continue
                            if filter_mode == "unpaired" and pev.pair_idx >= 0:
                                continue
                            visible.append((other_idx, pev.index, pev))
                            visible_set.add(key)

        self._populate_event_list(visible)
        self._update_scatter_visibility(visible_set)

        # Info – počítat i kolik prošlo min filtrem
        total = sum(
            len(self.events[d]) for d in range(2)
            if not (mode == "a" and d == 1) and not (mode == "b" and d == 0)
        )
        if self.min_value > 0:
            above_min = sum(
                1 for d in range(2)
                if not (mode == "a" and d == 1) and not (mode == "b" and d == 0)
                for ev in self.events[d]
                if max(ev.val3, ev.val4) >= self.min_value
            )
            self.info_label.setText(
                f"Párů: {self.pair_count} | "
                f"Nad min: {above_min} / {total} | "
                f"Filtrováno: {len(visible)}"
            )
        else:
            self.info_label.setText(
                f"Párů: {self.pair_count} | Filtrováno: {len(visible)} / {total}"
            )

    def _populate_event_list(self, visible: list[tuple[int, int, CEvent]]) -> None:
        self.event_list.blockSignals(True)
        self.event_list.clear()
        for det_idx, ev_idx, ev in visible:
            other_idx = 1 - det_idx
            if ev.pair_idx >= 0:
                pev = self.events[other_idx][ev.pair_idx]
                dt = abs(ev.machine_time - pev.machine_time)
                pair_text = f" ↔ [{pev.val3},{pev.val4}] Δ{dt}"
            else:
                pair_text = " ✗"
            ts = ev.timestamp[-8:] if len(ev.timestamp) >= 8 else ev.timestamp
            short_name = self.names[det_idx][-4:]
            text = (
                f"{short_name} #{ev_idx} {ts} mt={ev.machine_time} "
                f"[{ev.val3},{ev.val4}]{pair_text}"
            )
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, (det_idx, ev_idx))
            # Barva textu podle detektoru
            item.setForeground(QtGui.QBrush(QtGui.QColor(DETECTOR_COLORS[det_idx])))
            self.event_list.addItem(item)
        self.event_list.blockSignals(False)
        self.count_label.setText(f"Vybrané: {len(visible)}")

    def _update_scatter_visibility(self, visible_set: set[tuple[int, int]]) -> None:
        """Skrýt body mimo filtr/ROI (velikost 0), zobrazit viditelné."""
        for det_idx in range(2):
            scatter = self.scatters[det_idx]
            if scatter is None:
                continue
            color = DETECTOR_COLORS[det_idx]
            events = self.events[det_idx]
            brushes = []
            pens = []
            sizes = []
            for ev in events:
                if (det_idx, ev.index) in visible_set:
                    brushes.append(pg.mkBrush(color + "80"))
                    pens.append(pg.mkPen(color, width=0.5))
                    sizes.append(5 if ev.pair_idx >= 0 else 7)
                else:
                    brushes.append(pg.mkBrush("#00000000"))
                    pens.append(pg.mkPen("#00000000", width=0))
                    sizes.append(0)
            scatter.setBrush(brushes)
            scatter.setPen(pens)
            scatter.setSize(sizes)

    # -- ROI --

    def _make_roi_points(self) -> list[list[float]]:
        vr = self.plot_widget.viewRange()
        xc = (vr[0][0] + vr[0][1]) / 2
        yc = (vr[1][0] + vr[1][1]) / 2
        hw = (vr[0][1] - vr[0][0]) * 0.15
        hh = (vr[1][1] - vr[1][0]) * 0.15
        return [
            [xc - hw, yc - hh],
            [xc + hw, yc - hh],
            [xc + hw, yc + hh],
            [xc - hw, yc + hh],
        ]

    def _toggle_roi(self, enabled: bool) -> None:
        if enabled:
            if self.roi is None:
                self.roi = pg.PolyLineROI(
                    self._make_roi_points(), closed=True,
                    pen=pg.mkPen("r", width=2),
                )
                self.roi.sigRegionChanged.connect(self._on_roi_changed)
            self.plot_widget.addItem(self.roi)
            self.roi_reset_button.setEnabled(True)
        else:
            if self.roi is not None:
                self.plot_widget.removeItem(self.roi)
            self.roi_reset_button.setEnabled(False)
        self._apply_filters()

    def _reset_roi(self) -> None:
        if self.roi is not None:
            self.plot_widget.removeItem(self.roi)
            self.roi = None
        if self.roi_check.isChecked():
            self.roi = pg.PolyLineROI(
                self._make_roi_points(), closed=True,
                pen=pg.mkPen("r", width=2),
            )
            self.roi.sigRegionChanged.connect(self._on_roi_changed)
            self.plot_widget.addItem(self.roi)
        self._apply_filters()

    def _get_roi_polygon(self) -> np.ndarray | None:
        if self.roi is None:
            return None
        state = self.roi.getState()
        local_pts = [h["pos"] if isinstance(h, dict) else h for h in state["points"]]
        mapped = [self.roi.mapToParent(p) for p in local_pts]
        return np.array([[p.x(), p.y()] for p in mapped])

    @staticmethod
    def _point_in_polygon(px: float, py: float, polygon: np.ndarray) -> bool:
        """Ray-casting test bodu uvnitř polygonu."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    # -- Crosshair a interakce --

    def _clear_crosshairs(self) -> None:
        for item in self.crosshair_items:
            self.plot_widget.removeItem(item)
        self.crosshair_items.clear()

    def _show_crosshairs(self, det_idx: int, ev_idx: int) -> None:
        """Zobrazit crosshair na vybraný bod a jeho pár (pokud existuje)."""
        self._clear_crosshairs()

        ev = self.events[det_idx][ev_idx]
        color = DETECTOR_COLORS[det_idx]

        # Crosshair na vybraný bod
        pen = pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DashLine)
        vl = pg.InfiniteLine(pos=ev.val3, angle=90, pen=pen)
        hl = pg.InfiniteLine(pos=ev.val4, angle=0, pen=pen)
        self.plot_widget.addItem(vl)
        self.plot_widget.addItem(hl)
        self.crosshair_items.extend([vl, hl])

        marker = pg.ScatterPlotItem(
            x=[ev.val3], y=[ev.val4],
            pen=pg.mkPen("#ffffff", width=2),
            brush=pg.mkBrush(color),
            size=14, symbol="crosshair",
        )
        marker.setZValue(100)
        self.plot_widget.addItem(marker)
        self.crosshair_items.append(marker)

        # Crosshair na párový bod (pokud existuje)
        if ev.pair_idx >= 0:
            other_idx = 1 - det_idx
            pev = self.events[other_idx][ev.pair_idx]
            other_color = DETECTOR_COLORS[other_idx]

            pen2 = pg.mkPen(other_color, width=1, style=QtCore.Qt.PenStyle.DashLine)
            vl2 = pg.InfiniteLine(pos=pev.val3, angle=90, pen=pen2)
            hl2 = pg.InfiniteLine(pos=pev.val4, angle=0, pen=pen2)
            self.plot_widget.addItem(vl2)
            self.plot_widget.addItem(hl2)
            self.crosshair_items.extend([vl2, hl2])

            marker2 = pg.ScatterPlotItem(
                x=[pev.val3], y=[pev.val4],
                pen=pg.mkPen("#ffffff", width=2),
                brush=pg.mkBrush(other_color),
                size=14, symbol="crosshair",
            )
            marker2.setZValue(100)
            self.plot_widget.addItem(marker2)
            self.crosshair_items.append(marker2)

    def _highlight_in_raw_log(self, det_idx: int, ev_idx: int) -> None:
        """Zvýraznit řádek události (a páru) v raw log tabu."""
        ev = self.events[det_idx][ev_idx]
        pair_det = -1
        pair_ln = -1
        if ev.pair_idx >= 0:
            other = 1 - det_idx
            pev = self.events[other][ev.pair_idx]
            pair_det = other
            pair_ln = pev.line_no
        self.raw_log_widget.highlight_event(
            det_idx, ev.line_no, ev.timestamp, pair_det, pair_ln,
        )

    def _on_raw_log_event_clicked(self, det_idx: int, ev_idx: int) -> None:
        """Kliknutí na $C řádek v raw logu – vyznačit ve scatteru."""
        self._show_crosshairs(det_idx, ev_idx)
        self._highlight_in_list(det_idx, ev_idx)
        self._show_detail(det_idx, ev_idx)
        self._highlight_in_raw_log(det_idx, ev_idx)

    def _show_detail(self, det_idx: int, ev_idx: int) -> None:
        """Zobrazit detail události a jejího páru."""
        ev = self.events[det_idx][ev_idx]
        lines = [
            f"{self.names[det_idx]} #{ev_idx}",
            f"  Čas: {ev.timestamp}",
            f"  Strojový čas: {ev.machine_time}",
            f"  val3={ev.val3}  val4={ev.val4}",
        ]
        if not self.single_mode and ev.pair_idx >= 0:
            other_idx = 1 - det_idx
            pev = self.events[other_idx][ev.pair_idx]
            dt = abs(ev.machine_time - pev.machine_time)
            lines.append(f"Pár: {self.names[other_idx]} #{pev.index}")
            lines.append(f"  Strojový čas: {pev.machine_time} (Δ={dt})")
            lines.append(f"  val3={pev.val3}  val4={pev.val4}")
        elif not self.single_mode:
            lines.append("Pár: nenalezen")
        self.detail_label.setText("\n".join(lines))

    def _on_scatter_clicked(self, det_idx: int, scatter_item, points, ev) -> None:
        if len(points) == 0:
            return
        point = points[0]
        ev_idx = point.index()
        if 0 <= ev_idx < len(self.events[det_idx]):
            self._show_crosshairs(det_idx, ev_idx)
            self._highlight_in_list(det_idx, ev_idx)
            self._show_detail(det_idx, ev_idx)
            self._highlight_in_raw_log(det_idx, ev_idx)

    def _highlight_in_list(self, det_idx: int, ev_idx: int) -> None:
        for row in range(self.event_list.count()):
            item = self.event_list.item(row)
            data = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if data == (det_idx, ev_idx):
                self.event_list.blockSignals(True)
                self.event_list.setCurrentRow(row)
                self.event_list.blockSignals(False)
                break

    def _on_event_selected(self, current, _prev=None) -> None:
        if current is None:
            return
        det_idx, ev_idx = current.data(QtCore.Qt.ItemDataRole.UserRole)
        self._show_crosshairs(det_idx, ev_idx)
        self._show_detail(det_idx, ev_idx)
        self._highlight_in_raw_log(det_idx, ev_idx)

    def _prev_event(self) -> None:
        row = self.event_list.currentRow()
        if row > 0:
            self.event_list.setCurrentRow(row - 1)

    def _next_event(self) -> None:
        row = self.event_list.currentRow()
        if row < self.event_list.count() - 1:
            self.event_list.setCurrentRow(row + 1)

    # -- Status --

    def _update_status(self) -> None:
        if self.single_mode:
            self.statusBar().showMessage(
                f"{self.names[0]}: {len(self.events[0])} událostí"
            )
        else:
            self.statusBar().showMessage(
                f"{self.names[0]}: {len(self.events[0])} | "
                f"{self.names[1]}: {len(self.events[1])} | "
                f"Spárováno: {self.pair_count} | Práh: {self.threshold}"
            )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Left:
            self._prev_event()
            return
        elif event.key() == QtCore.Qt.Key.Key_Right:
            self._next_event()
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Firmware log viewer – scatter $C zpráv s párováním částic.",
    )
    parser.add_argument(
        "files", nargs="+",
        help="Cesty k CSV log souborům (1 nebo 2 detektory).",
    )
    parser.add_argument(
        "--threshold", type=int, default=DEFAULT_PAIR_THRESHOLD,
        help=f"Práh pro párování strojového času [tiky] (výchozí: {DEFAULT_PAIR_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-value", type=int, default=DEFAULT_MIN_VALUE,
        help="Minimální hodnota max(val3, val4) pro zobrazení události (výchozí: 0 = bez filtru).",
    )
    args = parser.parse_args()
    if len(args.files) > 2:
        parser.error("Maximálně 2 soubory.")

    app = QtWidgets.QApplication(sys.argv)
    path_b = args.files[1] if len(args.files) >= 2 else None
    viewer = FwLogViewer(args.files[0], path_b, args.threshold, args.min_value)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
