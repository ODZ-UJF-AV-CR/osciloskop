#!/usr/bin/env python3

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


pg.setConfigOptions(antialias=False)

CURRENT_COLORS = ["#ffd166", "#4cc9f0", "#ef476f", "#95d67b", "#f78c6b", "#c77dff"]
MAX_RENDER_POINTS = 1500


@dataclass(frozen=True)
class Waveform:
    sample_id: int
    series_name: str
    x_us: np.ndarray
    y_v: np.ndarray


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    title: str
    sample_ids: tuple[int, ...]
    container_path: str | None = None
    metadata_in_attrs: bool = False
    sample_suffix: str = ""
    xinc_key: str = "XINC"
    yinc_key: str = "YINC"
    xorigin_key: str = "XORIGIN"
    yorigin_key: str = "YORIGIN"


class H5WaveformDataset:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")
        self.waveform_cache: dict[tuple[int, str], Waveform] = {}
        self.display_waveform_cache: dict[tuple[int, str, int], Waveform] = {}
        self.series_specs = self._discover_series_specs()
        self.series_by_name = {spec.name: spec for spec in self.series_specs}
        self.sample_ids = sorted({sample_id for spec in self.series_specs for sample_id in spec.sample_ids})

    def close(self) -> None:
        if getattr(self, "handle", None) is not None:
            self.handle.close()
            self.handle = None

    def __del__(self) -> None:
        self.close()

    def _scalar(self, key: str, default: float = 0.0) -> float:
        if key not in self.handle:
            return default
        value = np.asarray(self.handle[key])
        if value.size == 0:
            return default
        return float(value.reshape(-1)[0])

    def _container_scalar(self, container, key: str, metadata_in_attrs: bool, default: float = 0.0) -> float:
        if metadata_in_attrs:
            if key not in container.attrs:
                return default
            value = np.asarray(container.attrs[key])
        else:
            if key not in container:
                return default
            value = np.asarray(container[key])
        if value.size == 0:
            return default
        return float(value.reshape(-1)[0])

    def _discover_sample_ids_in_container(self, container) -> list[int]:
        sample_ids: set[int] = set()
        for key in container.keys():
            if key.isdigit():
                sample_ids.add(int(key))
        return sorted(sample_ids)

    def _discover_series_specs(self) -> list[SeriesSpec]:
        grouped_specs: list[SeriesSpec] = []
        for key, item in self.handle.items():
            if isinstance(item, h5py.Group):
                sample_ids = tuple(self._discover_sample_ids_in_container(item))
                if sample_ids:
                    grouped_specs.append(
                        SeriesSpec(
                            name=key,
                            title=key,
                            sample_ids=sample_ids,
                            container_path=key,
                            metadata_in_attrs=True,
                        )
                    )

        if grouped_specs:
            return grouped_specs

        legacy_specs: list[SeriesSpec] = []
        normal_ids = tuple(sorted(int(key) for key in self.handle.keys() if key.isdigit()))
        raw_ids = tuple(sorted(int(key[:-1]) for key in self.handle.keys() if key.endswith("R") and key[:-1].isdigit()))

        if normal_ids:
            legacy_specs.append(SeriesSpec(name="normal", title="NORMAL", sample_ids=normal_ids))
        if raw_ids:
            legacy_specs.append(
                SeriesSpec(
                    name="raw",
                    title="RAW",
                    sample_ids=raw_ids,
                    sample_suffix="R",
                    xinc_key="XINCR",
                    yinc_key="YINCR",
                    xorigin_key="XORIGINR",
                    yorigin_key="YORIGINR",
                )
            )

        return legacy_specs

    def _time_string(self) -> str:
        if "START_TIME" in self.handle.attrs:
            start_time = self.handle.attrs["START_TIME"]
            end_time = self.handle.attrs.get("END_TIME", "")
            return f"{start_time} -> {end_time}".strip()
        for key in ("THETIME", "TIME"):
            if key in self.handle:
                timestamp = self._scalar(key, default=0.0)
                try:
                    return dt.datetime.utcfromtimestamp(timestamp).isoformat(sep=" ", timespec="seconds")
                except (OverflowError, OSError, ValueError):
                    return str(timestamp)
        return "n/a"

    def summary_text(self) -> str:
        lines = [
            f"Soubor: {self.path.name}",
            f"Vzorky: {len(self.sample_ids)}",
            f"Série: {', '.join(spec.title for spec in self.series_specs) if self.series_specs else 'žádné'}",
            f"Čas: {self._time_string()}",
        ]

        for attr_key in ("SCOPE_NAME", "IP", "MEASUREMENT_NAME"):
            if attr_key in self.handle.attrs:
                lines.append(f"{attr_key}: {self.handle.attrs[attr_key]}")

        for key in ("NSPART", "TRIG", "TRIGR"):
            if key in self.handle:
                lines.append(f"{key}: {self._scalar(key)}")

        for spec in self.series_specs:
            lines.append(f"{spec.title}: {len(spec.sample_ids)} vzorků")
            container = self.handle if spec.container_path is None else self.handle[spec.container_path]
            for key in ("XINC", "YINC", "XORIGIN", "YORIGIN", "TRIG_LEVEL", "TRIG_CHANNEL"):
                if spec.metadata_in_attrs and key in container.attrs:
                    lines.append(f"{spec.title}.{key}: {container.attrs[key]}")
                elif not spec.metadata_in_attrs and key in container:
                    lines.append(f"{spec.title}.{key}: {self._container_scalar(container, key, False)}")

        return "\n".join(lines)

    def waveform(self, sample_id: int, series_name: str) -> Waveform | None:
        cache_key = (sample_id, series_name)
        if cache_key in self.waveform_cache:
            return self.waveform_cache[cache_key]

        spec = self.series_by_name[series_name]
        container = self.handle if spec.container_path is None else self.handle[spec.container_path]
        dataset_key = f"{sample_id}{spec.sample_suffix}"
        if dataset_key not in container:
            return None

        data = np.asarray(container[dataset_key], dtype=np.float64).reshape(-1)
        yorigin = self._container_scalar(container, spec.yorigin_key, spec.metadata_in_attrs)
        yinc = self._container_scalar(container, spec.yinc_key, spec.metadata_in_attrs, default=1.0)
        xinc = self._container_scalar(container, spec.xinc_key, spec.metadata_in_attrs, default=1.0)
        xorigin = self._container_scalar(container, spec.xorigin_key, spec.metadata_in_attrs, default=0.0)

        y_v = (data - 128.0 - yorigin) * yinc
        x_us = np.arange(data.size, dtype=np.float64) * xinc * 1e6
        x_us += xorigin * 1e6

        waveform = Waveform(sample_id=sample_id, series_name=series_name, x_us=x_us, y_v=y_v)
        self.waveform_cache[cache_key] = waveform
        return waveform

    def display_waveform(self, sample_id: int, series_name: str, max_points: int = MAX_RENDER_POINTS) -> Waveform | None:
        cache_key = (sample_id, series_name, max_points)
        if cache_key in self.display_waveform_cache:
            return self.display_waveform_cache[cache_key]

        waveform = self.waveform(sample_id, series_name)
        if waveform is None:
            return None

        x_us, y_v = self._decimate_for_display(waveform.x_us, waveform.y_v, max_points)
        display_waveform = Waveform(
            sample_id=waveform.sample_id,
            series_name=waveform.series_name,
            x_us=x_us,
            y_v=y_v,
        )
        self.display_waveform_cache[cache_key] = display_waveform
        return display_waveform

    @staticmethod
    def _decimate_for_display(x_us: np.ndarray, y_v: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
        point_count = x_us.size
        if point_count <= max_points or max_points < 4:
            return x_us, y_v

        bucket_count = max(1, max_points)
        bucket_size = int(np.ceil(point_count / bucket_count))

        x_parts: list[float] = []
        y_parts: list[float] = []

        for start in range(0, point_count, bucket_size):
            end = min(point_count, start + bucket_size)
            x_bucket = x_us[start:end]
            y_bucket = y_v[start:end]
            if x_bucket.size == 0:
                continue

            x_parts.append(float(np.mean(x_bucket)))
            y_parts.append(float(np.mean(y_bucket)))

        return np.asarray(x_parts, dtype=np.float64), np.asarray(y_parts, dtype=np.float64)


class WaveformViewer(QtWidgets.QMainWindow):
    def __init__(self, initial_path: str | None = None, initial_sample: int | None = None) -> None:
        super().__init__()
        self.setWindowTitle("HDF5 Waveform Viewer")
        self.resize(1400, 900)

        self.dataset: H5WaveformDataset | None = None
        self.current_position = 0
        self.locked_samples: OrderedDict[int, QtGui.QColor] = OrderedDict()
        self.main_plot: pg.PlotWidget | None = None
        self.empty_plot_label: QtWidgets.QLabel | None = None
        self.pending_autorange = True

        self._build_ui()

        if initial_path:
            self.load_file(initial_path, initial_sample)
        else:
            self._set_empty_state()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setSpacing(8)
        root_layout.addLayout(controls_layout)

        self.open_button = QtWidgets.QPushButton("Otevřít H5")
        self.open_button.clicked.connect(self._open_file_dialog)
        controls_layout.addWidget(self.open_button)

        controls_layout.addWidget(QtWidgets.QLabel("Vzorek"))

        self.sample_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sample_slider.setTracking(True)
        self.sample_slider.valueChanged.connect(self._on_slider_changed)
        controls_layout.addWidget(self.sample_slider, stretch=1)

        self.sample_spinbox = QtWidgets.QSpinBox()
        self.sample_spinbox.setKeyboardTracking(False)
        self.sample_spinbox.valueChanged.connect(self._on_spinbox_changed)
        controls_layout.addWidget(self.sample_spinbox)

        self.current_label = QtWidgets.QLabel("0 / 0")
        controls_layout.addWidget(self.current_label)

        self.lock_button = QtWidgets.QPushButton("Zamknout aktuální")
        self.lock_button.clicked.connect(self._toggle_current_lock)
        controls_layout.addWidget(self.lock_button)

        self.remove_button = QtWidgets.QPushButton("Odemknout vybraný")
        self.remove_button.clicked.connect(self._remove_selected_lock)
        controls_layout.addWidget(self.remove_button)

        self.clear_button = QtWidgets.QPushButton("Vymazat zamknuté")
        self.clear_button.clicked.connect(self._clear_locks)
        controls_layout.addWidget(self.clear_button)

        self.path_label = QtWidgets.QLabel()
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        root_layout.addWidget(self.path_label)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        self.plot_panel = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_panel)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(8)
        splitter.addWidget(self.plot_panel)

        side_panel = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)

        side_layout.addWidget(QtWidgets.QLabel("Uzamčené vzorky"))

        self.locked_list = QtWidgets.QListWidget()
        self.locked_list.itemDoubleClicked.connect(self._jump_to_locked_item)
        side_layout.addWidget(self.locked_list, stretch=1)

        side_layout.addWidget(QtWidgets.QLabel("Metadata"))

        self.metadata_text = QtWidgets.QPlainTextEdit()
        self.metadata_text.setReadOnly(True)
        side_layout.addWidget(self.metadata_text, stretch=1)

        splitter.addWidget(side_panel)
        splitter.setSizes([1100, 300])

        self.statusBar().showMessage("Vyber HDF5 soubor.")

    def _configure_plot(self, plot: pg.PlotWidget, title: str) -> None:
        plot.setBackground("#101418")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("left", "Amplituda", units="V")
        plot.setLabel("bottom", "Čas", units="us")
        plot.setTitle(title)
        plot.setClipToView(True)

    def _clear_plot_layout(self) -> None:
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.main_plot = None
        self.empty_plot_label = None

    def _rebuild_plots(self) -> None:
        self._clear_plot_layout()
        if self.dataset is None or not self.dataset.series_specs:
            self.empty_plot_label = QtWidgets.QLabel("Nenahrán žádný waveform dataset.")
            self.empty_plot_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.plot_layout.addWidget(self.empty_plot_label, stretch=1)
            return

        self.main_plot = pg.PlotWidget()
        self._configure_plot(self.main_plot, "Waveforms")
        self.main_plot.addLegend(offset=(10, 10))
        self.plot_layout.addWidget(self.main_plot, stretch=1)
        self.pending_autorange = True

    def _set_empty_state(self) -> None:
        self.path_label.setText("Soubor: nenačteno")
        self.metadata_text.setPlainText("")
        self.sample_slider.setEnabled(False)
        self.sample_spinbox.setEnabled(False)
        self.lock_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.locked_list.setEnabled(False)
        self.current_label.setText("0 / 0")
        self._rebuild_plots()

    def _set_loaded_state(self) -> None:
        self.sample_slider.setEnabled(True)
        self.sample_spinbox.setEnabled(True)
        self.lock_button.setEnabled(True)
        self.remove_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.locked_list.setEnabled(True)

    def _open_file_dialog(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Vyber HDF5 soubor",
            str(Path.cwd()),
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if filename:
            self.load_file(filename)

    def load_file(self, path: str, initial_sample: int | None = None) -> None:
        try:
            new_dataset = H5WaveformDataset(path)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Nelze otevřít soubor", str(exc))
            return

        if not new_dataset.sample_ids:
            new_dataset.close()
            QtWidgets.QMessageBox.warning(self, "Prázdný dataset", "Soubor neobsahuje žádné waveformy.")
            return

        if self.dataset is not None:
            self.dataset.close()

        self.dataset = new_dataset
        self.locked_samples.clear()
        self.locked_list.clear()
        self._set_loaded_state()
        self._rebuild_plots()

        self.path_label.setText(f"Soubor: {self.dataset.path}")
        self.metadata_text.setPlainText(self.dataset.summary_text())

        minimum = self.dataset.sample_ids[0]
        maximum = self.dataset.sample_ids[-1]

        self.sample_slider.blockSignals(True)
        self.sample_slider.setRange(0, len(self.dataset.sample_ids) - 1)
        self.sample_slider.blockSignals(False)

        self.sample_spinbox.blockSignals(True)
        self.sample_spinbox.setRange(minimum, maximum)
        self.sample_spinbox.setSingleStep(1)
        self.sample_spinbox.blockSignals(False)

        target_position = 0 if initial_sample is None else self._position_for_sample(initial_sample)
        self._set_position(target_position)
        self.statusBar().showMessage(f"Načteno {self.dataset.path}")

    def _position_for_sample(self, requested_sample_id: int) -> int:
        assert self.dataset is not None
        sample_ids = self.dataset.sample_ids
        index = bisect.bisect_left(sample_ids, requested_sample_id)
        if index <= 0:
            return 0
        if index >= len(sample_ids):
            return len(sample_ids) - 1
        before = sample_ids[index - 1]
        after = sample_ids[index]
        if abs(before - requested_sample_id) <= abs(after - requested_sample_id):
            return index - 1
        return index

    def _current_sample_id(self) -> int:
        assert self.dataset is not None
        return self.dataset.sample_ids[self.current_position]

    def _set_position(self, position: int) -> None:
        if self.dataset is None:
            return
        position = max(0, min(position, len(self.dataset.sample_ids) - 1))
        self.current_position = position
        sample_id = self._current_sample_id()

        self.sample_slider.blockSignals(True)
        self.sample_slider.setValue(position)
        self.sample_slider.blockSignals(False)

        self.sample_spinbox.blockSignals(True)
        self.sample_spinbox.setValue(sample_id)
        self.sample_spinbox.blockSignals(False)

        self.current_label.setText(f"{position + 1} / {len(self.dataset.sample_ids)}")
        self._update_lock_button_text()
        self._refresh_plots()

    def _on_slider_changed(self, position: int) -> None:
        self._set_position(position)

    def _on_spinbox_changed(self, sample_id: int) -> None:
        if self.dataset is None:
            return
        self._set_position(self._position_for_sample(sample_id))

    def _next_lock_color(self) -> QtGui.QColor:
        hue = (len(self.locked_samples) * 53) % 360
        return QtGui.QColor.fromHsv(hue, 180, 255)

    def _toggle_current_lock(self) -> None:
        if self.dataset is None:
            return
        sample_id = self._current_sample_id()
        if sample_id in self.locked_samples:
            del self.locked_samples[sample_id]
        else:
            self.locked_samples[sample_id] = self._next_lock_color()

        self._refresh_locked_list()
        self._update_lock_button_text()
        self._refresh_plots()

    def _remove_selected_lock(self) -> None:
        item = self.locked_list.currentItem()
        if item is None:
            return
        sample_id = int(item.data(QtCore.Qt.ItemDataRole.UserRole))
        if sample_id in self.locked_samples:
            del self.locked_samples[sample_id]
            self._refresh_locked_list()
            self._update_lock_button_text()
            self._refresh_plots()

    def _clear_locks(self) -> None:
        if not self.locked_samples:
            return
        self.locked_samples.clear()
        self._refresh_locked_list()
        self._update_lock_button_text()
        self._refresh_plots()

    def _refresh_locked_list(self) -> None:
        self.locked_list.clear()
        for sample_id, color in self.locked_samples.items():
            item = QtWidgets.QListWidgetItem(f"Vzorek {sample_id}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, sample_id)
            item.setForeground(QtGui.QBrush(color))
            self.locked_list.addItem(item)

    def _jump_to_locked_item(self, item: QtWidgets.QListWidgetItem) -> None:
        sample_id = int(item.data(QtCore.Qt.ItemDataRole.UserRole))
        self._set_position(self._position_for_sample(sample_id))

    def _update_lock_button_text(self) -> None:
        if self.dataset is None:
            self.lock_button.setText("Zamknout aktuální")
            return
        if self._current_sample_id() in self.locked_samples:
            self.lock_button.setText("Odemknout aktuální")
        else:
            self.lock_button.setText("Zamknout aktuální")

    def _current_pen(self, index: int):
        return pg.mkPen(CURRENT_COLORS[index % len(CURRENT_COLORS)], width=2.6)

    def _draw_plot(self, plot: pg.PlotWidget) -> None:
        if self.dataset is None:
            plot.clear()
            plot.setTitle("Waveforms")
            return

        previous_range = None
        if not self.pending_autorange:
            previous_range = plot.plotItem.vb.viewRange()

        plot.clear()
        plot.addLegend(offset=(10, 10))

        current_sample_id = self._current_sample_id()

        for sample_id, lock_color in self.locked_samples.items():
            for spec in self.dataset.series_specs:
                waveform = self.dataset.display_waveform(sample_id, spec.name)
                if waveform is None:
                    continue
                pen = pg.mkPen(color=lock_color, width=1.2, style=QtCore.Qt.PenStyle.DashLine)
                plot.plot(waveform.x_us, waveform.y_v, pen=pen, name=f"{spec.title} locked {sample_id}")

        for index, spec in enumerate(self.dataset.series_specs):
            current_waveform = self.dataset.display_waveform(current_sample_id, spec.name)
            if current_waveform is None:
                continue
            plot.plot(
                current_waveform.x_us,
                current_waveform.y_v,
                pen=self._current_pen(index),
                name=f"{spec.title} current {current_sample_id}",
            )

        if previous_range is not None:
            plot.setXRange(*previous_range[0], padding=0)
            plot.setYRange(*previous_range[1], padding=0)
        else:
            plot.autoRange()
            self.pending_autorange = False

        plot.setTitle(
            f"Waveforms | aktuální vzorek {current_sample_id} | série {len(self.dataset.series_specs)} | uzamčeno {len(self.locked_samples)}"
        )

    def _refresh_plots(self) -> None:
        if self.dataset is None or self.main_plot is None:
            return
        self._draw_plot(self.main_plot)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.dataset is not None:
            self.dataset.close()
        super().closeEvent(event)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyQtGraph viewer pro waveformy uložené v HDF5.")
    parser.add_argument("path", nargs="?", help="Cesta k HDF5 souboru.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Vzorek, na kterém se má viewer po startu otevřít.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    app = QtWidgets.QApplication(sys.argv)
    viewer = WaveformViewer(initial_path=args.path, initial_sample=args.sample)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
