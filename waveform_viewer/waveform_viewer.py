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

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


if HAS_PYQTGRAPH:
    pg.setConfigOptions(antialias=False)

CURRENT_COLORS = ["#ffd166", "#4cc9f0", "#ef476f", "#95d67b", "#f78c6b", "#c77dff"]
MAX_RENDER_POINTS = 1500
SMOOTHING_WINDOW = 31


@dataclass(frozen=True)
class Waveform:
    sample_id: int
    series_name: str
    x_us: np.ndarray
    y_v: np.ndarray
    xinc: float = 1.0


def centered_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Centrovaný klouzavý průměr."""
    if window <= 1:
        return data.copy()
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')


def _find_peak_region(y: np.ndarray, peak_idx: int, baseline: float) -> tuple[int, int]:
    """Najde hranice peaku (včetně podkmitu) pomocí derivace vyhlazeného signálu.

    Levá: zleva doprava, kde začíná významná kladná derivace (náběh).
    Pravá: zprava doleva, kde končí významná záporná derivace (konec podkmitu).
    """
    dy = np.gradient(y)
    dy_smooth = centered_moving_average(dy, max(len(y) // 30, 5))
    dy_max = np.max(np.abs(dy_smooth))
    left_threshold = 0.15 * dy_max
    right_threshold = 0.15 * dy_max

    # Levá: zleva doprava, kde derivace poprvé překročí +práh
    left = 0
    while left < peak_idx and dy_smooth[left] < left_threshold:
        left += 1

    # Pravá: zprava doleva, kde derivace poprvé klesne pod -práh
    right = len(y) - 1
    while right > peak_idx and dy_smooth[right] > -right_threshold:
        right -= 1
    # Pokračujeme doprava až kde záporná derivace skončí
    while right < len(y) - 1 and dy_smooth[right] < -right_threshold:
        right += 1

    return left, right


DELAY_OFFSET_US = 20.0  # offset od peaku pro výpočet delay hodnoty [µs]


def compute_waveform_measurements(wf: Waveform) -> tuple[float, float, float]:
    """Vrátí (amplituda_V, plocha_Vs, delay_value_V) pro daný waveform.

    delay_value_V je hodnota waveformu v čase (peak + DELAY_OFFSET_US µs).
    """
    smooth = centered_moving_average(wf.y_v, SMOOTHING_WINDOW)
    peak_idx = int(np.argmax(smooth))
    amplitude = float(smooth[peak_idx])
    baseline = float(np.mean(np.concatenate([wf.y_v[:20], wf.y_v[-20:]])))
    if amplitude > baseline:
        left, right = _find_peak_region(smooth, peak_idx, baseline)
        peak_baseline = np.linspace(smooth[left], smooth[right], right - left + 1)
        area = float(np.trapezoid(np.abs(smooth[left:right + 1] - peak_baseline), dx=wf.xinc))
    else:
        area = 0.0
    delay_samples = int(round(DELAY_OFFSET_US / (wf.xinc * 1e6)))
    delay_idx = min(peak_idx + delay_samples, len(wf.y_v) - 1)
    delay_value = float(smooth[delay_idx])
    return amplitude, area, delay_value


def compute_waveform_markers(wf: Waveform) -> tuple[float, float, float, float]:
    """Vrátí (peak_time_us, amplitude_V, delay_time_us, delay_value_V)."""
    smooth = centered_moving_average(wf.y_v, SMOOTHING_WINDOW)
    peak_idx = int(np.argmax(smooth))
    amplitude = float(smooth[peak_idx])
    peak_time = float(wf.x_us[peak_idx])
    delay_samples = int(round(DELAY_OFFSET_US / (wf.xinc * 1e6)))
    delay_idx = min(peak_idx + delay_samples, len(wf.y_v) - 1)
    delay_value = float(smooth[delay_idx])
    delay_time = float(wf.x_us[delay_idx])
    return peak_time, amplitude, delay_time, delay_value


def compute_peak_fill_data(wf: Waveform) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Vrátí (x_us_region, y_v_region, baseline_region) pro vyplnění plochy pod peakem.

    baseline_region je šikmá přímka mezi okraji peaku (na vyhlazeném signálu).
    Vrátí None pokud peak není nad baseline.
    """
    smooth = centered_moving_average(wf.y_v, SMOOTHING_WINDOW)
    peak_idx = int(np.argmax(smooth))
    baseline = float(np.mean(np.concatenate([wf.y_v[:20], wf.y_v[-20:]])))
    if float(smooth[peak_idx]) <= baseline:
        return None
    left, right = _find_peak_region(smooth, peak_idx, baseline)
    peak_baseline = np.linspace(smooth[left], smooth[right], right - left + 1)
    return wf.x_us[left:right + 1], smooth[left:right + 1], peak_baseline


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

    def exposure_seconds(self) -> float:
        """Vrátí celkovou expoziční dobu v sekundách z START_TIMESTAMP/END_TIMESTAMP."""
        attrs = self.handle.attrs
        start = attrs.get("START_TIMESTAMP", None)
        end = attrs.get("END_TIMESTAMP", None)
        if start is not None and end is not None:
            try:
                return float(end) - float(start)
            except (ValueError, TypeError):
                pass
        return 0.0

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

        waveform = Waveform(sample_id=sample_id, series_name=series_name, x_us=x_us, y_v=y_v, xinc=xinc)
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
            xinc=waveform.xinc,
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


class CombinedDataset:
    """Sloučení více H5WaveformDataset instancí do jednoho virtuálního datasetu."""

    def __init__(self, datasets: list[H5WaveformDataset]) -> None:
        if not datasets:
            raise ValueError("Alespoň jeden dataset je povinný.")
        self.datasets = datasets
        self.path = datasets[0].path
        self.handle = datasets[0].handle
        self.series_specs = datasets[0].series_specs
        self.series_by_name = datasets[0].series_by_name

        # Mapování globální sample_id -> (dataset_idx, lokální sample_id)
        self._id_map: dict[int, tuple[int, int]] = {}
        self.sample_ids: list[int] = []

        global_id = 0
        for ds_idx, ds in enumerate(datasets):
            for local_id in ds.sample_ids:
                self._id_map[global_id] = (ds_idx, local_id)
                self.sample_ids.append(global_id)
                global_id += 1

    def _scalar(self, key: str, default: float = 0.0) -> float:
        return self.datasets[0]._scalar(key, default)

    def _container_scalar(self, container, key: str, metadata_in_attrs: bool, default: float = 0.0) -> float:
        return self.datasets[0]._container_scalar(container, key, metadata_in_attrs, default)

    def waveform(self, sample_id: int, series_name: str) -> Waveform | None:
        mapping = self._id_map.get(sample_id)
        if mapping is None:
            return None
        ds_idx, local_id = mapping
        ds = self.datasets[ds_idx]
        if series_name not in ds.series_by_name:
            return None
        wf = ds.waveform(local_id, series_name)
        if wf is None:
            return None
        return Waveform(
            sample_id=sample_id,
            series_name=wf.series_name,
            x_us=wf.x_us,
            y_v=wf.y_v,
            xinc=wf.xinc,
        )

    def display_waveform(self, sample_id: int, series_name: str,
                         max_points: int = MAX_RENDER_POINTS) -> Waveform | None:
        mapping = self._id_map.get(sample_id)
        if mapping is None:
            return None
        ds_idx, local_id = mapping
        ds = self.datasets[ds_idx]
        if series_name not in ds.series_by_name:
            return None
        wf = ds.display_waveform(local_id, series_name, max_points)
        if wf is None:
            return None
        return Waveform(
            sample_id=sample_id,
            series_name=wf.series_name,
            x_us=wf.x_us,
            y_v=wf.y_v,
            xinc=wf.xinc,
        )

    def exposure_seconds(self) -> float:
        """Sčítá expoziční doby všech datasetů."""
        return sum(ds.exposure_seconds() for ds in self.datasets)

    def summary_text(self) -> str:
        lines = [f"Sloučeno {len(self.datasets)} souborů:"]
        for ds in self.datasets:
            lines.append(f"  {ds.path.name}: {len(ds.sample_ids)} vzorků")
        lines.append(f"Celkem vzorků: {len(self.sample_ids)}")
        lines.append(f"Série: {', '.join(s.title for s in self.series_specs)}")
        exposure = self.exposure_seconds()
        if exposure > 0:
            lines.append(f"Celková expozice: {exposure:.1f} s")
        return "\n".join(lines)

    def close(self) -> None:
        for ds in self.datasets:
            ds.close()


# ---------------------------------------------------------------------------
#  Matplotlib oscilloscope-style preview renderer
# ---------------------------------------------------------------------------

_MPL_BACKGROUND  = "#0a0a0a"
_MPL_PLOT_BG     = "#0c1117"
_MPL_GRID_COLOR  = "#1a3a2a"
_MPL_GRID_MINOR  = "#0f1f17"
_MPL_AXIS_COLOR  = "#3a7a5a"
_MPL_TEXT_COLOR  = "#b0d0b8"
_MPL_TITLE_COLOR = "#40e070"
_MPL_TRACE_COLORS = CURRENT_COLORS


def _h5_attr_str(attrs, key: str, default: str = "") -> str:
    """Read an HDF5 attribute as a Python str."""
    if key not in attrs:
        return default
    val = attrs[key]
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _apply_mpl_oscilloscope_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor":  _MPL_BACKGROUND,
        "axes.facecolor":    _MPL_PLOT_BG,
        "axes.edgecolor":    _MPL_AXIS_COLOR,
        "axes.labelcolor":   _MPL_TEXT_COLOR,
        "xtick.color":       _MPL_AXIS_COLOR,
        "ytick.color":       _MPL_AXIS_COLOR,
        "text.color":        _MPL_TEXT_COLOR,
        "grid.color":        _MPL_GRID_COLOR,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "grid.alpha":        0.7,
        "font.family":       "monospace",
        "font.size":         10,
    })


def _draw_trigger_lines(ax, dataset: H5WaveformDataset,
                        color_by_series: dict[str, str]) -> None:
    """Draw trigger level (horizontal) and trigger time (vertical, x=0) lines."""
    drawn: set[tuple] = set()
    trig_x_drawn = False

    for spec in dataset.series_specs:
        container = (dataset.handle if spec.container_path is None
                     else dataset.handle[spec.container_path])

        trig_level = None
        if spec.metadata_in_attrs:
            if "TRIG_LEVEL" in container.attrs:
                trig_level = float(np.asarray(container.attrs["TRIG_LEVEL"]).reshape(-1)[0])
        elif "TRIG_LEVEL" in container:
            trig_level = dataset._container_scalar(container, "TRIG_LEVEL", False)
        if trig_level is None:
            continue

        trig_channel: str | None = None
        if spec.metadata_in_attrs and "TRIG_CHANNEL" in container.attrs:
            raw = container.attrs["TRIG_CHANNEL"]
            trig_channel = raw.decode() if isinstance(raw, bytes) else str(raw)
        elif not spec.metadata_in_attrs and "TRIG_CHANNEL" in container:
            raw = np.asarray(container["TRIG_CHANNEL"]).flat[0]
            trig_channel = raw.decode() if isinstance(raw, bytes) else str(raw)

        key = (trig_channel, trig_level)
        if key in drawn:
            continue
        drawn.add(key)

        color = color_by_series.get(trig_channel, _MPL_AXIS_COLOR) if trig_channel else _MPL_AXIS_COLOR
        ch_label = f" ({trig_channel})" if trig_channel else ""
        ax.axhline(y=trig_level, color=color, linestyle="--", linewidth=0.7, alpha=0.6,
                   label=f"Trig{ch_label} {trig_level:.4g} V")
        if not trig_x_drawn:
            ax.axvline(x=0.0, color=color, linestyle="--", linewidth=0.7, alpha=0.6)
            trig_x_drawn = True


def _format_run_time(dataset: H5WaveformDataset) -> str:
    """Return a human-readable measurement time string for the figure footer."""
    if "START_TIME" in dataset.handle.attrs:
        start_raw = _h5_attr_str(dataset.handle.attrs, "START_TIME")
        end_raw = _h5_attr_str(dataset.handle.attrs, "END_TIME")
        try:
            start_dt = dt.datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return f"{start_raw} \u2192 {end_raw}".strip() if end_raw else start_raw
        if not end_raw:
            return start_str
        try:
            end_dt = dt.datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
            if start_dt.date() == end_dt.date():
                end_str = end_dt.strftime("%H:%M:%S")
            else:
                end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            dur = (end_dt - start_dt).total_seconds()
            return f"{start_str} \u2192 {end_str} ({dur:.1f}s)"
        except ValueError:
            return f"{start_str} \u2192 {end_raw}"
    for key in ("THETIME", "TIME"):
        if key in dataset.handle:
            ts = dataset._scalar(key, default=0.0)
            try:
                return dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
            except (OverflowError, OSError, ValueError):
                pass
    return ""


def _format_si(value: float, unit: str) -> str:
    """Format a value with SI prefix, e.g. 0.0256 V -> '25.6 mV'."""
    if value == 0:
        return f"0 {unit}"
    prefixes = [
        (1e-12, "p"), (1e-9, "n"), (1e-6, "\u00b5"), (1e-3, "m"),
        (1, ""), (1e3, "k"), (1e6, "M"),
    ]
    abs_val = abs(value)
    for scale, prefix in prefixes:
        if abs_val < scale * 1000:
            return f"{value / scale:.4g} {prefix}{unit}"
    return f"{value:.4g} {unit}"


def _add_oscilloscope_footer(fig, dataset: H5WaveformDataset, sample_id: int) -> None:
    """Add scope name, IP, time range, V/div, s/div below the plot."""
    left_parts: list[str] = []
    scope = _h5_attr_str(dataset.handle.attrs, "SCOPE_NAME")
    ip = _h5_attr_str(dataset.handle.attrs, "IP")
    if scope:
        left_parts.append(f"Scope: {scope}")
    if ip:
        left_parts.append(ip)
    left_parts.append(dataset.path.name)

    right_parts = [f"Sample {sample_id}/{len(dataset.sample_ids)}"]
    run_time = _format_run_time(dataset)
    if run_time:
        right_parts.append(run_time)

    meas_parts: list[str] = []
    for spec in dataset.series_specs:
        wf = dataset.waveform(sample_id, spec.name)
        if wf is None:
            continue
        amp, area, _delay = compute_waveform_measurements(wf)
        meas_parts.append(f"{spec.title}: A={_format_si(amp, 'V')} S={_format_si(area, 'V·s')}")

    fig.text(0.02, 0.01, "  |  ".join(left_parts), fontsize=8,
             color=_MPL_TEXT_COLOR, family="monospace", va="bottom")
    if meas_parts:
        fig.text(0.5, 0.01, "  |  ".join(meas_parts), fontsize=8,
                 color=_MPL_TEXT_COLOR, family="monospace", va="bottom", ha="center")
    fig.text(0.98, 0.01, "  |  ".join(right_parts), fontsize=8,
             color=_MPL_TEXT_COLOR, family="monospace", va="bottom", ha="right")


def _create_waveform_figure(
    dataset: H5WaveformDataset,
    sample_id: int,
    figsize: tuple[float, float] = (14, 9),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    locked_samples: dict[int, str] | None = None,
):
    """Create a matplotlib figure with waveform plot in oscilloscope style.

    Parameters
    ----------
    locked_samples : dict[int, str] | None
        Mapping ``{sample_id: series_name_for_color}`` of locked samples to
        overlay as dashed traces.  Pass ``None`` or empty dict to skip.

    Returns (fig, has_data).  Caller is responsible for saving / showing /
    closing the figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    _apply_mpl_oscilloscope_style()

    fig, ax = plt.subplots(figsize=figsize)

    # -- plot waveforms --------------------------------------------------------
    color_by_series: dict[str, str] = {}
    has_data = False
    for idx, spec in enumerate(dataset.series_specs):
        wf = dataset.waveform(sample_id, spec.name)
        if wf is None:
            continue
        has_data = True
        color = _MPL_TRACE_COLORS[idx % len(_MPL_TRACE_COLORS)]
        color_by_series[spec.name] = color
        ax.plot(wf.x_us, wf.y_v, color=color, linewidth=0.8, alpha=0.92,
                label=spec.title)

    # -- peak area fill --------------------------------------------------------
    for idx, spec in enumerate(dataset.series_specs):
        wf = dataset.waveform(sample_id, spec.name)
        if wf is None:
            continue
        fill_data = compute_peak_fill_data(wf)
        if fill_data is None:
            continue
        x_fill, y_fill, bl = fill_data
        color = _MPL_TRACE_COLORS[idx % len(_MPL_TRACE_COLORS)]
        ax.fill_between(x_fill, bl, y_fill, where=(y_fill > bl),
                        alpha=0.25, color=color)
        ax.plot(x_fill, bl, color=color, linewidth=0.6, linestyle='--', alpha=0.5)

    # -- locked samples (dashed overlay) ---------------------------------------
    if locked_samples:
        for locked_id in locked_samples:
            for idx, spec in enumerate(dataset.series_specs):
                wf = dataset.waveform(locked_id, spec.name)
                if wf is None:
                    continue
                color = color_by_series.get(spec.name,
                            _MPL_TRACE_COLORS[idx % len(_MPL_TRACE_COLORS)])
                ax.plot(wf.x_us, wf.y_v, color=color, linewidth=0.6, alpha=0.5,
                        linestyle="--", label=f"{spec.title} #{locked_id}")

    if not has_data:
        return fig, False

    # -- trigger lines ---------------------------------------------------------
    _draw_trigger_lines(ax, dataset, color_by_series)

    # -- axes, grid, labels ----------------------------------------------------
    ax.set_xlabel("Time [\u00b5s]", fontsize=11)
    ax.set_ylabel("Amplitude [V]", fontsize=11)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, which="major", color=_MPL_GRID_COLOR, linewidth=0.5)
    ax.grid(True, which="minor", color=_MPL_GRID_MINOR, linewidth=0.3, alpha=0.5)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", length=4)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    # -- title -----------------------------------------------------------------
    meas = _h5_attr_str(dataset.handle.attrs, "MEASUREMENT_NAME")
    title = f"Waveform #{sample_id}"
    if meas:
        title = f"{meas}  \u2014  {title}"
    ax.set_title(title, fontsize=13, color=_MPL_TITLE_COLOR, fontweight="bold", pad=12)

    # -- legend ----------------------------------------------------------------
    ax.legend(loc="upper right", fontsize=9, framealpha=0.5,
              edgecolor=_MPL_AXIS_COLOR, facecolor=_MPL_PLOT_BG,
              labelcolor=_MPL_TEXT_COLOR)

    # -- footer metadata -------------------------------------------------------
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _add_oscilloscope_footer(fig, dataset, sample_id)

    return fig, True


def render_waveform_preview(
    dataset: H5WaveformDataset,
    sample_id: int,
    output_path: str | Path | None = None,
    dpi: int = 200,
    figsize: tuple[float, float] = (14, 9),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generate a PNG waveform preview in oscilloscope style."""
    import matplotlib.pyplot as plt

    fig, has_data = _create_waveform_figure(dataset, sample_id, figsize=figsize,
                                            xlim=xlim, ylim=ylim)
    if not has_data:
        plt.close(fig)
        print(f"Sample {sample_id}: no data to plot.")
        return

    if output_path is None:
        output_path = dataset.path.with_name(f"{dataset.path.stem}_{sample_id}.png")
    output_path = Path(output_path)

    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def render_all_previews(
    path: str | Path,
    sample_ids: list[int] | None = None,
    output_dir: str | Path | None = None,
    dpi: int = 200,
) -> None:
    """Vygeneruje PNG náhledy pro vybrané (nebo všechny) vzorky v H5 souboru.

    Výstupní složka se ve výchozím stavu pojmenuje stejně jako zdrojový H5
    soubor (bez přípony).  Všechny obrázky sdílejí stejné měřítko os.
    """
    dataset = H5WaveformDataset(path)
    try:
        ids = sample_ids if sample_ids else dataset.sample_ids

        # -- výstupní adresář pojmenovaný podle H5 souboru ---------------------
        if output_dir is not None:
            out = Path(output_dir)
        else:
            out = dataset.path.parent / dataset.path.stem
        out.mkdir(parents=True, exist_ok=True)

        # -- 1. průchod: zjistit globální rozsahy os ---------------------------
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        valid_ids: list[int] = []

        for sid in ids:
            if sid not in dataset.sample_ids:
                print(f"Vzorek {sid} nenalezen, přeskakuji.")
                continue
            for spec in dataset.series_specs:
                wf = dataset.waveform(sid, spec.name)
                if wf is None:
                    continue
                x_min = min(x_min, float(wf.x_us[0]))
                x_max = max(x_max, float(wf.x_us[-1]))
                y_min = min(y_min, float(np.min(wf.y_v)))
                y_max = max(y_max, float(np.max(wf.y_v)))
                if sid not in valid_ids:
                    valid_ids.append(sid)

        if not valid_ids:
            print("Žádná platná data k vykreslení.")
            return

        # malý padding kolem dat
        y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
        x_pad = (x_max - x_min) * 0.01 if x_max > x_min else 0.1
        xlim = (x_min - x_pad, x_max + x_pad)
        ylim = (y_min - y_pad, y_max + y_pad)

        print(f"Generuji {len(valid_ids)} náhledů do {out}/  (xlim={xlim}, ylim={ylim})")

        # -- 2. průchod: vykreslení se sjednoceným měřítkem --------------------
        from tqdm import tqdm
        for sid in tqdm(valid_ids, desc="Rendering", unit="img"):
            out_file = out / f"{dataset.path.stem}_{sid}.png"
            render_waveform_preview(dataset, sid, output_path=out_file, dpi=dpi,
                                    xlim=xlim, ylim=ylim)
    finally:
        dataset.close()


# ---------------------------------------------------------------------------
#  PyQtGraph interactive viewer (vyžaduje PyQt / pyqtgraph)
# ---------------------------------------------------------------------------

class Histogram2DWidget(QtWidgets.QWidget):
    """Widget pro 2D histogram amplituda vs plocha s ROI výběrem a filtrem peaků."""

    def __init__(self, dataset: H5WaveformDataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        self._sample_callback = None
        self._sample_preview_callback = None
        # measurements[series_name][sample_id] = (amplitude, area, delay_value)
        self.measurements: dict[str, dict[int, tuple[float, float, float]]] = {}
        self.plot_mode: str = "amp_area"  # "amp_area" nebo "amp_delay"
        self.computed = False
        self.scatter_items: dict[str, pg.ScatterPlotItem] = {}
        self.scatter_sids: dict[str, list[int]] = {}
        self.scatter_colors: dict[str, str] = {}
        self.heatmap_item: pg.ImageItem | None = None
        self.roi: pg.PolyLineROI | None = None
        self.filtered_ids: list[int] = []
        self.filtered_position: int = -1
        self.crosshair_lines: list[pg.InfiniteLine] = []
        self.crosshair_markers: dict[str, pg.ScatterPlotItem] = {}
        self._build_ui()

    def set_sample_callback(self, callback) -> None:
        self._sample_callback = callback

    def set_sample_preview_callback(self, callback) -> None:
        """Callback pro změnu vzorku v grafu bez přepnutí tabu."""
        self._sample_preview_callback = callback

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # -- Ovládací prvky --
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)

        self.compute_button = QtWidgets.QPushButton("Vypočítat měření")
        self.compute_button.clicked.connect(self._compute_all_measurements)
        controls.addWidget(self.compute_button)

        controls.addWidget(QtWidgets.QLabel("Filtr peaků:"))
        self.peak_filter = QtWidgets.QComboBox()
        self.peak_filter.currentIndexChanged.connect(self._apply_filters)
        controls.addWidget(self.peak_filter)

        controls.addWidget(QtWidgets.QLabel("Práh [V]:"))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 100.0)
        self.threshold_spin.setValue(0.01)
        self.threshold_spin.setSingleStep(0.001)
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.valueChanged.connect(self._apply_filters)
        controls.addWidget(self.threshold_spin)

        self.roi_check = QtWidgets.QCheckBox("ROI výběr")
        self.roi_check.toggled.connect(self._toggle_roi)
        controls.addWidget(self.roi_check)

        self.roi_reset_button = QtWidgets.QPushButton("Reset ROI")
        self.roi_reset_button.setEnabled(False)
        self.roi_reset_button.clicked.connect(self._reset_roi)
        controls.addWidget(self.roi_reset_button)

        controls.addWidget(QtWidgets.QLabel("ROI kanál:"))
        self.roi_channel_combo = QtWidgets.QComboBox()
        self.roi_channel_combo.currentIndexChanged.connect(self._apply_filters)
        controls.addWidget(self.roi_channel_combo)

        controls.addWidget(QtWidgets.QLabel("Osa X:"))
        self.plot_mode_combo = QtWidgets.QComboBox()
        self.plot_mode_combo.addItem("Amplituda vs Plocha", "amp_area")
        self.plot_mode_combo.addItem(f"Amplituda vs Delay (+{int(DELAY_OFFSET_US)} µs)", "amp_delay")
        self.plot_mode_combo.currentIndexChanged.connect(self._on_plot_mode_changed)
        controls.addWidget(self.plot_mode_combo)

        controls.addWidget(QtWidgets.QLabel("Zobrazit:"))
        self.visible_channel_combo = QtWidgets.QComboBox()
        self.visible_channel_combo.currentIndexChanged.connect(self._on_visible_channel_changed)
        controls.addWidget(self.visible_channel_combo)

        controls.addStretch()
        layout.addLayout(controls)

        # -- Splitter: graf + seznam --
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.scatter_plot = pg.PlotWidget()
        self.scatter_plot.setBackground("#101418")
        self.scatter_plot.showGrid(x=True, y=True, alpha=0.25)
        self.scatter_plot.setLabel("left", "Amplituda [V]")
        self.scatter_plot.setLabel("bottom", "Plocha [V·s]")
        self.scatter_plot.setTitle("2D Histogram: Amplituda vs Plocha")
        splitter.addWidget(self.scatter_plot)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(4)

        self.count_label = QtWidgets.QLabel("Vybrané vzorky: 0 / 0")
        side_layout.addWidget(self.count_label)

        nav = QtWidgets.QHBoxLayout()
        self.prev_button = QtWidgets.QPushButton("◀ Předchozí")
        self.prev_button.clicked.connect(self._prev_filtered)
        nav.addWidget(self.prev_button)
        self.next_button = QtWidgets.QPushButton("Další ▶")
        self.next_button.clicked.connect(self._next_filtered)
        nav.addWidget(self.next_button)
        side_layout.addLayout(nav)

        self.sample_list = QtWidgets.QListWidget()
        self.sample_list.currentItemChanged.connect(self._on_sample_selected)
        self.sample_list.itemDoubleClicked.connect(self._on_sample_double_clicked)
        side_layout.addWidget(self.sample_list, stretch=1)

        splitter.addWidget(side)
        splitter.setSizes([900, 300])
        layout.addWidget(splitter, stretch=1)

        self._populate_peak_filter()
        self._populate_roi_channel_combo()
        self._populate_visible_channel_combo()

    def _populate_roi_channel_combo(self) -> None:
        self.roi_channel_combo.blockSignals(True)
        self.roi_channel_combo.clear()
        specs = self.dataset.series_specs
        self.roi_channel_combo.addItem("Oba kanály", "any")
        for i, spec in enumerate(specs):
            self.roi_channel_combo.addItem(spec.title, f"ch_{i}")
        self.roi_channel_combo.blockSignals(False)

    def _populate_visible_channel_combo(self) -> None:
        self.visible_channel_combo.blockSignals(True)
        self.visible_channel_combo.clear()
        self.visible_channel_combo.addItem("Všechny kanály", None)
        for spec in self.dataset.series_specs:
            self.visible_channel_combo.addItem(f"Pouze {spec.title}", spec.name)
        self.visible_channel_combo.blockSignals(False)

    def _visible_series_specs(self) -> list:
        """Vrátit série viditelné podle výběru kanálů."""
        selected = self.visible_channel_combo.currentData()
        if selected is None:
            return self.dataset.series_specs
        return [s for s in self.dataset.series_specs if s.name == selected]

    def _on_visible_channel_changed(self) -> None:
        if self.computed:
            self._refresh_scatter()
            self._apply_filters()

    def _populate_peak_filter(self) -> None:
        self.peak_filter.blockSignals(True)
        self.peak_filter.clear()
        self.peak_filter.addItem("Všechny", "all")
        specs = self.dataset.series_specs
        if len(specs) >= 2:
            self.peak_filter.addItem(f"Peak jen v {specs[0].title}", "only_0")
            self.peak_filter.addItem(f"Peak jen v {specs[1].title}", "only_1")
            self.peak_filter.addItem(f"Peak v obou ({specs[0].title} + {specs[1].title})", "both")
        elif len(specs) == 1:
            self.peak_filter.addItem(f"S peakem v {specs[0].title}", "has_0")
        self.peak_filter.blockSignals(False)

    def _compute_all_measurements(self) -> None:
        """Vypočítat amplitudu a plochu pro všechny vzorky ve všech sériích."""
        self.measurements.clear()
        total = len(self.dataset.sample_ids) * len(self.dataset.series_specs)

        progress = QtWidgets.QProgressDialog("Počítám měření...", "Zrušit", 0, total, self)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(500)

        count = 0
        for spec in self.dataset.series_specs:
            series_meas: dict[int, tuple[float, float, float]] = {}
            for sid in self.dataset.sample_ids:
                if progress.wasCanceled():
                    return
                wf = self.dataset.waveform(sid, spec.name)
                if wf is not None:
                    amp, area, delay_value = compute_waveform_measurements(wf)
                    series_meas[sid] = (amp, area, delay_value)
                count += 1
                progress.setValue(count)
            self.measurements[spec.name] = series_meas

        progress.close()
        self.computed = True
        self.compute_button.setEnabled(False)
        self.compute_button.setText("Vypočítáno ✓")
        self._refresh_scatter()
        self._apply_filters()

    def _on_plot_mode_changed(self) -> None:
        self.plot_mode = self.plot_mode_combo.currentData()
        if self.computed:
            self._refresh_scatter()
            self._apply_filters()

    def _x_value(self, meas_tuple: tuple[float, float, float]) -> float:
        """Vrátí X souřadnici pro aktuální plot_mode."""
        amp, area, delay_value = meas_tuple
        return area if self.plot_mode == "amp_area" else delay_value

    def _refresh_scatter(self) -> None:
        """Překreslit scatter plot a heatmapu."""
        self.scatter_plot.clear()
        self.scatter_items.clear()
        self.scatter_sids.clear()
        self.scatter_colors.clear()
        self.heatmap_item = None

        use_delay = self.plot_mode == "amp_delay"
        if use_delay:
            self.scatter_plot.setLabel("bottom", f"Delay value +{int(DELAY_OFFSET_US)} µs [V]")
            self.scatter_plot.setTitle(f"2D Histogram: Amplituda vs Delay (+{int(DELAY_OFFSET_US)} µs)")
        else:
            self.scatter_plot.setLabel("bottom", "Plocha [V·s]")
            self.scatter_plot.setTitle("2D Histogram: Amplituda vs Plocha")

        all_xs: list[float] = []
        all_amps: list[float] = []

        visible = self._visible_series_specs()

        for idx, spec in enumerate(self.dataset.series_specs):
            if spec not in visible:
                continue
            meas = self.measurements.get(spec.name, {})
            if not meas:
                continue
            sids = sorted(meas.keys())
            xs = [self._x_value(meas[s]) for s in sids]
            amps = [meas[s][0] for s in sids]

            all_xs.extend(xs)
            all_amps.extend(amps)

            color = CURRENT_COLORS[idx % len(CURRENT_COLORS)]
            self.scatter_sids[spec.name] = sids
            self.scatter_colors[spec.name] = color
            scatter = pg.ScatterPlotItem(
                x=np.array(xs),
                y=np.array(amps),
                pen=pg.mkPen(color, width=0.5),
                brush=pg.mkBrush(color + "80"),
                size=5,
                name=spec.title,
            )
            scatter.sigClicked.connect(self._on_scatter_clicked)
            self.scatter_plot.addItem(scatter)
            self.scatter_items[spec.name] = scatter

        # 2D histogram heatmap na pozadí
        if len(all_xs) > 10:
            areas_arr = np.array(all_xs)
            amps_arr = np.array(all_amps)
            bins = min(80, max(15, int(np.sqrt(len(all_xs)))))
            hist, xedges, yedges = np.histogram2d(areas_arr, amps_arr, bins=bins)

            img = pg.ImageItem()
            positions = np.array([0.0, 0.01, 0.1, 0.4, 0.7, 1.0])
            colors_rgba = np.array([
                [0, 0, 0, 0],
                [10, 20, 60, 80],
                [30, 80, 150, 120],
                [60, 180, 100, 150],
                [200, 220, 50, 180],
                [255, 60, 30, 220],
            ], dtype=np.ubyte)
            cmap = pg.ColorMap(positions, colors_rgba)
            lut = cmap.getLookupTable(nPts=256)

            img.setImage(hist)
            img.setLookupTable(lut)
            img.setRect(QtCore.QRectF(
                xedges[0], yedges[0],
                xedges[-1] - xedges[0], yedges[-1] - yedges[0],
            ))
            img.setZValue(-10)
            img.setOpacity(0.8)
            self.scatter_plot.addItem(img)
            self.heatmap_item = img

        self.scatter_plot.addLegend(offset=(10, 10))
        self.scatter_plot.autoRange()
        self.scatter_plot.setYRange(0, 0.35)

    def _make_roi_points(self) -> list[list[float]]:
        """Vrátit 4 rohové body obdélníku uprostřed aktuálního pohledu."""
        vr = self.scatter_plot.viewRange()
        x_center = (vr[0][0] + vr[0][1]) / 2
        y_center = (vr[1][0] + vr[1][1]) / 2
        hw = (vr[0][1] - vr[0][0]) * 0.15
        hh = (vr[1][1] - vr[1][0]) * 0.15
        return [
            [x_center - hw, y_center - hh],
            [x_center + hw, y_center - hh],
            [x_center + hw, y_center + hh],
            [x_center - hw, y_center + hh],
        ]

    def _toggle_roi(self, enabled: bool) -> None:
        if enabled:
            if self.roi is None:
                points = self._make_roi_points()
                self.roi = pg.PolyLineROI(
                    points,
                    closed=True,
                    pen=pg.mkPen("r", width=2),
                )
                self.roi.sigRegionChanged.connect(self._apply_filters)
            self.scatter_plot.addItem(self.roi)
            self.roi_reset_button.setEnabled(True)
        else:
            if self.roi is not None:
                self.scatter_plot.removeItem(self.roi)
            self.roi_reset_button.setEnabled(False)
        self._apply_filters()

    def _reset_roi(self) -> None:
        """Smazat aktuální ROI a vytvořit nové výchozí."""
        if self.roi is not None:
            self.scatter_plot.removeItem(self.roi)
            self.roi = None
        if self.roi_check.isChecked():
            points = self._make_roi_points()
            self.roi = pg.PolyLineROI(
                points,
                closed=True,
                pen=pg.mkPen("r", width=2),
            )
            self.roi.sigRegionChanged.connect(self._apply_filters)
            self.scatter_plot.addItem(self.roi)
        self._apply_filters()

    def _get_roi_polygon(self) -> np.ndarray | None:
        """Vrátit pole bodů [[x,y], ...] polygonu ROI v datových souřadnicích."""
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

    def _apply_filters(self) -> None:
        """Aplikovat filtr peaků a ROI výběr."""
        if not self.computed:
            return

        threshold = self.threshold_spin.value()
        filter_mode = self.peak_filter.currentData()
        specs = self.dataset.series_specs

        # -- Filtr podle přítomnosti peaku v kanálech --
        candidate_ids = set(self.dataset.sample_ids)

        if filter_mode and filter_mode != "all" and len(specs) >= 2:
            filtered: set[int] = set()
            for sid in self.dataset.sample_ids:
                has_peak: list[bool] = []
                for spec in specs:
                    meas = self.measurements.get(spec.name, {})
                    if sid in meas:
                        amp, _area, _delay = meas[sid]
                        has_peak.append(amp >= threshold)
                    else:
                        has_peak.append(False)

                if filter_mode == "only_0" and len(has_peak) >= 1 and has_peak[0] and not any(has_peak[1:]):
                    filtered.add(sid)
                elif filter_mode == "only_1" and len(has_peak) >= 2 and has_peak[1] and not has_peak[0]:
                    filtered.add(sid)
                elif filter_mode == "both" and all(has_peak):
                    filtered.add(sid)
            candidate_ids = filtered
        elif filter_mode == "has_0" and len(specs) >= 1:
            filtered = set()
            spec = specs[0]
            meas = self.measurements.get(spec.name, {})
            for sid in self.dataset.sample_ids:
                if sid in meas:
                    amp, _area, _delay = meas[sid]
                    if amp >= threshold:
                        filtered.add(sid)
            candidate_ids = filtered

        # -- Filtr podle ROI (polygon point-in-polygon test) --
        if self.roi_check.isChecked() and self.roi is not None:
            polygon = self._get_roi_polygon()
            if polygon is not None and len(polygon) >= 3:
                roi_channel = self.roi_channel_combo.currentData()
                if roi_channel and roi_channel.startswith("ch_"):
                    ch_idx = int(roi_channel[3:])
                    roi_specs = [specs[ch_idx]] if ch_idx < len(specs) else []
                else:
                    roi_specs = list(specs)
                roi_ids: set[int] = set()
                for spec in roi_specs:
                    meas = self.measurements.get(spec.name, {})
                    for sid in candidate_ids:
                        if sid in meas:
                            amp = meas[sid][0]
                            x_val = self._x_value(meas[sid])
                            if self._point_in_polygon(x_val, amp, polygon):
                                roi_ids.add(sid)
                candidate_ids = roi_ids

        self.filtered_ids = sorted(candidate_ids)
        self.filtered_position = 0 if self.filtered_ids else -1
        self._refresh_sample_list()
        self._update_scatter_visibility()

    def _refresh_sample_list(self) -> None:
        self.sample_list.clear()
        for sid in self.filtered_ids:
            parts: list[str] = []
            for spec in self.dataset.series_specs:
                meas = self.measurements.get(spec.name, {})
                if sid in meas:
                    amp, area, delay_value = meas[sid]
                    parts.append(
                        f"{spec.title}: A={_format_si(amp, 'V')} "
                        f"S={_format_si(area, 'V·s')} "
                        f"D={_format_si(delay_value, 'V')}"
                    )
            text = f"#{sid}  " + "  |  ".join(parts)
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, sid)
            self.sample_list.addItem(item)

        total = len(self.dataset.sample_ids)
        selected = len(self.filtered_ids)
        exposure = self.dataset.exposure_seconds()
        if exposure > 0:
            self.count_label.setText(
                f"Vybrané: {selected} / {total}  |  "
                f"Expozice: {exposure:.1f} s  |  "
                f"Celkem: {total / exposure:.2f} CPS  |  "
                f"ROI: {selected / exposure:.2f} CPS"
            )
        else:
            self.count_label.setText(f"Vybrané vzorky: {selected} / {total}")

    def _update_scatter_visibility(self) -> None:
        """Ztlumit body, které nejsou ve výběru."""
        if not self.computed:
            return
        filtered_set = set(self.filtered_ids)
        for spec_name, scatter in self.scatter_items.items():
            sids = self.scatter_sids.get(spec_name, [])
            color = self.scatter_colors.get(spec_name, "#ffffff")
            brushes = []
            pens = []
            for sid in sids:
                if sid in filtered_set:
                    brushes.append(pg.mkBrush(color + "80"))
                    pens.append(pg.mkPen(color, width=0.5))
                else:
                    brushes.append(pg.mkBrush(color + "18"))
                    pens.append(pg.mkPen(color + "25", width=0.3))
            scatter.setBrush(brushes)
            scatter.setPen(pens)

    def _on_scatter_clicked(self, scatter_item, points, ev) -> None:
        """Kliknutí na bod ve scatter plotu – zobrazit křížový kurzor (bez přepnutí tabu)."""
        if len(points) == 0:
            return
        point = points[0]
        clicked_series = None
        for spec_name, item in self.scatter_items.items():
            if item is scatter_item:
                clicked_series = spec_name
                break
        if clicked_series is None:
            return
        sids = self.scatter_sids.get(clicked_series, [])
        idx = point.index()
        if idx < 0 or idx >= len(sids):
            return
        sid = sids[idx]
        self._show_crosshair(sid)
        self._highlight_sample_in_list(sid)
        if self._sample_preview_callback:
            self._sample_preview_callback(sid)

    def _highlight_sample_in_list(self, sample_id: int) -> None:
        """Zvýraznit vzorek v seznamu bez vyvolání signálu."""
        if sample_id in self.filtered_ids:
            row = self.filtered_ids.index(sample_id)
            self.sample_list.blockSignals(True)
            self.sample_list.setCurrentRow(row)
            self.sample_list.blockSignals(False)
            self.filtered_position = row

    def _show_crosshair(self, sample_id: int) -> None:
        """Zobrazit křížový kurzor na pozici daného vzorku – jeden pár čar na kanál."""
        # Odstranit staré
        for line in self.crosshair_lines:
            self.scatter_plot.removeItem(line)
        self.crosshair_lines.clear()
        for marker in self.crosshair_markers.values():
            self.scatter_plot.removeItem(marker)
        self.crosshair_markers.clear()

        # Najít souřadnice pro všechny kanály
        positions: list[tuple[float, float, str]] = []  # (x_val, amp, series_name)
        for spec in self.dataset.series_specs:
            meas = self.measurements.get(spec.name, {})
            if sample_id in meas:
                amp = meas[sample_id][0]
                x_val = self._x_value(meas[sample_id])
                positions.append((x_val, amp, spec.name))

        if not positions:
            return

        # Křížový kurzor pro každý kanál zvlášť
        for area, amp, series_name in positions:
            color = self.scatter_colors.get(series_name, "#ffffff")
            pen = pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DashLine)
            vline = pg.InfiniteLine(pos=area, angle=90, pen=pen)
            hline = pg.InfiniteLine(pos=amp, angle=0, pen=pen)
            self.scatter_plot.addItem(vline)
            self.scatter_plot.addItem(hline)
            self.crosshair_lines.extend([vline, hline])

            marker = pg.ScatterPlotItem(
                x=[area], y=[amp],
                pen=pg.mkPen("#ffffff", width=2),
                brush=pg.mkBrush(color),
                size=14,
                symbol="crosshair",
            )
            marker.setZValue(100)
            self.scatter_plot.addItem(marker)
            self.crosshair_markers[series_name] = marker

    def _on_sample_selected(self, current: QtWidgets.QListWidgetItem, _previous=None) -> None:
        """Jedno kliknutí v seznamu – zobrazit crosshair a změnit graf."""
        if current is None:
            return
        sid = int(current.data(QtCore.Qt.ItemDataRole.UserRole))
        self._show_crosshair(sid)
        if self._sample_preview_callback:
            self._sample_preview_callback(sid)

    def _on_sample_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """Dvojklik v seznamu – přepnout na oscilogram."""
        sid = int(item.data(QtCore.Qt.ItemDataRole.UserRole))
        if self._sample_callback:
            self._sample_callback(sid)

    def _prev_filtered(self) -> None:
        if not self.filtered_ids:
            return
        self.filtered_position = max(0, self.filtered_position - 1)
        self.sample_list.setCurrentRow(self.filtered_position)
        sid = self.filtered_ids[self.filtered_position]
        self._show_crosshair(sid)
        if self._sample_callback:
            self._sample_callback(sid)

    def _next_filtered(self) -> None:
        if not self.filtered_ids:
            return
        self.filtered_position = min(len(self.filtered_ids) - 1, self.filtered_position + 1)
        self.sample_list.setCurrentRow(self.filtered_position)
        sid = self.filtered_ids[self.filtered_position]
        self._show_crosshair(sid)
        if self._sample_callback:
            self._sample_callback(sid)


class ChannelHeatmapWidget(QtWidgets.QWidget):
    """Persistence heatmapa pro jednotlivé kanály (2D histogram: čas vs amplituda)."""

    def __init__(self, dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        self.heatmap_plots: list[pg.PlotWidget] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # -- Ovládací prvky --
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)

        self.compute_button = QtWidgets.QPushButton("Vypočítat heatmapu")
        self.compute_button.clicked.connect(self._compute_heatmaps)
        controls.addWidget(self.compute_button)

        controls.addWidget(QtWidgets.QLabel("Biny X:"))
        self.xbins_spin = QtWidgets.QSpinBox()
        self.xbins_spin.setRange(50, 2000)
        self.xbins_spin.setValue(500)
        self.xbins_spin.setSingleStep(50)
        controls.addWidget(self.xbins_spin)

        controls.addWidget(QtWidgets.QLabel("Biny Y:"))
        self.ybins_spin = QtWidgets.QSpinBox()
        self.ybins_spin.setRange(50, 1000)
        self.ybins_spin.setValue(200)
        self.ybins_spin.setSingleStep(50)
        controls.addWidget(self.ybins_spin)

        self.log_check = QtWidgets.QCheckBox("Log škála")
        self.log_check.setChecked(True)
        controls.addWidget(self.log_check)

        controls.addStretch()
        layout.addLayout(controls)

        # -- Oblast pro heatmapy (jedna na kanál) --
        self.plot_container = QtWidgets.QWidget()
        self.plot_vlayout = QtWidgets.QVBoxLayout(self.plot_container)
        self.plot_vlayout.setContentsMargins(0, 0, 0, 0)
        self.plot_vlayout.setSpacing(4)
        layout.addWidget(self.plot_container, stretch=1)

    def _compute_heatmaps(self) -> None:
        """Vypočítat persistence heatmapu pro každý kanál."""
        while self.plot_vlayout.count():
            item = self.plot_vlayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.heatmap_plots.clear()

        x_bins = self.xbins_spin.value()
        y_bins = self.ybins_spin.value()
        use_log = self.log_check.isChecked()

        total = len(self.dataset.sample_ids) * len(self.dataset.series_specs)
        progress = QtWidgets.QProgressDialog("Počítám heatmapy...", "Zrušit", 0, total, self)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(500)

        count = 0
        for idx, spec in enumerate(self.dataset.series_specs):
            all_x: list[np.ndarray] = []
            all_y: list[np.ndarray] = []

            for sid in self.dataset.sample_ids:
                if progress.wasCanceled():
                    progress.close()
                    return
                wf = self.dataset.waveform(sid, spec.name)
                if wf is not None:
                    all_x.append(wf.x_us)
                    all_y.append(wf.y_v)
                count += 1
                progress.setValue(count)

            if not all_x:
                continue

            x_cat = np.concatenate(all_x)
            y_cat = np.concatenate(all_y)
            hist, xedges, yedges = np.histogram2d(x_cat, y_cat, bins=[x_bins, y_bins])

            display_data = np.log1p(hist) if use_log else hist.astype(np.float64)

            plot = pg.PlotWidget()
            plot.setBackground("#101418")
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("left", "Amplituda [V]")
            plot.setLabel("bottom", "Čas [µs]")
            color = CURRENT_COLORS[idx % len(CURRENT_COLORS)]
            plot.setTitle(
                f"Heatmapa: {spec.title} ({len(all_x)} waveformů)",
                color=color,
            )

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

            img.setImage(display_data)
            img.setLookupTable(lut)
            img.setRect(QtCore.QRectF(
                xedges[0], yedges[0],
                xedges[-1] - xedges[0], yedges[-1] - yedges[0],
            ))
            plot.addItem(img)
            plot.autoRange()

            self.heatmap_plots.append(plot)
            self.plot_vlayout.addWidget(plot, stretch=1)

        progress.close()
        self.compute_button.setText("Přepočítat heatmapu")


class WaveformTab(QtWidgets.QWidget):
    """Widget pro jeden otevřený H5 soubor (jedna karta)."""

    def __init__(self, dataset: H5WaveformDataset, initial_sample: int | None = None) -> None:
        super().__init__()
        self.dataset = dataset
        self.current_position = 0
        self.locked_samples: OrderedDict[int, QtGui.QColor] = OrderedDict()
        self.main_plot: pg.PlotWidget | None = None
        self.empty_plot_label: QtWidgets.QLabel | None = None
        self.pending_autorange = True
        self.high_res = False
        self.detached_window: _DetachedPlotWindow | None = None

        self._build_ui()
        self._init_dataset(initial_sample)

    def _build_ui(self) -> None:
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(8)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setSpacing(8)
        root_layout.addLayout(controls_layout)

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

        self.highres_button = QtWidgets.QPushButton("High-res")
        self.highres_button.setCheckable(True)
        self.highres_button.setToolTip("Přepnout mezi decimovanými a plnými daty")
        self.highres_button.clicked.connect(self._toggle_high_res)
        controls_layout.addWidget(self.highres_button)

        self.mpl_button = QtWidgets.QPushButton("Matplotlib")
        self.mpl_button.setToolTip("Otevřít aktuální vzorek v matplotlib okně")
        self.mpl_button.clicked.connect(self._open_matplotlib)
        controls_layout.addWidget(self.mpl_button)

        self.detach_button = QtWidgets.QPushButton("Detach")
        self.detach_button.setToolTip("Otevřít oscilogram v samostatném okně")
        self.detach_button.clicked.connect(self._detach_plot)
        controls_layout.addWidget(self.detach_button)

        controls_layout.addWidget(QtWidgets.QLabel("Kanály:"))
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        controls_layout.addWidget(self.channel_combo)

        self.path_label = QtWidgets.QLabel()
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        root_layout.addWidget(self.path_label)

        self.measurements_label = QtWidgets.QLabel()
        self.measurements_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.measurements_label.setStyleSheet("font-family: monospace; font-size: 11px; color: #f0d060;")
        root_layout.addWidget(self.measurements_label)

        # -- Podzáložky (Oscilogramy + 2D Histogram) --
        self.sub_tabs = QtWidgets.QTabWidget()
        root_layout.addWidget(self.sub_tabs, stretch=1)

        # -- Oscilogramy --
        oscilo_widget = QtWidgets.QWidget()
        oscilo_layout = QtWidgets.QVBoxLayout(oscilo_widget)
        oscilo_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        oscilo_layout.addWidget(splitter)

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

        self.sub_tabs.addTab(oscilo_widget, "Oscilogramy")

        # -- 2D Histogram --
        self.histogram_widget = Histogram2DWidget(self.dataset)
        self.histogram_widget.set_sample_callback(self._on_histogram_sample_requested)
        self.histogram_widget.set_sample_preview_callback(self._on_histogram_sample_preview)
        self.sub_tabs.addTab(self.histogram_widget, "2D Histogram")

        # -- Heatmapa kanálů --
        self.channel_heatmap_widget = ChannelHeatmapWidget(self.dataset)
        self.sub_tabs.addTab(self.channel_heatmap_widget, "Heatmapa kanálů")

    def _init_dataset(self, initial_sample: int | None = None) -> None:
        self._rebuild_plots()
        self.path_label.setText(f"Soubor: {self.dataset.path}")
        self.metadata_text.setPlainText(self.dataset.summary_text())

        # Naplnit výběr kanálů
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItem("Všechny kanály", None)
        for spec in self.dataset.series_specs:
            self.channel_combo.addItem(f"Pouze {spec.title}", spec.name)
        self.channel_combo.blockSignals(False)

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
        if not self.dataset.series_specs:
            self.empty_plot_label = QtWidgets.QLabel("Nenahrán žádný waveform dataset.")
            self.empty_plot_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.plot_layout.addWidget(self.empty_plot_label, stretch=1)
            return

        self.main_plot = pg.PlotWidget()
        self._configure_plot(self.main_plot, "Waveforms")
        self.main_plot.addLegend(offset=(10, 10))
        self.plot_layout.addWidget(self.main_plot, stretch=1)
        self.pending_autorange = True

    def _position_for_sample(self, requested_sample_id: int) -> int:
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
        return self.dataset.sample_ids[self.current_position]

    def _set_position(self, position: int) -> None:
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
        self._set_position(self._position_for_sample(sample_id))

    def _on_channel_changed(self) -> None:
        self.pending_autorange = True
        self._refresh_plots()

    def _visible_specs(self) -> list:
        """Vrátit série viditelné podle výběru kanálů."""
        selected = self.channel_combo.currentData()
        if selected is None:
            return self.dataset.series_specs
        return [s for s in self.dataset.series_specs if s.name == selected]

    def _next_lock_color(self) -> QtGui.QColor:
        hue = (len(self.locked_samples) * 53) % 360
        return QtGui.QColor.fromHsv(hue, 180, 255)

    def _toggle_current_lock(self) -> None:
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
        if self._current_sample_id() in self.locked_samples:
            self.lock_button.setText("Odemknout aktuální")
        else:
            self.lock_button.setText("Zamknout aktuální")

    def _current_pen(self, index: int):
        return pg.mkPen(CURRENT_COLORS[index % len(CURRENT_COLORS)], width=2.6)

    def _toggle_high_res(self) -> None:
        self.high_res = self.highres_button.isChecked()
        self.pending_autorange = True
        self._refresh_plots()

    def _get_waveform(self, sample_id: int, series_name: str) -> Waveform | None:
        """Return full or decimated waveform based on high_res toggle."""
        if self.high_res:
            return self.dataset.waveform(sample_id, series_name)
        return self.dataset.display_waveform(sample_id, series_name)

    def _draw_plot(self, plot: pg.PlotWidget) -> None:
        # Zjistit, zda tento plot potřebuje autorange
        is_detached = (self.detached_window is not None
                       and plot is self.detached_window.plot_widget)
        need_autorange = (self.detached_window.pending_autorange
                          if is_detached else self.pending_autorange)

        previous_range = None
        if not need_autorange:
            previous_range = plot.plotItem.vb.viewRange()

        plot.clear()
        plot.addLegend(offset=(10, 10))

        current_sample_id = self._current_sample_id()

        visible = self._visible_specs()

        for sample_id, lock_color in self.locked_samples.items():
            for spec in visible:
                waveform = self._get_waveform(sample_id, spec.name)
                if waveform is None:
                    continue
                pen = pg.mkPen(color=lock_color, width=1.2, style=QtCore.Qt.PenStyle.DashLine)
                plot.plot(waveform.x_us, waveform.y_v, pen=pen, name=f"{spec.title} locked {sample_id}")

        for index, spec in enumerate(self.dataset.series_specs):
            if spec not in visible:
                continue
            current_waveform = self._get_waveform(current_sample_id, spec.name)
            if current_waveform is None:
                continue
            plot.plot(
                current_waveform.x_us,
                current_waveform.y_v,
                pen=self._current_pen(index),
                name=f"{spec.title} current {current_sample_id}",
            )
            # V non-high-res režimu zobrazit vyhlazenou křivku
            if not self.high_res:
                wf_full = self.dataset.waveform(current_sample_id, spec.name)
                if wf_full is not None:
                    smooth = centered_moving_average(wf_full.y_v, SMOOTHING_WINDOW)
                    color = CURRENT_COLORS[index % len(CURRENT_COLORS)]
                    smooth_pen = pg.mkPen(color, width=1.2, style=QtCore.Qt.PenStyle.DashLine)
                    plot.plot(wf_full.x_us, smooth, pen=smooth_pen)

        # -- Peak area fill --
        for index, spec in enumerate(self.dataset.series_specs):
            if spec not in visible:
                continue
            wf = self.dataset.waveform(current_sample_id, spec.name)
            if wf is None:
                continue
            fill_data = compute_peak_fill_data(wf)
            if fill_data is None:
                continue
            x_fill, y_fill, bl = fill_data
            color = CURRENT_COLORS[index % len(CURRENT_COLORS)]
            fill_color = QtGui.QColor(color)
            fill_color.setAlpha(80)
            curve_top = pg.PlotCurveItem(x_fill, y_fill, pen=pg.mkPen(None))
            curve_bot = pg.PlotCurveItem(x_fill, bl, pen=pg.mkPen(color, width=0.6, style=QtCore.Qt.PenStyle.DashLine))
            fill = pg.FillBetweenItem(curve_top, curve_bot, brush=fill_color)
            fill.setZValue(-5)
            plot.addItem(curve_top)
            plot.addItem(curve_bot)
            plot.addItem(fill)

        # -- Marker čáry pro peak a delay --
        for index, spec in enumerate(self.dataset.series_specs):
            if spec not in visible:
                continue
            wf = self.dataset.waveform(current_sample_id, spec.name)
            if wf is None:
                continue
            peak_time, amplitude, delay_time, delay_value = compute_waveform_markers(wf)
            color = CURRENT_COLORS[index % len(CURRENT_COLORS)]
            marker_pen = pg.mkPen(color, width=1.0, style=QtCore.Qt.PenStyle.DashLine)
            plot.addItem(pg.InfiniteLine(pos=peak_time, angle=90, pen=marker_pen))
            plot.addItem(pg.InfiniteLine(pos=delay_time, angle=90, pen=marker_pen))
            plot.addItem(pg.InfiniteLine(pos=amplitude, angle=0, pen=marker_pen))
            plot.addItem(pg.InfiniteLine(pos=delay_value, angle=0, pen=marker_pen))

        if previous_range is not None:
            plot.setXRange(*previous_range[0], padding=0)
            plot.setYRange(*previous_range[1], padding=0)
        else:
            plot.autoRange()
            if is_detached:
                self.detached_window.pending_autorange = False
            else:
                self.pending_autorange = False

        plot.setTitle(
            f"Waveforms | aktuální vzorek {current_sample_id} | série {len(self.dataset.series_specs)} | uzamčeno {len(self.locked_samples)}"
        )

    def _compute_measurements(self) -> str:
        sample_id = self._current_sample_id()
        parts = []
        for spec in self._visible_specs():
            wf = self.dataset.waveform(sample_id, spec.name)
            if wf is None:
                continue
            amp, area, delay_value = compute_waveform_measurements(wf)
            parts.append(
                f"{spec.title}: amplituda = {_format_si(amp, 'V')}, "
                f"plocha = {_format_si(area, 'V·s')}, "
                f"delay = {_format_si(delay_value, 'V')}"
            )
        return "  |  ".join(parts) if parts else ""

    def _refresh_plots(self) -> None:
        if self.main_plot is None:
            return
        self._draw_plot(self.main_plot)
        self.measurements_label.setText(self._compute_measurements())
        if self.detached_window is not None and self.detached_window.isVisible():
            self._draw_plot(self.detached_window.plot_widget)

    def _detach_plot(self) -> None:
        """Otevřít oscilogram v samostatném okně."""
        if self.detached_window is not None and self.detached_window.isVisible():
            self.detached_window.raise_()
            self.detached_window.activateWindow()
            return
        self.detached_window = _DetachedPlotWindow(self)
        self._configure_plot(self.detached_window.plot_widget, "Waveforms (detached)")
        self.detached_window.show()
        self.detached_window.pending_autorange = True
        self._draw_plot(self.detached_window.plot_widget)

    def _open_matplotlib(self) -> None:
        self._show_matplotlib_window(self._current_sample_id())

    def _show_matplotlib_window(self, sample_id: int) -> None:
        import matplotlib.pyplot as plt

        locked = dict(self.locked_samples) if self.locked_samples else None
        fig, has_data = _create_waveform_figure(
            self.dataset, sample_id, locked_samples=locked)
        if not has_data:
            plt.close(fig)
            return
        plt.show(block=False)

    def _on_histogram_sample_preview(self, sample_id: int) -> None:
        """Změnit graf na vybraný vzorek bez přepínání tabu."""
        self._set_position(self._position_for_sample(sample_id))

    def _on_histogram_sample_requested(self, sample_id: int) -> None:
        """Přepnout na oscilogram vybraného vzorku z histogramu."""
        self._set_position(self._position_for_sample(sample_id))
        self.sub_tabs.setCurrentIndex(0)

    def _has_active_filter(self) -> bool:
        """Zjistit, zda histogram má aktivní filtr (méně vzorků než celkem)."""
        hist = self.histogram_widget
        return (hist.computed
                and hist.filtered_ids
                and len(hist.filtered_ids) < len(self.dataset.sample_ids))

    def navigate(self, delta: int) -> None:
        """Posunout se o delta vzorků (pro šipky). Respektuje filtr z histogramu."""
        if self._has_active_filter():
            hist = self.histogram_widget
            current_sid = self._current_sample_id()
            try:
                idx = hist.filtered_ids.index(current_sid)
            except ValueError:
                idx = bisect.bisect_left(hist.filtered_ids, current_sid)
                idx = min(idx, len(hist.filtered_ids) - 1)
            new_idx = max(0, min(idx + delta, len(hist.filtered_ids) - 1))
            hist.filtered_position = new_idx
            hist.sample_list.setCurrentRow(new_idx)
            self._set_position(self._position_for_sample(hist.filtered_ids[new_idx]))
        else:
            self._set_position(self.current_position + delta)

    def close_dataset(self) -> None:
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None


class _DetachedPlotWindow(QtWidgets.QWidget):
    """Samostatné okno s oscilogramem, synchronizované s hlavním WaveformTab."""

    def __init__(self, parent_tab: WaveformTab) -> None:
        super().__init__(None, QtCore.Qt.WindowType.Window)
        self.parent_tab = parent_tab
        self.pending_autorange = True
        self.setWindowTitle("Oscilogram (detached)")
        self.resize(1200, 700)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend(offset=(10, 10))
        layout.addWidget(self.plot_widget)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Left:
            self.parent_tab.navigate(-1)
        elif event.key() == QtCore.Qt.Key.Key_Right:
            self.parent_tab.navigate(1)
        else:
            super().keyPressEvent(event)


class WaveformViewer(QtWidgets.QMainWindow):
    def __init__(self, initial_paths: list[str] | None = None, initial_sample: int | None = None) -> None:
        super().__init__()
        self.setWindowTitle("HDF5 Waveform Viewer")
        self.resize(1400, 900)

        self._build_ui()

        if initial_paths:
            if len(initial_paths) > 1:
                self.open_combined(initial_paths, initial_sample)
            else:
                self.open_file(initial_paths[0], initial_sample)

    def _build_ui(self) -> None:
        toolbar = self.addToolBar("Hlavní")
        toolbar.setMovable(False)
        open_action = toolbar.addAction("Otevřít H5")
        open_action.triggered.connect(self._open_file_dialog)
        combine_action = toolbar.addAction("Sloučit H5")
        combine_action.triggered.connect(self._open_combined_dialog)

        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        self.setCentralWidget(self.tab_widget)

        self.statusBar().showMessage("Vyber HDF5 soubor.")

    def _open_file_dialog(self) -> None:
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Vyber HDF5 soubory",
            str(Path.cwd()),
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        for filename in filenames:
            self.open_file(filename)

    def open_file(self, path: str, initial_sample: int | None = None) -> None:
        try:
            dataset = H5WaveformDataset(path)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Nelze otevřít soubor", str(exc))
            return

        if not dataset.sample_ids:
            dataset.close()
            QtWidgets.QMessageBox.warning(self, "Prázdný dataset", "Soubor neobsahuje žádné waveformy.")
            return

        tab = WaveformTab(dataset, initial_sample)
        index = self.tab_widget.addTab(tab, dataset.path.name)
        self.tab_widget.setCurrentIndex(index)
        self.statusBar().showMessage(f"Načteno {dataset.path}")

    def open_combined(self, paths: list[str], initial_sample: int | None = None) -> None:
        """Otevřít více souborů jako jeden sloučený dataset."""
        datasets: list[H5WaveformDataset] = []
        for path in paths:
            try:
                ds = H5WaveformDataset(path)
                if ds.sample_ids:
                    datasets.append(ds)
                else:
                    ds.close()
            except OSError as exc:
                QtWidgets.QMessageBox.critical(self, "Nelze otevřít soubor", f"{path}: {exc}")

        if not datasets:
            return

        combined = CombinedDataset(datasets)
        tab = WaveformTab(combined, initial_sample)
        title = f"{len(datasets)} souborů ({len(combined.sample_ids)} vzorků)"
        index = self.tab_widget.addTab(tab, title)
        self.tab_widget.setCurrentIndex(index)
        self.statusBar().showMessage(
            f"Sloučeno {len(datasets)} souborů, celkem {len(combined.sample_ids)} vzorků"
        )

    def _open_combined_dialog(self) -> None:
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Vyber HDF5 soubory ke sloučení",
            str(Path.cwd()),
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if filenames:
            self.open_combined(filenames)

    def _close_tab(self, index: int) -> None:
        tab = self.tab_widget.widget(index)
        if isinstance(tab, WaveformTab):
            tab.close_dataset()
        self.tab_widget.removeTab(index)

    def _current_tab(self) -> WaveformTab | None:
        tab = self.tab_widget.currentWidget()
        return tab if isinstance(tab, WaveformTab) else None

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        tab = self._current_tab()
        if tab is not None:
            if event.key() == QtCore.Qt.Key.Key_Left:
                tab.navigate(-1)
                return
            elif event.key() == QtCore.Qt.Key.Key_Right:
                tab.navigate(1)
                return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            if isinstance(tab, WaveformTab):
                tab.close_dataset()
        super().closeEvent(event)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyQtGraph viewer / matplotlib preview pro waveformy uložené v HDF5.")
    parser.add_argument("path", nargs="*", help="Cesty k HDF5 souborům.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Vzorek, na kterém se má viewer po startu otevřít (GUI) nebo vygenerovat náhled (--preview).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Místo GUI vygeneruj PNG náhled(y) waveformů přes matplotlib.",
    )
    parser.add_argument(
        "--preview-all",
        action="store_true",
        help="Vygeneruj náhledy pro všechny vzorky v souboru.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Adresář pro výstupní PNG soubory (výchozí: vedle H5 souboru).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI výstupních náhledů (výchozí: 200).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # -- režim matplotlib náhledů ----------------------------------------------
    if args.preview or args.preview_all:
        if not args.path:
            print("Chyba: pro --preview/--preview-all je nutné zadat cestu k H5 souboru.")
            return 1

        preview_path = args.path[0]
        if args.preview_all:
            render_all_previews(preview_path, output_dir=args.output_dir, dpi=args.dpi)
        else:
            dataset = H5WaveformDataset(preview_path)
            try:
                sid = args.sample if args.sample is not None else dataset.sample_ids[0]
                out_dir = Path(args.output_dir) if args.output_dir else None
                out_file = (out_dir / f"{dataset.path.stem}_{sid}.png") if out_dir else None
                if out_dir:
                    out_dir.mkdir(parents=True, exist_ok=True)
                render_waveform_preview(dataset, sid, output_path=out_file, dpi=args.dpi)
            finally:
                dataset.close()
        return 0

    # -- režim GUI -------------------------------------------------------------
    if not HAS_PYQTGRAPH:
        print("Chyba: pro GUI režim je nutné mít nainstalovaný pyqtgraph a PyQt.")
        return 1

    app = QtWidgets.QApplication(sys.argv)
    viewer = WaveformViewer(
        initial_paths=args.path if args.path else None,
        initial_sample=args.sample,
    )
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
