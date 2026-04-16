#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import vxi11

os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
os.environ.setdefault("QT_API", "pyside6")

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from waveform_viewer import H5WaveformDataset, render_all_previews


pg.setConfigOptions(antialias=False)

DEFAULT_SCOPE_IP = "10.11.111.36"
DEFAULT_SCOPE_NAME = "osc1"
DEFAULT_OUTDIR = "~/captures"
DEFAULT_SAMPLES = 14000
DEFAULT_TRANSPORT = "raw_socket"
TRANSPORT_LABELS = {"vxi11": "VXI11", "raw_socket": "Raw socket"}
CHANNEL_CANDIDATES = tuple(f"CHAN{i}" for i in range(1, 9))
TRACE_COLORS = ["#ffd166", "#4cc9f0", "#ef476f", "#95d67b", "#f78c6b", "#c77dff", "#4ade80", "#f97316"]


def sanitize_measurement_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return safe.strip("_")


def decode_wave_block(block: bytes) -> bytes:
    if block.startswith(b"#") and len(block) >= 3:
        digits = int(block[1:2])
        data_len = int(block[2 : 2 + digits])
        start = 2 + digits
        return block[start : start + data_len]
    return block.rstrip(b"\n")


@dataclass(frozen=True)
class ScopeConfig:
    name: str
    ip: str
    transport: str = DEFAULT_TRANSPORT


@dataclass(frozen=True)
class AcquisitionConfig:
    scopes: tuple[ScopeConfig, ...]
    channels: tuple[str, ...]
    outdir: Path
    samples: int
    timeout_s: float
    measurement_name: str
    high_res: bool
    trigger_mode: str
    trigger_count: int
    trigger_source: str
    save_png: bool


@dataclass(frozen=True)
class CaptureResult:
    scope_name: str
    ip: str
    h5_path: Path
    frames: int
    channels: tuple[str, ...]


@dataclass(frozen=True)
class ChannelDescriptor:
    name: str
    enabled: bool


@dataclass(frozen=True)
class ChannelWaveform:
    channel: str
    status: str
    x_us: np.ndarray
    y_v: np.ndarray
    xinc: float
    yinc: float


@dataclass(frozen=True)
class LiveSnapshot:
    scope_name: str
    ip: str
    transport: str
    waveforms: tuple[ChannelWaveform, ...]


class RigolScope:
    def __init__(self, config: ScopeConfig, timeout_ms: int = 3000) -> None:
        self.config = config
        self.name = config.name
        self.ip = config.ip
        self.transport = config.transport
        self.timeout_ms = timeout_ms
        self.idn: str | None = None

        if self.transport == "vxi11":
            self.vxi = vxi11.Instrument(f"TCPIP::{self.ip}::INSTR")
            self.vxi.timeout = timeout_ms
        elif self.transport == "raw_socket":
            self.vxi = None
        else:
            raise ValueError(f"Unsupported transport: {self.transport}")

    def transport_label(self) -> str:
        return TRANSPORT_LABELS.get(self.transport, self.transport)

    def write(self, cmd: str) -> None:
        if self.transport == "vxi11":
            self.vxi.write(cmd)
            return
        self._socket_query(cmd, expect_reply=False)

    def ask(self, cmd: str) -> str:
        if self.transport == "vxi11":
            return self.vxi.ask(cmd)
        return self._socket_query(cmd, expect_reply=True).decode("utf-8", errors="replace")

    def query_binary(self, cmd: str) -> bytes:
        if self.transport == "vxi11":
            self.vxi.write(cmd)
            return self.vxi.read_raw()
        return self._socket_query(cmd, expect_reply=True, expect_binary=True)

    def _socket_query(self, cmd: str, expect_reply: bool, expect_binary: bool = False) -> bytes:
        with socket.create_connection((self.ip, 5555), timeout=self.timeout_ms / 1000.0) as sock:
            sock.settimeout(self.timeout_ms / 1000.0)
            sock.sendall(cmd.encode("ascii") + b"\n")
            if not expect_reply:
                return b""

            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                data = b"".join(chunks)
                if expect_binary and data.startswith(b"#") and len(data) >= 2:
                    digits = int(data[1:2])
                    if len(data) >= 2 + digits:
                        data_len = int(data[2 : 2 + digits])
                        total = 2 + digits + data_len
                        if len(data) >= total:
                            return data[:total]
                if not expect_binary and data.endswith(b"\n"):
                    return data
            return b"".join(chunks)

    def identify(self) -> str:
        if self.idn is None:
            self.idn = self.ask("*IDN?").strip()
        return self.idn

    def stop(self) -> None:
        self.write(":STOP")

    def run(self) -> None:
        self.write(":RUN")

    def single(self) -> None:
        self.write(":SING")

    def force_trigger(self) -> None:
        self.write(":TFORce")

    def set_record_mode(self) -> None:
        self.write(":FUNC:WRM RECORD")

    def trigger_status(self) -> str:
        return self.ask(":TRIG:STAT?").strip()

    def replay_end_frame(self) -> int:
        return int(float(self.ask(":FUNC:WREP:FEND?").strip()))

    def configure_capture(self, samples: int, high_res: bool) -> None:
        self.write(f":ACQuire:MDEPth {samples}")
        self.write(":TRIGger:SWEep NORMal")
        self.write(f":WAV:MODE {'MAX' if high_res else 'NORM'}")
        self.write(f":WAV:POIN {samples if high_res else 7000}")

    def configure_waveform_readout(self, channel: str, samples: int, high_res: bool) -> int:
        waveform_points = samples if high_res else min(samples, 7000)
        self.write(f":WAV:SOUR {channel}")
        self.write(":WAV:FORM BYTE")
        self.write(f":WAV:MODE {'MAX' if high_res else 'NORM'}")
        self.write(f":WAV:POIN {waveform_points}")
        return waveform_points

    def discover_channels(self) -> tuple[ChannelDescriptor, ...]:
        channels: list[ChannelDescriptor] = []
        for channel in CHANNEL_CANDIDATES:
            try:
                resp = self.ask(f":{channel}:DISP?").strip()
            except Exception:
                continue
            channels.append(ChannelDescriptor(name=channel, enabled=resp != "0"))
        return tuple(channels)

    def fetch_waveform(self, channel: str, samples: int, high_res: bool) -> ChannelWaveform:
        self.configure_waveform_readout(channel, samples, high_res)
        xinc = float(self.ask(":WAV:XINC?").strip())
        yinc = float(self.ask(":WAV:YINC?").strip())
        yorigin = float(self.ask(":WAVeform:YORigin?").strip())
        xorigin = float(self.ask(":WAVeform:XORigin?").strip())
        status = self.trigger_status().upper()

        raw_block = self.query_binary(":WAV:DATA?")
        wave = decode_wave_block(raw_block)
        if not wave:
            raise RuntimeError(f"{self.name}: empty waveform for {channel}")

        data = np.frombuffer(wave, dtype=np.uint8).astype(np.float64, copy=False)
        y_v = (data - 128.0 - yorigin) * yinc
        x_us = np.arange(data.size, dtype=np.float64) * xinc * 1e6
        x_us += xorigin * 1e6
        return ChannelWaveform(channel=channel, status=status, x_us=x_us, y_v=y_v, xinc=xinc, yinc=yinc)


def prepare_scope_for_capture(scope: RigolScope, config: AcquisitionConfig, log: Callable[[str], None] | None = None) -> None:
    if log:
        log(f"{scope.name}: transport connected ({scope.transport_label()})")
    scope.configure_capture(config.samples, config.high_res)
    scope.stop()
    scope.set_record_mode()
    time.sleep(0.2)


def arm_scopes_for_capture(
    scopes: list[RigolScope],
    config: AcquisitionConfig,
    log: Callable[[str], None] | None = None,
    status: Callable[[str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> datetime:
    def should_stop() -> bool:
        return bool(stop_requested and stop_requested())

    for index, scope in enumerate(scopes, start=1):
        if status:
            status(f"Preparing {scope.name} ({index}/{len(scopes)})")
        prepare_scope_for_capture(scope, config, log=log)

    if config.trigger_mode == "single":
        for scope in scopes:
            scope.single()
        if config.trigger_source == "force":
            time.sleep(0.2)
            for scope in scopes:
                if should_stop():
                    break
                scope.force_trigger()
    else:
        for scope in scopes:
            scope.run()
        if config.trigger_source == "force":
            for idx in range(config.trigger_count):
                if should_stop():
                    break
                time.sleep(0.15)
                for scope in scopes:
                    scope.force_trigger()
                if log:
                    log(f"Forced trigger {idx + 1}/{config.trigger_count} on {len(scopes)} scope(s)")

    return datetime.now(UTC)


def wait_for_scopes(
    scopes: list[RigolScope],
    config: AcquisitionConfig,
    progress: Callable[[str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> datetime:
    def should_stop() -> bool:
        return bool(stop_requested and stop_requested())

    deadline = time.monotonic() + config.timeout_s
    wanted_frames = max(1, config.trigger_count)

    while time.monotonic() < deadline:
        if should_stop():
            raise RuntimeError("Acquisition cancelled.")

        all_done = True
        status_parts: list[str] = []
        for scope in scopes:
            trig_status = scope.trigger_status().upper()
            try:
                frames = scope.replay_end_frame()
            except Exception:
                frames = 0
            status_parts.append(f"{scope.name}: {trig_status} frames={frames}")
            done = (trig_status == "STOP" or frames >= 1) if config.trigger_mode == "single" else (frames >= wanted_frames)
            if not done:
                all_done = False

        if progress:
            progress(" | ".join(status_parts))
        if all_done:
            break
        time.sleep(0.2)
    else:
        raise TimeoutError(f"Capture timed out after {config.timeout_s:.1f}s")

    end_time = datetime.now(UTC)
    for scope in scopes:
        scope.stop()
    time.sleep(0.2)
    return end_time


def download_scope_data(
    scope: RigolScope,
    config: AcquisitionConfig,
    start_time: datetime,
    end_time: datetime,
    log: Callable[[str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> CaptureResult:
    def should_stop() -> bool:
        return bool(stop_requested and stop_requested())

    wanted_frames = max(1, config.trigger_count)
    available = {item.name for item in scope.discover_channels()}
    channels = tuple(channel for channel in config.channels if channel in available)
    if not channels:
        raise RuntimeError(f"{scope.name}: none of the selected channels are available.")

    config.outdir.mkdir(parents=True, exist_ok=True)
    filename = start_time.strftime("%Y%m%d_%H%M%S")
    if config.measurement_name:
        filename = f"{filename}_{config.measurement_name}"
    h5_path = config.outdir / f"{filename}_{scope.name}.h5"

    saved_channels: list[str] = []
    final_frame_count = 0

    with h5py.File(h5_path, "w") as hf:
        hf.attrs["CAPTURING"] = (end_time - start_time).total_seconds()
        hf.attrs["START_TIME"] = start_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        hf.attrs["START_TIMESTAMP"] = start_time.timestamp()
        hf.attrs["END_TIME"] = end_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        hf.attrs["END_TIMESTAMP"] = end_time.timestamp()
        hf.attrs["SCOPE_NAME"] = scope.name
        hf.attrs["IP"] = scope.ip
        hf.attrs["MEASUREMENT_NAME"] = config.measurement_name
        hf.attrs["TRIGGER_MODE"] = config.trigger_mode
        hf.attrs["TRIGGER_SOURCE"] = config.trigger_source
        hf.attrs["REQUESTED_TRIGGERS"] = config.trigger_count
        hf.attrs["TRANSPORT"] = scope.transport

        for channel in channels:
            if log:
                log(f"{scope.name}: downloading {channel}")

            scope.configure_waveform_readout(channel, config.samples, config.high_res)
            xinc = float(scope.ask(":WAV:XINC?").strip())
            yinc = float(scope.ask(":WAV:YINC?").strip())
            trig_level = float(scope.ask(":TRIGger:EDGe:LEVel?").strip())
            trig_channel = scope.ask(":TRIGger:EDGe:SOUR?").strip()
            yorigin = float(scope.ask(":WAVeform:YORigin?").strip())
            xorigin = float(scope.ask(":WAVeform:XORigin?").strip())
            frames = max(1, scope.replay_end_frame())
            if config.trigger_mode == "multi":
                frames = min(frames, wanted_frames)
            final_frame_count = max(final_frame_count, frames)

            channel_group = hf.create_group(channel)
            channel_group.attrs["FRAMES"] = frames
            channel_group.attrs["XINC"] = xinc
            channel_group.attrs["YINC"] = yinc
            channel_group.attrs["YORIGIN"] = yorigin
            channel_group.attrs["XORIGIN"] = xorigin
            channel_group.attrs["TRIG_LEVEL"] = trig_level
            channel_group.attrs["TRIG_CHANNEL"] = trig_channel
            channel_group.attrs["CHANNEL"] = channel

            preamble = scope.ask(":WAV:PRE?").strip()
            last_wave = None

            for frame_index in range(1, frames + 1):
                if should_stop():
                    raise RuntimeError("Acquisition cancelled during download.")

                scope.write(f":FUNC:WREP:FCUR {frame_index}")
                for _ in range(40):
                    if scope.ask(":FUNC:WREP:FCUR?").strip() == str(frame_index):
                        break
                    time.sleep(0.02)

                ctag_raw = scope.ask(":FUNCtion:WREPlay:CTAG?").strip()
                try:
                    ctag = float(eval(ctag_raw, {}, {}))
                except Exception:
                    ctag = 0.0

                raw_block = scope.query_binary(":WAV:DATA?")
                wave = decode_wave_block(raw_block)
                if not wave and last_wave is not None:
                    wave = last_wave
                last_wave = wave

                dset = channel_group.create_dataset(str(frame_index), data=np.frombuffer(wave, dtype=np.uint8))
                dset.attrs["frame_index"] = frame_index
                dset.attrs["channel"] = channel
                dset.attrs["scope_name"] = scope.name
                dset.attrs["CTAG"] = ctag
                trg_time = start_time + timedelta(seconds=ctag)
                dset.attrs["TRG_TIME"] = trg_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                dset.attrs["TRG_TIMESTAMP"] = trg_time.timestamp()
                dset.attrs["preamble"] = preamble

            saved_channels.append(channel)

    if config.save_png:
        png_dir = h5_path.parent / h5_path.stem
        if log:
            log(f"{scope.name}: rendering PNG previews into {png_dir}")
        render_all_previews(h5_path, output_dir=png_dir)

    if log:
        log(f"{scope.name}: saved {h5_path}")
    return CaptureResult(scope_name=scope.name, ip=scope.ip, h5_path=h5_path, frames=final_frame_count, channels=tuple(saved_channels))


class ChannelDiscoveryWorker(QtCore.QObject):
    finished = QtCore.Signal(object, object)
    failed = QtCore.Signal(str)

    def __init__(self, scope_config: ScopeConfig) -> None:
        super().__init__()
        self.scope_config = scope_config

    @QtCore.Slot()
    def run(self) -> None:
        try:
            scope = RigolScope(self.scope_config, timeout_ms=1500)
            channels = scope.discover_channels()
            self.finished.emit(self.scope_config, channels)
        except Exception:
            self.failed.emit(traceback.format_exc())


class AcquisitionWorker(QtCore.QObject):
    log_message = QtCore.Signal(str)
    status_message = QtCore.Signal(str)
    finished = QtCore.Signal(list)
    failed = QtCore.Signal(str)

    def __init__(self, config: AcquisitionConfig) -> None:
        super().__init__()
        self.config = config
        self._stop_requested = False

    @QtCore.Slot()
    def run(self) -> None:
        results: list[CaptureResult] = []
        try:
            scopes: list[RigolScope] = []
            for index, scope_cfg in enumerate(self.config.scopes, start=1):
                if self._stop_requested:
                    raise RuntimeError("Acquisition cancelled.")
                self.status_message.emit(f"Connecting to {scope_cfg.name} ({index}/{len(self.config.scopes)})")
                scopes.append(RigolScope(scope_cfg, timeout_ms=3000))

            start_time = arm_scopes_for_capture(scopes, self.config, self.log_message.emit, self.status_message.emit, lambda: self._stop_requested)
            end_time = wait_for_scopes(scopes, self.config, self.status_message.emit, lambda: self._stop_requested)

            for scope in scopes:
                self.status_message.emit(f"Downloading {scope.name}")
                results.append(
                    download_scope_data(scope, self.config, start_time, end_time, self.log_message.emit, lambda: self._stop_requested)
                )
            self.finished.emit(results)
        except Exception:
            self.failed.emit(traceback.format_exc())

    @QtCore.Slot()
    def request_stop(self) -> None:
        self._stop_requested = True
        self.log_message.emit("Stop requested. Waiting for current instrument operation to finish.")


class LivePreviewWorker(QtCore.QObject):
    snapshot_ready = QtCore.Signal(object)
    status_message = QtCore.Signal(str)
    log_message = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, scope_config: ScopeConfig, channels: tuple[str, ...], samples: int, high_res: bool, interval_ms: int) -> None:
        super().__init__()
        self.scope_config = scope_config
        self.channels = channels
        self.samples = samples
        self.high_res = high_res
        self.interval_ms = interval_ms
        self._stop_requested = False

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.status_message.emit(f"Connecting live preview to {self.scope_config.name}")
            scope = RigolScope(self.scope_config, timeout_ms=2000)
            self.log_message.emit(
                f"Live preview transport connected: {scope.name} ({scope.ip}) via {scope.transport_label()}"
            )
            while not self._stop_requested:
                waveforms = tuple(scope.fetch_waveform(channel, self.samples, self.high_res) for channel in self.channels)
                self.snapshot_ready.emit(
                    LiveSnapshot(scope_name=scope.name, ip=scope.ip, transport=scope.transport, waveforms=waveforms)
                )
                self.status_message.emit(f"Live preview | {scope.name} | {len(waveforms)} channel(s)")
                time.sleep(max(0.1, self.interval_ms / 1000.0))
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.finished.emit()

    @QtCore.Slot()
    def request_stop(self) -> None:
        self._stop_requested = True


class CaptureBrowserPanel(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.dataset: H5WaveformDataset | None = None
        self.current_position = 0

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Sample"))
        self.sample_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sample_slider.setEnabled(False)
        self.sample_slider.valueChanged.connect(self._on_slider_changed)
        controls.addWidget(self.sample_slider, stretch=1)
        self.sample_spinbox = QtWidgets.QSpinBox()
        self.sample_spinbox.setEnabled(False)
        self.sample_spinbox.valueChanged.connect(self._on_spinbox_changed)
        controls.addWidget(self.sample_spinbox)
        self.sample_label = QtWidgets.QLabel("0 / 0")
        controls.addWidget(self.sample_label)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("#101418")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("left", "Amplitude", units="V")
        self.plot.setLabel("bottom", "Time", units="us")
        self.info = QtWidgets.QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(180)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.info)

    def clear(self) -> None:
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None
        self.plot.clear()
        self.info.clear()
        self.sample_slider.setEnabled(False)
        self.sample_spinbox.setEnabled(False)
        self.sample_label.setText("0 / 0")

    def load_capture(self, result: CaptureResult) -> None:
        self.load_file(result.h5_path)

    def load_file(self, path: str | Path) -> None:
        if self.dataset is not None:
            self.dataset.close()
        self.dataset = H5WaveformDataset(path)
        self.plot.clear()
        if not self.dataset.sample_ids:
            self.info.setPlainText(f"{path}\nNo samples in file.")
            return
        self.sample_slider.setEnabled(True)
        self.sample_spinbox.setEnabled(True)
        self.sample_slider.blockSignals(True)
        self.sample_slider.setRange(0, len(self.dataset.sample_ids) - 1)
        self.sample_slider.blockSignals(False)
        self.sample_spinbox.blockSignals(True)
        self.sample_spinbox.setRange(self.dataset.sample_ids[0], self.dataset.sample_ids[-1])
        self.sample_spinbox.blockSignals(False)
        self._set_position(0)

    def _set_position(self, position: int) -> None:
        if self.dataset is None:
            return
        position = max(0, min(position, len(self.dataset.sample_ids) - 1))
        self.current_position = position
        sample_id = self.dataset.sample_ids[position]
        self.sample_slider.blockSignals(True)
        self.sample_slider.setValue(position)
        self.sample_slider.blockSignals(False)
        self.sample_spinbox.blockSignals(True)
        self.sample_spinbox.setValue(sample_id)
        self.sample_spinbox.blockSignals(False)
        self.sample_label.setText(f"{position + 1} / {len(self.dataset.sample_ids)}")
        self._refresh_plot(sample_id)

    def _on_slider_changed(self, value: int) -> None:
        self._set_position(value)

    def _on_spinbox_changed(self, value: int) -> None:
        if self.dataset is None:
            return
        nearest = min(range(len(self.dataset.sample_ids)), key=lambda idx: abs(self.dataset.sample_ids[idx] - value))
        self._set_position(nearest)

    def _refresh_plot(self, sample_id: int) -> None:
        assert self.dataset is not None
        self.plot.clear()
        for idx, spec in enumerate(self.dataset.series_specs):
            waveform = self.dataset.display_waveform(sample_id, spec.name, max_points=2500)
            if waveform is None:
                continue
            self.plot.plot(waveform.x_us, waveform.y_v, pen=pg.mkPen(TRACE_COLORS[idx % len(TRACE_COLORS)], width=2.2), name=spec.title)

        self.plot.setTitle(f"{self.dataset.path.name} | sample {sample_id}")
        self.plot.autoRange()
        self.info.setPlainText(self.dataset.summary_text())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Oscilloscope Capture App")
        self.resize(1800, 1000)

        self.worker_thread: QtCore.QThread | None = None
        self.worker: AcquisitionWorker | None = None
        self.preview_thread: QtCore.QThread | None = None
        self.preview_worker: LivePreviewWorker | None = None
        self.discovery_thread: QtCore.QThread | None = None
        self.discovery_worker: ChannelDiscoveryWorker | None = None

        self.discovered_channels: dict[str, tuple[ChannelDescriptor, ...]] = {}
        self.current_channel_scope_key: str | None = None

        self._build_ui()
        self._add_scope_row(DEFAULT_SCOPE_NAME, DEFAULT_SCOPE_IP, DEFAULT_TRANSPORT)
        self.scope_table.selectRow(0)

    def _build_ui(self) -> None:
        self.main_tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.main_tabs)

        self.config_tab = QtWidgets.QWidget()
        self.live_tab = QtWidgets.QWidget()
        self.captures_tab = QtWidgets.QWidget()
        self.main_tabs.addTab(self.config_tab, "Config")
        self.main_tabs.addTab(self.live_tab, "Live")
        self.main_tabs.addTab(self.captures_tab, "Captures")

        self._build_config_tab()
        self._build_live_tab()
        self._build_captures_tab()
        self.statusBar().showMessage("Ready.")

    def _build_config_tab(self) -> None:
        root = QtWidgets.QVBoxLayout(self.config_tab)

        scope_box = QtWidgets.QGroupBox("Oscilloscopes")
        root.addWidget(scope_box)
        scope_layout = QtWidgets.QVBoxLayout(scope_box)

        scope_toolbar = QtWidgets.QHBoxLayout()
        self.add_scope_button = QtWidgets.QPushButton("Add")
        self.add_scope_button.clicked.connect(lambda: self._add_scope_row("", "", DEFAULT_TRANSPORT))
        scope_toolbar.addWidget(self.add_scope_button)
        self.remove_scope_button = QtWidgets.QPushButton("Remove selected")
        self.remove_scope_button.clicked.connect(self._remove_selected_scope_row)
        scope_toolbar.addWidget(self.remove_scope_button)
        self.detect_channels_button = QtWidgets.QPushButton("Detect channels")
        self.detect_channels_button.clicked.connect(self._detect_channels_for_selected_scope)
        scope_toolbar.addWidget(self.detect_channels_button)
        scope_toolbar.addStretch(1)
        scope_layout.addLayout(scope_toolbar)

        self.scope_table = QtWidgets.QTableWidget(0, 3)
        self.scope_table.setHorizontalHeaderLabels(["Name", "IP", "Transport"])
        self.scope_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.scope_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.scope_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.scope_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.scope_table.setMaximumHeight(200)
        self.scope_table.itemSelectionChanged.connect(self._refresh_channels_for_selected_scope)
        scope_layout.addWidget(self.scope_table)

        content = QtWidgets.QSplitter()
        content.setOrientation(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(content, stretch=1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QGridLayout(left)
        content.addWidget(left)

        self.measurement_edit = QtWidgets.QLineEdit()
        left_layout.addWidget(QtWidgets.QLabel("Measurement"), 0, 0)
        left_layout.addWidget(self.measurement_edit, 0, 1)

        self.outdir_edit = QtWidgets.QLineEdit(os.path.expanduser(DEFAULT_OUTDIR))
        left_layout.addWidget(QtWidgets.QLabel("Output dir"), 0, 2)
        left_layout.addWidget(self.outdir_edit, 0, 3)
        self.outdir_button = QtWidgets.QPushButton("Browse")
        self.outdir_button.clicked.connect(self._pick_output_dir)
        left_layout.addWidget(self.outdir_button, 0, 4)

        self.samples_spin = QtWidgets.QSpinBox()
        self.samples_spin.setRange(100, 10000000)
        self.samples_spin.setValue(DEFAULT_SAMPLES)
        left_layout.addWidget(QtWidgets.QLabel("Samples"), 1, 0)
        left_layout.addWidget(self.samples_spin, 1, 1)

        self.timeout_spin = QtWidgets.QDoubleSpinBox()
        self.timeout_spin.setRange(1.0, 36000.0)
        self.timeout_spin.setValue(30.0)
        self.timeout_spin.setSuffix(" s")
        left_layout.addWidget(QtWidgets.QLabel("Timeout"), 1, 2)
        left_layout.addWidget(self.timeout_spin, 1, 3)

        self.trigger_mode_combo = QtWidgets.QComboBox()
        self.trigger_mode_combo.addItem("Single trigger", "single")
        self.trigger_mode_combo.addItem("Multiple triggers", "multi")
        left_layout.addWidget(QtWidgets.QLabel("Trigger mode"), 2, 0)
        left_layout.addWidget(self.trigger_mode_combo, 2, 1)

        self.trigger_count_spin = QtWidgets.QSpinBox()
        self.trigger_count_spin.setRange(1, 10000)
        self.trigger_count_spin.setValue(1)
        left_layout.addWidget(QtWidgets.QLabel("Trigger count"), 2, 2)
        left_layout.addWidget(self.trigger_count_spin, 2, 3)

        self.trigger_source_combo = QtWidgets.QComboBox()
        self.trigger_source_combo.addItem("Scope / external trigger", "scope")
        self.trigger_source_combo.addItem("Force trigger from app", "force")
        left_layout.addWidget(QtWidgets.QLabel("Trigger source"), 3, 0)
        left_layout.addWidget(self.trigger_source_combo, 3, 1)

        self.high_res_check = QtWidgets.QCheckBox("High-resolution waveform")
        self.high_res_check.setChecked(True)
        left_layout.addWidget(self.high_res_check, 3, 2)
        self.save_png_check = QtWidgets.QCheckBox("Generate PNG previews")
        self.save_png_check.setChecked(True)
        left_layout.addWidget(self.save_png_check, 3, 3)

        self.capture_start_button = QtWidgets.QPushButton("Start capture")
        self.capture_start_button.clicked.connect(self._start_capture)
        left_layout.addWidget(self.capture_start_button, 4, 0)
        self.capture_stop_button = QtWidgets.QPushButton("Stop capture")
        self.capture_stop_button.clicked.connect(self._request_stop)
        self.capture_stop_button.setEnabled(False)
        left_layout.addWidget(self.capture_stop_button, 4, 1)

        self.config_status = QtWidgets.QLabel("Select a scope and detect channels.")
        left_layout.addWidget(self.config_status, 5, 0, 1, 5)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        content.addWidget(right)

        self.capture_channels_group = QtWidgets.QGroupBox("Capture channels")
        self.capture_channels_layout = QtWidgets.QVBoxLayout(self.capture_channels_group)
        right_layout.addWidget(self.capture_channels_group)

        self.config_log = QtWidgets.QPlainTextEdit()
        self.config_log.setReadOnly(True)
        right_layout.addWidget(self.config_log, stretch=1)

        content.setSizes([1100, 500])

    def _build_live_tab(self) -> None:
        root = QtWidgets.QVBoxLayout(self.live_tab)

        controls = QtWidgets.QHBoxLayout()
        root.addLayout(controls)

        controls.addWidget(QtWidgets.QLabel("Scope"))
        self.live_scope_combo = QtWidgets.QComboBox()
        self.live_scope_combo.currentIndexChanged.connect(self._sync_live_scope_channels)
        controls.addWidget(self.live_scope_combo)

        controls.addWidget(QtWidgets.QLabel("Refresh"))
        self.live_interval_spin = QtWidgets.QSpinBox()
        self.live_interval_spin.setRange(100, 5000)
        self.live_interval_spin.setValue(500)
        self.live_interval_spin.setSuffix(" ms")
        controls.addWidget(self.live_interval_spin)

        self.live_detect_button = QtWidgets.QPushButton("Refresh channels")
        self.live_detect_button.clicked.connect(self._detect_channels_for_live_scope)
        controls.addWidget(self.live_detect_button)

        self.live_button = QtWidgets.QPushButton("Start live preview")
        self.live_button.clicked.connect(self._toggle_live_preview)
        controls.addWidget(self.live_button)
        controls.addStretch(1)

        live_split = QtWidgets.QSplitter()
        live_split.setOrientation(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(live_split, stretch=1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        live_split.addWidget(left)

        self.live_plot = pg.PlotWidget()
        self.live_plot.setBackground("#101418")
        self.live_plot.showGrid(x=True, y=True, alpha=0.25)
        self.live_plot.setLabel("left", "Amplitude", units="V")
        self.live_plot.setLabel("bottom", "Time", units="us")
        left_layout.addWidget(self.live_plot, stretch=1)

        self.live_info = QtWidgets.QPlainTextEdit()
        self.live_info.setReadOnly(True)
        self.live_info.setMaximumHeight(160)
        left_layout.addWidget(self.live_info)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        live_split.addWidget(right)

        self.live_channels_group = QtWidgets.QGroupBox("Live channels")
        self.live_channels_layout = QtWidgets.QVBoxLayout(self.live_channels_group)
        right_layout.addWidget(self.live_channels_group)

        self.live_status_label = QtWidgets.QLabel("Detect channels to start live preview.")
        right_layout.addWidget(self.live_status_label)
        right_layout.addStretch(1)

        live_split.setSizes([1400, 300])

    def _build_captures_tab(self) -> None:
        root = QtWidgets.QVBoxLayout(self.captures_tab)
        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        self.capture_browser = CaptureBrowserPanel()
        splitter.addWidget(self.capture_browser)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        splitter.addWidget(right)

        right_layout.addWidget(QtWidgets.QLabel("Saved captures"))
        self.results_list = QtWidgets.QListWidget()
        self.results_list.currentItemChanged.connect(self._load_selected_capture)
        self.results_list.itemDoubleClicked.connect(self._open_selected_file)
        right_layout.addWidget(self.results_list, stretch=1)

        self.open_last_button = QtWidgets.QPushButton("Open last H5")
        self.open_last_button.clicked.connect(self._open_last_file)
        self.open_last_button.setEnabled(False)
        right_layout.addWidget(self.open_last_button)

        splitter.setSizes([1300, 500])

    def _transport_editor(self, value: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        for key, label in TRANSPORT_LABELS.items():
            combo.addItem(label, key)
        idx = combo.findData(value)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.currentIndexChanged.connect(self._refresh_scope_dependent_ui)
        return combo

    def _add_scope_row(self, name: str, ip: str, transport: str) -> None:
        row = self.scope_table.rowCount()
        self.scope_table.insertRow(row)
        self.scope_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
        self.scope_table.setItem(row, 1, QtWidgets.QTableWidgetItem(ip))
        self.scope_table.setCellWidget(row, 2, self._transport_editor(transport))
        self._refresh_scope_dependent_ui()

    def _remove_selected_scope_row(self) -> None:
        row = self.scope_table.currentRow()
        if row >= 0:
            self.scope_table.removeRow(row)
            self._refresh_scope_dependent_ui()

    def _scope_configs(self) -> list[ScopeConfig]:
        configs: list[ScopeConfig] = []
        for row in range(self.scope_table.rowCount()):
            name_item = self.scope_table.item(row, 0)
            ip_item = self.scope_table.item(row, 1)
            transport_widget = self.scope_table.cellWidget(row, 2)
            name = (name_item.text() if name_item else "").strip()
            ip = (ip_item.text() if ip_item else "").strip()
            transport = transport_widget.currentData() if isinstance(transport_widget, QtWidgets.QComboBox) else DEFAULT_TRANSPORT
            if name and ip:
                configs.append(ScopeConfig(name=name, ip=ip, transport=transport))
        return configs

    def _scope_key(self, scope: ScopeConfig) -> str:
        return f"{scope.name}|{scope.ip}|{scope.transport}"

    def _selected_scope_config(self) -> ScopeConfig | None:
        row = self.scope_table.currentRow()
        configs = self._scope_configs()
        return configs[row] if 0 <= row < len(configs) else None

    def _refresh_scope_dependent_ui(self) -> None:
        current_live = self.live_scope_combo.currentData()
        self.live_scope_combo.blockSignals(True)
        self.live_scope_combo.clear()
        for scope in self._scope_configs():
            self.live_scope_combo.addItem(f"{scope.name} ({scope.ip}, {TRANSPORT_LABELS.get(scope.transport, scope.transport)})", scope)
        if self.live_scope_combo.count():
            index = 0
            for idx in range(self.live_scope_combo.count()):
                if self.live_scope_combo.itemData(idx) == current_live:
                    index = idx
                    break
            self.live_scope_combo.setCurrentIndex(index)
        self.live_scope_combo.blockSignals(False)
        self._sync_live_scope_channels()

    def _refresh_channels_for_selected_scope(self) -> None:
        scope = self._selected_scope_config()
        if scope is None:
            return
        descriptors = self.discovered_channels.get(self._scope_key(scope), ())
        self.capture_channel_boxes = self._set_checkbox_group(self.capture_channels_layout, descriptors, checked_by_default=True)

    def _set_checkbox_group(self, layout: QtWidgets.QVBoxLayout, descriptors: tuple[ChannelDescriptor, ...], checked_by_default: bool) -> dict[str, QtWidgets.QCheckBox]:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        boxes: dict[str, QtWidgets.QCheckBox] = {}
        if not descriptors:
            layout.addWidget(QtWidgets.QLabel("No channels detected."))
            layout.addStretch(1)
            return boxes

        for descriptor in descriptors:
            checkbox = QtWidgets.QCheckBox(descriptor.name)
            checkbox.setChecked(descriptor.enabled if checked_by_default else False)
            layout.addWidget(checkbox)
            boxes[descriptor.name] = checkbox
        layout.addStretch(1)
        return boxes

    def _apply_detected_channels(self, scope: ScopeConfig, descriptors: tuple[ChannelDescriptor, ...]) -> None:
        key = self._scope_key(scope)
        self.discovered_channels[key] = descriptors
        if self._selected_scope_config() == scope:
            self.capture_channel_boxes = self._set_checkbox_group(self.capture_channels_layout, descriptors, checked_by_default=True)
        current_live_scope = self.live_scope_combo.currentData()
        if current_live_scope == scope:
            self.live_channel_boxes = self._set_checkbox_group(self.live_channels_layout, descriptors, checked_by_default=True)
        self.current_channel_scope_key = key
        self.config_status.setText(f"Detected {len(descriptors)} channel(s) on {scope.name}.")
        self.live_status_label.setText(f"Detected {len(descriptors)} channel(s) on {scope.name}.")

    def _detect_channels_for_scope(self, scope: ScopeConfig) -> None:
        if self.discovery_worker is not None:
            return
        self.config_status.setText(f"Detecting channels on {scope.name}...")
        self.live_status_label.setText(f"Detecting channels on {scope.name}...")
        self.discovery_thread = QtCore.QThread(self)
        self.discovery_worker = ChannelDiscoveryWorker(scope)
        self.discovery_worker.moveToThread(self.discovery_thread)
        self.discovery_thread.started.connect(self.discovery_worker.run)
        self.discovery_worker.finished.connect(self._handle_channels_detected)
        self.discovery_worker.failed.connect(self._handle_channel_discovery_failed)
        self.discovery_worker.finished.connect(self.discovery_thread.quit)
        self.discovery_worker.failed.connect(self.discovery_thread.quit)
        self.discovery_thread.finished.connect(self._cleanup_discovery)
        self.discovery_thread.start()

    def _detect_channels_for_selected_scope(self) -> None:
        scope = self._selected_scope_config()
        if scope is not None:
            self._detect_channels_for_scope(scope)

    def _detect_channels_for_live_scope(self) -> None:
        scope = self.live_scope_combo.currentData()
        if scope is not None:
            self._detect_channels_for_scope(scope)

    @QtCore.Slot(object, object)
    def _handle_channels_detected(self, scope: ScopeConfig, descriptors: tuple[ChannelDescriptor, ...]) -> None:
        self._append_log(
            f"Detected channels on {scope.name}: {', '.join(d.name + ('*' if d.enabled else '') for d in descriptors) or 'none'}"
        )
        self._apply_detected_channels(scope, descriptors)

    @QtCore.Slot(str)
    def _handle_channel_discovery_failed(self, trace: str) -> None:
        self._append_log(trace.rstrip())
        self.config_status.setText("Channel detection failed.")
        self.live_status_label.setText("Channel detection failed.")

    def _cleanup_discovery(self) -> None:
        if self.discovery_worker is not None:
            self.discovery_worker.deleteLater()
            self.discovery_worker = None
        if self.discovery_thread is not None:
            self.discovery_thread.deleteLater()
            self.discovery_thread = None

    def _sync_live_scope_channels(self) -> None:
        scope = self.live_scope_combo.currentData()
        if scope is None:
            return
        descriptors = self.discovered_channels.get(self._scope_key(scope), ())
        self.live_channel_boxes = self._set_checkbox_group(self.live_channels_layout, descriptors, checked_by_default=True)

    def _selected_capture_channels(self) -> tuple[str, ...]:
        return tuple(name for name, box in getattr(self, "capture_channel_boxes", {}).items() if box.isChecked())

    def _selected_live_channels(self) -> tuple[str, ...]:
        return tuple(name for name, box in getattr(self, "live_channel_boxes", {}).items() if box.isChecked())

    def _pick_output_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Output directory", self.outdir_edit.text())
        if directory:
            self.outdir_edit.setText(directory)

    def _current_config(self) -> AcquisitionConfig:
        scopes = tuple(self._scope_configs())
        if not scopes:
            raise ValueError("Add at least one oscilloscope.")
        channels = self._selected_capture_channels()
        if not channels:
            raise ValueError("Select at least one capture channel.")
        return AcquisitionConfig(
            scopes=scopes,
            channels=channels,
            outdir=Path(os.path.expanduser(self.outdir_edit.text().strip() or DEFAULT_OUTDIR)),
            samples=self.samples_spin.value(),
            timeout_s=self.timeout_spin.value(),
            measurement_name=sanitize_measurement_name(self.measurement_edit.text()),
            high_res=self.high_res_check.isChecked(),
            trigger_mode=self.trigger_mode_combo.currentData(),
            trigger_count=self.trigger_count_spin.value(),
            trigger_source=self.trigger_source_combo.currentData(),
            save_png=self.save_png_check.isChecked(),
        )

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.config_log.appendPlainText(f"[{timestamp}] {message}")

    def _set_running_state(self, running: bool) -> None:
        self.capture_start_button.setEnabled(not running)
        self.capture_stop_button.setEnabled(running)
        self.add_scope_button.setEnabled(not running)
        self.remove_scope_button.setEnabled(not running)
        self.detect_channels_button.setEnabled(not running)
        self.live_button.setEnabled(not running or self.preview_worker is not None)

    def _start_capture(self) -> None:
        if self.preview_worker is not None:
            QtWidgets.QMessageBox.warning(self, "Live preview active", "Stop live preview before starting capture.")
            return
        try:
            config = self._current_config()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid setup", str(exc))
            return

        self._append_log("Starting capture.")
        self.statusBar().showMessage("Connecting to oscilloscopes...")
        self._set_running_state(True)

        self.worker_thread = QtCore.QThread(self)
        self.worker = AcquisitionWorker(config)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self._append_log)
        self.worker.status_message.connect(self.statusBar().showMessage)
        self.worker.finished.connect(self._handle_capture_finished)
        self.worker.failed.connect(self._handle_capture_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_capture_worker)
        self.worker_thread.start()

    def _request_stop(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()

    def _toggle_live_preview(self) -> None:
        if self.preview_worker is not None:
            self.preview_worker.request_stop()
            return
        if self.worker is not None:
            QtWidgets.QMessageBox.warning(self, "Capture running", "Live preview cannot start during capture.")
            return
        scope = self.live_scope_combo.currentData()
        channels = self._selected_live_channels()
        if scope is None:
            QtWidgets.QMessageBox.warning(self, "No scope", "Select a scope for live preview.")
            return
        if not channels:
            QtWidgets.QMessageBox.warning(self, "No channels", "Select at least one live channel.")
            return

        self.preview_thread = QtCore.QThread(self)
        self.preview_worker = LivePreviewWorker(scope, channels, self.samples_spin.value(), self.high_res_check.isChecked(), self.live_interval_spin.value())
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_worker.snapshot_ready.connect(self._update_live_preview)
        self.preview_worker.status_message.connect(self.statusBar().showMessage)
        self.preview_worker.log_message.connect(self._append_log)
        self.preview_worker.failed.connect(self._handle_live_preview_failed)
        self.preview_worker.finished.connect(self.preview_thread.quit)
        self.preview_worker.finished.connect(self._cleanup_live_preview)
        self.preview_thread.finished.connect(self.preview_thread.deleteLater)
        self.preview_thread.start()
        self.live_button.setText("Stop live preview")

    @QtCore.Slot(object)
    def _update_live_preview(self, snapshot: LiveSnapshot) -> None:
        self.live_plot.clear()
        info_lines = [f"Scope: {snapshot.scope_name} ({snapshot.ip})", f"Transport: {TRANSPORT_LABELS.get(snapshot.transport, snapshot.transport)}"]
        for idx, waveform in enumerate(snapshot.waveforms):
            color = TRACE_COLORS[idx % len(TRACE_COLORS)]
            self.live_plot.plot(waveform.x_us, waveform.y_v, pen=pg.mkPen(color, width=1.8), name=waveform.channel)
            info_lines.append(f"{waveform.channel}: {waveform.status}, samples={waveform.x_us.size}, xinc={waveform.xinc}, yinc={waveform.yinc}")
        self.live_plot.setTitle(f"{snapshot.scope_name} | {len(snapshot.waveforms)} channel(s)")
        self.live_info.setPlainText("\n".join(info_lines))
        self.main_tabs.setCurrentWidget(self.live_tab)

    @QtCore.Slot(str)
    def _handle_live_preview_failed(self, trace: str) -> None:
        self._append_log(trace.rstrip())
        QtWidgets.QMessageBox.critical(self, "Live preview failed", trace)

    @QtCore.Slot()
    def _cleanup_live_preview(self) -> None:
        if self.preview_worker is not None:
            self.preview_worker.deleteLater()
            self.preview_worker = None
        self.preview_thread = None
        self.live_button.setText("Start live preview")
        self.statusBar().showMessage("Ready.")

    def _cleanup_capture_worker(self) -> None:
        self._set_running_state(False)
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
            self.worker_thread = None
        if self.preview_worker is None:
            self.statusBar().showMessage("Ready.")

    @QtCore.Slot(list)
    def _handle_capture_finished(self, results: list[CaptureResult]) -> None:
        self.statusBar().showMessage("Capture complete.")
        self._append_log("Capture finished.")
        self.open_last_button.setEnabled(bool(results))

        for result in results:
            item = QtWidgets.QListWidgetItem(f"{result.scope_name} | {result.h5_path.name}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(result.h5_path))
            self.results_list.insertItem(0, item)
            self._append_log(f"{result.scope_name}: {result.frames} frame(s), channels={', '.join(result.channels)}, file={result.h5_path}")

        if self.results_list.count():
            self.results_list.setCurrentRow(0)
        self.main_tabs.setCurrentWidget(self.captures_tab)

    @QtCore.Slot(str)
    def _handle_capture_failed(self, trace: str) -> None:
        self.statusBar().showMessage("Capture failed.")
        self._append_log(trace.rstrip())
        QtWidgets.QMessageBox.critical(self, "Capture failed", trace)

    def _open_selected_file(self, item: QtWidgets.QListWidgetItem) -> None:
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if path:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def _load_selected_capture(self, current: QtWidgets.QListWidgetItem | None, previous: QtWidgets.QListWidgetItem | None = None) -> None:
        if current is None:
            return
        path = current.data(QtCore.Qt.ItemDataRole.UserRole)
        if path:
            self.capture_browser.load_file(path)

    def _open_last_file(self) -> None:
        if self.results_list.count() > 0:
            self._open_selected_file(self.results_list.item(0))

    def _shutdown_threads(self) -> None:
        if self.preview_worker is not None:
            self.preview_worker.request_stop()
        if self.worker is not None:
            self.worker.request_stop()
        if self.discovery_thread is not None:
            self.discovery_thread.quit()
            self.discovery_thread.wait(1000)
        if self.preview_thread is not None:
            self.preview_thread.quit()
            self.preview_thread.wait(1000)
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(1000)
        self.capture_browser.clear()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._shutdown_threads()
        super().closeEvent(event)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PySide6 app for multi-oscilloscope waveform capture.")
    parser.add_argument("--scope", action="append", default=[], help="Initial scope in NAME=IP or NAME=IP,transport format.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window._shutdown_threads)

    if args.scope:
        window.scope_table.setRowCount(0)
        for item in args.scope:
            if "=" not in item:
                continue
            name, rest = item.split("=", 1)
            if "," in rest:
                ip, transport = rest.split(",", 1)
            else:
                ip, transport = rest, DEFAULT_TRANSPORT
            window._add_scope_row(name.strip(), ip.strip(), transport.strip())
        if window.scope_table.rowCount():
            window.scope_table.selectRow(0)

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
