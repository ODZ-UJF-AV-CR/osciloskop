#!/usr/bin/env python3

#%% Imports

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from glob2 import glob
from matplotlib.widgets import RectangleSelector


#%% Nastaveni


INPUT_FILES = [
    #"/home/roman/TEST/20260328_101618_DAY2_RUN6_mereni_EM.h5", # run 6 - s plexy, umisteno obracene vuci plexy. Smerem dolu
    #"/home/roman/TEST/20260328_103218_DAY2_RUN6_mereni_EM.h5",

    # "/home/roman/TEST/20260329_170117_DAY3_RUN3_LIF.h5",
    # "/home/roman/TEST/20260329_164912_DAY3_RUN3_LIF.h5",
    # "/home/roman/TEST/20260329_163333_DAY3_RUN3_LIF.h5"

    #"/home/roman/TEST/20260329_165601_DAY3_RUN3_LICO.h5",
    #"/home/roman/TEST/20260329_164405_DAY3_RUN3_LICO.h5",
    #"/home/roman/TEST/20260329_163314_DAY3_RUN3_LICO.h5"

    #"/home/roman/TEST/20260329_170117_DAY3_RUN3_LIF.h5",
    #"/home/roman/TEST/20260329_164912_DAY3_RUN3_LIF.h5",
    #"/home/roman/TEST/20260329_163333_DAY3_RUN3_LIF.h5"

    *glob("/home/roman/TEST/*_DAY3_RUN2_EM.h5"), # velke diody, sikmo, plexi
    #*glob("/home/roman/TEST/*_DAY3_RUN3_EM.h5"),
    #*glob("/home/roman/TEST/*_DAY3_RUN4_EM.h5"),

]

print(INPUT_FILES)

BINS = 200
SAVE_PLOT = True
OUTPUT_FILE = Path("waveform_max_shifted_hist2d.png")
SHIFT_TIME_US = 25  # microseconds
SMOOTHING_WINDOW = 11  # approximately 30 bins, must stay odd
MIN_AMPLITUDE = 0.01
CURVE_SLOPE = 0.5
CURVE_CURVATURE = 0.0


#%% Data

expanded_input_files = []
seen_input_files = set()
input_alerts = []

for input_pattern in INPUT_FILES:
    pattern = str(input_pattern).strip()
    if any(char in pattern for char in "*?[]"):
        matched_files = [Path(path) for path in sorted(glob(pattern))]
        if not matched_files:
            input_alerts.append(f"ALERT: no files matched pattern {pattern}")
    else:
        matched_files = [Path(pattern)]

    for matched_file in matched_files:
        resolved_file = matched_file.expanduser()
        if not resolved_file.exists():
            input_alerts.append(f"ALERT: file does not exist: {resolved_file}")
            continue
        if resolved_file in seen_input_files:
            continue
        seen_input_files.add(resolved_file)
        expanded_input_files.append(resolved_file)

for alert_message in input_alerts:
    print(alert_message)

if not expanded_input_files:
    print("ALERT: no existing input files available, nothing to process.")


def centered_moving_average(values, window_size):
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("SMOOTHING_WINDOW must be a positive odd integer.")

    half_window = window_size // 2
    padded_values = np.pad(values, (half_window, half_window), mode="edge")
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(padded_values, kernel, mode="valid")

max_values1 = []
shifted_values1 = []
max_values2 = []
shifted_values2 = []
files_sample_counts = []

for input_file in expanded_input_files:
    try:
        f = h5py.File(input_file, "r")
    except BlockingIOError as exc:
        print(f"ALERT: skipping {input_file} because file is locked: {exc}")
        continue

    with f:
        if "/CHAN1" not in f or "/CHAN2" not in f:
            missing_channels = []
            if "/CHAN1" not in f:
                missing_channels.append("CHAN1")
            if "/CHAN2" not in f:
                missing_channels.append("CHAN2")
            print(
                f"ALERT: skipping {input_file} because missing channel(s): "
                f"{', '.join(missing_channels)}"
            )
            continue

        ch1 = f["/CHAN1"]
        ch2 = f["/CHAN2"]

        current_ch1_name = str(ch1.attrs["CHANNEL"])
        current_ch2_name = str(ch2.attrs["CHANNEL"])

        current_yinc1 = float(ch1.attrs["YINC"])
        current_yorg1 = float(ch1.attrs["YORIGIN"])
        current_yinc2 = float(ch2.attrs["YINC"])
        current_yorg2 = float(ch2.attrs["YORIGIN"])
        current_xinc1 = float(ch1.attrs["XINC"])
        current_xinc2 = float(ch2.attrs["XINC"])

        if not max_values1:
            ch1_name = current_ch1_name
            ch2_name = current_ch2_name
            yinc1 = current_yinc1
            yorg1 = current_yorg1
            yinc2 = current_yinc2
            yorg2 = current_yorg2
            xinc1 = current_xinc1
            xinc2 = current_xinc2
        else:
            if current_ch1_name != ch1_name or current_ch2_name != ch2_name:
                raise ValueError(f"Inconsistent channel names in {input_file}")

        ids = sorted(set(ch1.keys()) & set(ch2.keys()), key=int)

        for sample_id in ids:
            data1 = np.asarray(ch1[sample_id], dtype=float).reshape(-1)
            data2 = np.asarray(ch2[sample_id], dtype=float).reshape(-1)

            wave1 = (data1 - 128.0 - current_yorg1) * current_yinc1
            wave2 = (data2 - 128.0 - current_yorg2) * current_yinc2
            smooth_wave1 = centered_moving_average(wave1, SMOOTHING_WINDOW)
            smooth_wave2 = centered_moving_average(wave2, SMOOTHING_WINDOW)

            max_idx1 = np.argmax(smooth_wave1)
            max_val1 = wave1[max_idx1]
            shift_samples1 = int(SHIFT_TIME_US * 1e-6 / current_xinc1)
            shifted_idx1 = max_idx1 + shift_samples1
            if shifted_idx1 < len(wave1):
                shifted_val1 = wave1[shifted_idx1]
            else:
                shifted_val1 = np.nan  # or handle out of bounds

            max_idx2 = np.argmax(smooth_wave2)
            max_val2 = wave2[max_idx2]
            shift_samples2 = int(SHIFT_TIME_US * 1e-6 / current_xinc2)
            shifted_idx2 = max_idx2 + shift_samples2
            if shifted_idx2 < len(wave2):
                shifted_val2 = wave2[shifted_idx2]
            else:
                shifted_val2 = np.nan

            max_values1.append(max_val1)
            shifted_values1.append(shifted_val1)
            max_values2.append(max_val2)
            shifted_values2.append(shifted_val2)

        files_sample_counts.append((input_file, len(ids)))

max_values1 = np.asarray(max_values1)
shifted_values1 = np.asarray(shifted_values1)
max_values2 = np.asarray(max_values2)
shifted_values2 = np.asarray(shifted_values2)

print("Input files:")
for input_file, sample_count in files_sample_counts:
    print(f"  {input_file} ({sample_count} paired samples)")
print(f"Total paired samples: {len(max_values1)}")

if max_values1.size == 0:
    print("ALERT: no paired samples found in the selected input files.")
    raise SystemExit(0)

print(
    f"{ch1_name}: max min={max_values1.min():.4g}, max={max_values1.max():.4g} V, "
    f"shifted min={np.nanmin(shifted_values1):.4g}, max={np.nanmax(shifted_values1):.4g} V"
)
print(
    f"{ch2_name}: max min={max_values2.min():.4g}, max={max_values2.max():.4g} V, "
    f"shifted min={np.nanmin(shifted_values2):.4g}, max={np.nanmax(shifted_values2):.4g} V"
)

figure_title = expanded_input_files[0].name


#%% Graf

fig, ax = plt.subplots(figsize=(14, 10))
fig.subplots_adjust(top=0.92)
fig.suptitle(figure_title)

valid1 = ~np.isnan(shifted_values1)
valid2 = ~np.isnan(shifted_values2)

ch2_x = max_values2[valid2]
ch2_y = shifted_values2[valid2]

curve_values = CURVE_SLOPE * ch2_x + CURVE_CURVATURE * (ch2_x ** 2)
amplitude_mask = ch2_x >= MIN_AMPLITUDE
below_curve_mask = (ch2_y < curve_values) & amplitude_mask
total_particles = ch2_x.size
particles_below_curve = int(np.sum(below_curve_mask))
particles_above_curve = total_particles - particles_below_curve
below_curve_pct = 100.0 * particles_below_curve / total_particles if total_particles else 0.0

ax.scatter(
    ch2_x[~below_curve_mask],
    ch2_y[~below_curve_mask],
    s=12,
    alpha=0.2,
    color="blue",
    label=f"Other particles: {particles_above_curve}",
)
ax.scatter(
    ch2_x[below_curve_mask],
    ch2_y[below_curve_mask],
    s=16,
    alpha=0.5,
    color="red",
    label=(
        f"Neutron classified: {particles_below_curve} "
        f"({below_curve_pct:.1f} %)"
    ),
)

x_curve = np.linspace(0, np.nanmax(ch2_x), 500)
y_curve = CURVE_SLOPE * x_curve + CURVE_CURVATURE * (x_curve ** 2)
ax.plot(x_curve, y_curve, color="black", linewidth=2, label="Threshold curve")

print(f"Total {ch2_name} particles: {total_particles}")
print(f"{ch2_name} particles below curve and above amplitude cut: {particles_below_curve}")

ax.set_facecolor("white")
ax.set_xlabel(f"Max amplitude {ch2_name} [V]")
ax.set_ylabel(f"Shifted amplitude {ch2_name} [V] ({SHIFT_TIME_US} us later)")
ax.set_title(f"{ch2_name}: max vs shifted amplitude")
#ax.set_xlim(MIN_AMPLITUDE, np.nanmax(ch2_x) * 1.02)
#ax.set_ylim(0, max(0.2, np.nanmax(ch2_y) * 1.05))
ax.grid(alpha=0.2)
ax.legend()
ax.text(
    0.02,
    0.98,
    (
        f"Total particles: {total_particles} | "
        f"Neutron classified: {particles_below_curve} ({below_curve_pct:.1f} %, "
        f"for max >= {MIN_AMPLITUDE:.3g} V)"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    color="red",
    fontsize=12,
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
)

if SAVE_PLOT:
    fig.savefig(OUTPUT_FILE, dpi=150)
    print(f"Graph saved to: {OUTPUT_FILE}")

plt.show()


#%% ROI Selection

# ROI selection over the shared scatter plot

def onselect(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print(f"Selected ROI: x=({x1:.4g}, {x2:.4g}), y=({y1:.4g}, {y2:.4g})")
    roi_mask = (
        (ch2_x >= min(x1, x2))
        & (ch2_x <= max(x1, x2))
        & (ch2_y >= min(y1, y2))
        & (ch2_y <= max(y1, y2))
    )
    selected_events = np.sum(roi_mask)
    print(f"Selected events: {selected_events}")

# Add RectangleSelector to the shared axis
rect_selector = RectangleSelector(ax, onselect, useblit=True,
                                  button=[1],  # left mouse button
                                  minspanx=5, minspany=5,
                                  spancoords='pixels',
                                  interactive=True)

plt.show()

# %%
