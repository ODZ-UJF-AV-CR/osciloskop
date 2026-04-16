#!/usr/bin/env python3

#%% Imports

import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


#%% Nastaveni

INPUT_FILES = [
    #"/home/roman/TEST/20260327_220052_RUN9_mereni_EM.h5",
    # "/home/roman/TEST/20260327_212709_RUN5_mereni_EM.h5",
    # "/home/roman/TEST/20260327_210224_RUN5_mereni_EM.h5",
    # "/home/roman/TEST/20260327_223301_RUN10_mereni_EM.h5",
    # "/home/roman/TEST/20260327_220052_RUN9_mereni_EM.h5"
    #"/home/roman/TEST/20260327_*_RUN11_mereni_EM.h5"
    #"/home/roman/TEST/20260328_083109_DAY2_RUN3_mereni_EM.h5",  # run 3 - s plexi
    #"/home/roman/TEST/20260328_090048_DAY2_RUN4_mereni_EM.h5",  # run 4 - bez plexi, 
    #"/home/roman/TEST/20260328_093916_DAY2_RUN5_mereni_EM.h5" # run 5 - kanystr s vodou
    "/home/roman/TEST/20260328_101618_DAY2_RUN6_mereni_EM.h5", # run 6 - s plexy, umisteno obracene vuci plexy. Smerem dolu
    "/home/roman/TEST/20260328_103218_DAY2_RUN6_mereni_EM.h5",
]

# INPUT_FILES = [
#   "/home/roman/TEST/20260327_220049_RUN9_mereni_LICO.h5",
#   "/home/roman/TEST/20260327_222337_RUN10_mereni_LICO.h5",
#   "/home/roman/TEST/20260327_224825_RUN11_mereni_LICO.h5"
# ]

# INPUT_FILES = [
#     "/home/roman/TEST/20260327_220049_RUN9_mereni_LIF.h5",
#     "/home/roman/TEST/20260327_222337_RUN10_mereni_LIF.h5",
#     "/home/roman/TEST/20260327_224825_RUN11_mereni_LIF.h5"
# ]



BINS = 200
SAVE_PLOT = True
OUTPUT_FILE = Path("waveform_max_hist2d.png")
SINGLE_DETECTOR_OUTPUT_FILE = Path("waveform_single_detector_energy_hist.png")
SMOOTHING_WINDOW = 7
SINGLE_DETECTOR_THRESHOLD_V = 0.020


#%% Data

expanded_input_files = []
seen_input_files = set()
input_alerts = []

for input_pattern in INPUT_FILES:
    pattern = str(input_pattern).strip()
    if any(char in pattern for char in "*?[]"):
        matched_files = [Path(path) for path in sorted(glob.glob(pattern))]
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

amplitude1 = []
amplitude2 = []
area1 = []
area2 = []
files_sample_counts = []

for input_file in expanded_input_files:
    with h5py.File(input_file, "r") as f:
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

        if not amplitude1:
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

            amplitude1.append(np.max(smooth_wave1))
            amplitude2.append(np.max(smooth_wave2))

            min1 = np.mean(np.concatenate([wave1[:30]]))
            min2 = np.mean(np.concatenate([wave2[:30]]))

            area1.append(np.trapezoid(np.clip(wave1, min=min1), dx=current_xinc1))
            area2.append(np.trapezoid(np.clip(wave2, min=min2), dx=current_xinc2))

        files_sample_counts.append((input_file, len(ids)))

amplitude1 = np.asarray(amplitude1)
amplitude2 = np.asarray(amplitude2)
area1 = np.asarray(area1)
area2 = np.asarray(area2)

print("Input files:")
for input_file, sample_count in files_sample_counts:
    print(f"  {input_file} ({sample_count} paired samples)")
print(f"Total paired samples: {len(amplitude1)}")

if amplitude1.size == 0:
    print("ALERT: no paired samples found in the selected input files.")
    raise SystemExit(0)

print(
    f"{ch1_name}: YINC={yinc1}, YORIGIN={yorg1}, "
    f"amplituda min={amplitude1.min():.4g}, max={amplitude1.max():.4g} V, "
    f"plocha min={area1.min():.4g}, max={area1.max():.4g} V*s"
)
print(
    f"{ch2_name}: YINC={yinc2}, YORIGIN={yorg2}, "
    f"amplituda min={amplitude2.min():.4g}, max={amplitude2.max():.4g} V, "
    f"plocha min={area2.min():.4g}, max={area2.max():.4g} V*s"
)

if len(expanded_input_files) == 1:
    figure_title = expanded_input_files[0].name
else:
    figure_title = f"{len(expanded_input_files)} files from {expanded_input_files}"


#%% Graf

fig, axes = plt.subplots(2, 3, figsize=(24, 12), constrained_layout=True)
fig.suptitle(figure_title)
axes[1, 1].sharex(axes[1, 0])
axes[1, 1].sharey(axes[1, 0])
axes[1, 2].sharex(axes[1, 0])
axes[1, 2].sharey(axes[1, 0])
BINS = 150

cmap = plt.cm.hot.copy()
cmap.set_bad("white")

shared_area_min = min(area1.min(), area2.min())
shared_area_max = max(area1.max(), area2.max())
shared_amplitude_min = min(amplitude1.min(), amplitude2.min())
shared_amplitude_max = max(amplitude1.max(), amplitude2.max())
shared_area_edges = np.linspace(shared_area_min, shared_area_max, BINS + 1)
shared_amplitude_edges = np.linspace(shared_amplitude_min, shared_amplitude_max, BINS + 1)

hist_amplitude, xedges, yedges = np.histogram2d(amplitude1, amplitude2, bins=BINS)
hist_amplitude = hist_amplitude.T

hist_area, _, _ = np.histogram2d(
    area1,
    area2,
    bins=(shared_area_edges, shared_area_edges),
)
hist_area = hist_area.T

area_amplitude_histograms = []
for area, amplitude in [(area1, amplitude1), (area2, amplitude2)]:
    hist_counts, _, _ = np.histogram2d(
        area,
        amplitude,
        bins=(shared_area_edges, shared_amplitude_edges),
    )
    area_amplitude_histograms.append(hist_counts.T)

channel1_no_peak_mask = amplitude1 <= SINGLE_DETECTOR_THRESHOLD_V
channel1_no_peak_area = area2[channel1_no_peak_mask]
channel1_no_peak_amplitude = amplitude2[channel1_no_peak_mask]

print(
    f"{ch1_name}-no-peak events "
    f"(threshold {SINGLE_DETECTOR_THRESHOLD_V:.4g} V): "
    f"{channel1_no_peak_area.size}"
)

histograms_for_scale = [hist_amplitude, hist_area, *area_amplitude_histograms]
hist_channel1_no_peak = None
if channel1_no_peak_area.size:
    hist_channel1_no_peak, _, _ = np.histogram2d(
        channel1_no_peak_area,
        channel1_no_peak_amplitude,
        bins=(shared_area_edges, shared_amplitude_edges),
    )
    hist_channel1_no_peak = hist_channel1_no_peak.T
    histograms_for_scale.append(hist_channel1_no_peak)

shared_hist_vmax = max(hist.max() for hist in histograms_for_scale)
shared_hist_norm = colors.LogNorm(vmin=1, vmax=shared_hist_vmax)

masked_counts = np.ma.masked_less_equal(hist_amplitude, 0)
mesh = axes[0, 0].pcolormesh(
    xedges,
    yedges,
    masked_counts,
    cmap=cmap,
    norm=shared_hist_norm,
    shading="auto",
)
axes[0, 0].set_facecolor("white")
axes[0, 0].set_xlabel(f"Peak amplitude {ch1_name} [V]")
axes[0, 0].set_ylabel(f"Peak amplitude {ch2_name} [V]")
axes[0, 0].set_title(f"{ch1_name} vs {ch2_name} peak amplitude")
axes[0, 0].grid(alpha=0.15)

masked_counts = np.ma.masked_less_equal(hist_area, 0)
mesh = axes[0, 1].pcolormesh(
    shared_area_edges,
    shared_area_edges,
    masked_counts,
    cmap=cmap,
    norm=shared_hist_norm,
    shading="auto",
)
axes[0, 1].set_facecolor("white")
axes[0, 1].set_xlabel(f"Area {ch1_name} [V*s]")
axes[0, 1].set_ylabel(f"Area {ch2_name} [V*s]")
axes[0, 1].set_title(f"{ch1_name} vs {ch2_name} area")
axes[0, 1].grid(alpha=0.15)

for ax, hist_counts, channel_name in [
    (axes[1, 0], area_amplitude_histograms[0], ch1_name),
    (axes[1, 1], area_amplitude_histograms[1], ch2_name),
]:
    masked_counts = np.ma.masked_less_equal(hist_counts, 0)
    mesh = ax.pcolormesh(
        shared_area_edges,
        shared_amplitude_edges,
        masked_counts,
        cmap=cmap,
        norm=shared_hist_norm,
        shading="auto",
    )
    ax.set_facecolor("white")
    ax.set_xlabel("Area [V*s]")
    ax.set_ylabel("Peak amplitude [V]")
    ax.set_title(f"{channel_name}: area vs peak amplitude")
    ax.set_xlim(shared_area_min, shared_area_max)
    ax.set_ylim(shared_amplitude_min, shared_amplitude_max)
    ax.set_aspect("auto")
    ax.set_xlim(-0.1e-5, 2.5e-5)
    ax.set_ylim(-0.002, 0.35)
    ax.grid(alpha=0.15)

ax_channel1_no_peak = axes[1, 2]
if hist_channel1_no_peak is not None:
    masked_counts = np.ma.masked_less_equal(hist_channel1_no_peak, 0)
    ax_channel1_no_peak.pcolormesh(
        shared_area_edges,
        shared_amplitude_edges,
        masked_counts,
        cmap=cmap,
        norm=shared_hist_norm,
        shading="auto",
    )
    ax_channel1_no_peak.set_facecolor("white")
    ax_channel1_no_peak.set_xlabel(f"Area {ch2_name} [V*s]")
    ax_channel1_no_peak.set_ylabel(f"Peak amplitude {ch2_name} [V]")
    ax_channel1_no_peak.set_title(
        f"{ch2_name}: area vs peak amplitude for {ch1_name}-no-peak events"
    )
    ax_channel1_no_peak.set_xlim(-0.1e-5, 2.5e-5)
    ax_channel1_no_peak.set_ylim(-0.002, 0.35)
    ax_channel1_no_peak.grid(alpha=0.15)
else:
    ax_channel1_no_peak.set_facecolor("white")
    ax_channel1_no_peak.set_title(
        f"{ch2_name}: area vs peak amplitude for {ch1_name}-no-peak events"
    )
    ax_channel1_no_peak.text(
        0.5,
        0.5,
        "No events matched\nthe current filter.",
        ha="center",
        va="center",
        transform=ax_channel1_no_peak.transAxes,
    )
    ax_channel1_no_peak.grid(alpha=0.15)

shared_colorbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=shared_hist_norm, cmap=cmap),
    ax=axes.ravel().tolist(),
    label="Event count (log)",
)
shared_colorbar.ax.set_ylabel("Event count (log)")

if SAVE_PLOT:
    fig.savefig(OUTPUT_FILE, dpi=150)
    print(f"Graph saved to: {OUTPUT_FILE}")

plt.show()


#%% Single-detector energy histogram

single_detector_mask = (
    (amplitude2 > SINGLE_DETECTOR_THRESHOLD_V)
    & (amplitude1 <= SINGLE_DETECTOR_THRESHOLD_V)
)
single_detector_energy = area2[single_detector_mask]

print(
    f"Single-detector {ch2_name} events "
    f"(threshold {SINGLE_DETECTOR_THRESHOLD_V:.4g} V): "
    f"{single_detector_energy.size}"
)

if single_detector_energy.size:
    fig_single, ax_single = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax_single.hist(single_detector_energy, bins=BINS, color="tab:green", alpha=0.85)
    ax_single.set_xlabel(f"{ch2_name} energy proxy from area [V*s]")
    ax_single.set_ylabel("Event count")
    ax_single.set_title(
        f"{ch2_name}-only events: energy histogram "
        f"({ch1_name} <= {SINGLE_DETECTOR_THRESHOLD_V:.4g} V, "
        f"{ch2_name} > {SINGLE_DETECTOR_THRESHOLD_V:.4g} V)"
    )
    ax_single.grid(alpha=0.15)

    if SAVE_PLOT:
        fig_single.savefig(SINGLE_DETECTOR_OUTPUT_FILE, dpi=150)
        print(f"Graph saved to: {SINGLE_DETECTOR_OUTPUT_FILE}")

    plt.show()
else:
    print("No single-detector events matched the current threshold.")


#%% Channel-1-no-peak area vs peak-amplitude histogram

    # %%
