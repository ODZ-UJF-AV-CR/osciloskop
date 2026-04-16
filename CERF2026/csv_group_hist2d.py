#!/usr/bin/env python3

# %%
from glob import glob
from math import ceil
from pathlib import Path
from datetime import datetime, timezone

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


#%% Nastaveni

prefix = "/home/roman/mnt/kapybara/storage/experiments/archive/2026/03_cerf/"

GROUPS = [
    {
        "name": "DAY3 RUN 2 - PLEXI",
        "patterns": [
             "/home/roman/Stažené/run2bigLi2CO3_USB1_20260329_172927.csv",
        ],
    },
    {
        "name": "DAY3 RUN 3 - NIC",
        "patterns": [
            "/home/roman/Stažené/run3bigLi2CO3_USB1_20260329_183506.csv",
        ],
    },
    {
        "name": "DAY3 RUN 4 - VODA",
        "patterns": [
            "/home/roman/Stažené/run4bigLi2CO3_USB1_20260329_193540.csv",
        ],
    },
]

# /home/roman/Stažené/run7bigLi2CO3_USB1_20260329_214826.csv /home/roman/Stažené/run6bigLi2CO3_USB1_20260329_211228.csv /home/roman/Stažené/run6AbigLi2CO3_USB1_20260329_211632.csv

GROUPS = [
    {
        "name": "DAY3 RUN 6 - PLEXI",
        "patterns": [
            "/home/roman/Stažené/run6*big*",
        ],
    },
    {
        "name": "DAY3 RUN 7 - NIC",
        "patterns": [
            "/home/roman/Stažené/run7*big*",
        ],
    },
]


CSV_DELIMITERS = [",", ";", "\t"]
COLUMN_X = 2  # third column, zero-based index
COLUMN_Y = 3  # fourth column, zero-based index
BINS = (1000, 1000)
HIST_RANGE = [[0, 3000], [0, 3000]]
MIN_COUNT_X = 500
LINE_SLOPE = 0.65
SAVE_PLOT = True
OUTPUT_FILE = Path("csv_group_hist2d.png")


#%% Helpers

def load_csv_columns(csv_file):
    filtered_lines = []
    time_values = []

    with open(csv_file, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            stripped_line = raw_line.strip()
            if stripped_line.startswith("$TIME,"):
                time_text = stripped_line.split(",", 1)[1]
                time_values.append(datetime.fromisoformat(time_text.replace("Z", "+00:00")))
                continue
            if not stripped_line.startswith("$C"):
                continue
            filtered_lines.append(stripped_line[2:].lstrip())

    if not filtered_lines:
        raise ValueError(f"No rows starting with $C in {csv_file}")
    if not time_values:
        raise ValueError(f"No $TIME rows found in {csv_file}")

    for delimiter in CSV_DELIMITERS:
        try:
            data = np.genfromtxt(
                filtered_lines,
                delimiter=delimiter,
                dtype=float,
                invalid_raise=False,
            )
        except ValueError:
            continue

        if data.size == 0:
            continue

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] <= COLUMN_Y:
            continue

        x_values = data[:, COLUMN_X]
        y_values = data[:, COLUMN_Y]
        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
        return x_values[valid_mask], y_values[valid_mask], time_values[0], time_values[-1]

    raise ValueError(f"Unable to parse columns 3 and 4 from {csv_file}")


def format_duration(total_seconds):
    total_seconds = int(round(total_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def expand_group_files(patterns):
    expanded_files = []
    seen_files = set()

    for pattern in patterns:
        for path in sorted(glob(str(pattern))):
            resolved_path = Path(path).expanduser()
            if not resolved_path.exists():
                continue
            if resolved_path in seen_files:
                continue
            seen_files.add(resolved_path)
            expanded_files.append(resolved_path)

    return expanded_files


#%% Data

group_results = []

for group in GROUPS:
    group_name = group["name"]
    csv_files = expand_group_files(group["patterns"])

    if not csv_files:
        print(f"ALERT: group '{group_name}' has no matching CSV files.")
        continue

    all_x = []
    all_y = []
    total_measurement_seconds = 0.0

    for csv_file in csv_files:
        try:
            x_values, y_values, start_time, end_time = load_csv_columns(csv_file)
        except ValueError as exc:
            print(f"ALERT: skipping {csv_file}: {exc}")
            continue

        if x_values.size == 0:
            print(f"ALERT: skipping {csv_file}: no numeric rows found.")
            continue

        all_x.append(x_values)
        all_y.append(y_values)
        total_measurement_seconds += max(0.0, (end_time - start_time).total_seconds())

    if not all_x:
        print(f"ALERT: group '{group_name}' has no usable CSV data.")
        continue

    group_x = np.concatenate(all_x)
    group_y = np.concatenate(all_y)
    countable_mask = group_x >= MIN_COUNT_X
    line_values = LINE_SLOPE * group_x
    below_line_mask = (group_y < line_values) & countable_mask
    total_counted_particles = int(np.sum(countable_mask))

    group_results.append(
        {
            "name": group_name,
            "files": csv_files,
            "x": group_x,
            "y": group_y,
            "total_particles": total_counted_particles,
            "particles_below_line": int(np.sum(below_line_mask)),
            "below_line_pct": (
                100.0 * np.sum(below_line_mask) / total_counted_particles
                if total_counted_particles
                else 0.0
            ),
            "measurement_seconds": total_measurement_seconds,
            "cps": (
                total_counted_particles / total_measurement_seconds
                if total_measurement_seconds > 0
                else 0.0
            ),
        }
    )

if not group_results:
    print("ALERT: no usable CSV groups found.")
    raise SystemExit(0)


#%% Graf

n_groups = len(group_results)
ncols = min(2, n_groups)
nrows = ceil(n_groups / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)
axes_flat = axes.ravel()
cmap = plt.cm.jet.copy()
cmap.set_bad("white")

for ax, result in zip(axes_flat, group_results):
    x_values = result["x"]
    y_values = result["y"]
    hist = ax.hist2d(x_values, y_values, bins=BINS, range=HIST_RANGE, cmap=cmap, cmin=1)
    x_line = np.linspace(HIST_RANGE[0][0], HIST_RANGE[0][1], 500)
    y_line = LINE_SLOPE * x_line
    ax.plot(x_line, y_line, color="red", linewidth=2)
    ax.axvline(MIN_COUNT_X, color="orange", linestyle="--", linewidth=2)

    ax.set_title(result["name"])
    ax.set_xlabel("3rd column")
    ax.set_ylabel("4th column")
    ax.grid(alpha=0.15)

    ax.set_xlim(HIST_RANGE[0])
    ax.set_ylim(HIST_RANGE[1])

    legend_handles = [
        Line2D([0], [0], color="red", linewidth=2, label=f"Limit line (slope={LINE_SLOPE:.3g})"),
        Line2D([0], [0], color="orange", linestyle="--", linewidth=2, label=f"Count from x >= {MIN_COUNT_X}"),
        Line2D(
            [0],
            [0],
            color="none",
            label=f"Total particles: {result['total_particles']} ({result['cps']:.2f} CPS)",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            label=f"Measurement time: {format_duration(result['measurement_seconds'])}",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            label=(
                f"Below line: {result['particles_below_line']} "
                f"({result['below_line_pct']:.1f} %)"
            ),
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)

    colorbar = fig.colorbar(hist[3], ax=ax)
    colorbar.set_label("Count")

for ax in axes_flat[n_groups:]:
    ax.remove()

fig.suptitle("CSV groups: 3rd vs 4th column histograms", fontsize=14)
fig.tight_layout()

if SAVE_PLOT:
    fig.savefig(OUTPUT_FILE, dpi=150)
    print(f"Graph saved to: {OUTPUT_FILE}")

plt.show()

# %%
