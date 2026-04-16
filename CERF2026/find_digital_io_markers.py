#!/usr/bin/env python3
"""
Skript na detekci digitálních IO značek v H5 souboru
Pro každou waveformu určí první a druhý interval, kdy je signál v HIGH
vypisuje jejich začátek a délku s identifikátorem waveformy
"""

# %%

import h5py
import numpy as np
from pathlib import Path
import csv
from collections import Counter

# %% Nastavení
INPUT_FILE = "/home/roman/TEST/20260331_080309_DAY5_RUN1_EM_D1.h5"
THRESHOLD_V = 1.5  # V; pokud je None, použije se střed mezi minimem a maximem waveformy
MIN_GAP_US = 2.0
MAX_GAP_US = 6.0

FILTER_MIN = 25
FILTER_MAX = 26

def parse_preamble(waveform_dataset):
    preamble = waveform_dataset.attrs.get('preamble')
    if preamble:
        parts = str(preamble).split(',')
        if len(parts) > 9:
            return {
                'xinc': float(parts[4]),
                'xorigin': float(parts[5]),
                'yinc': float(parts[7]),
                'yorigin': float(parts[8]),
                'yreference': float(parts[9]),
            }

    return {
        'xinc': 1e-6,
        'xorigin': 0.0,
        'yinc': 1.0,
        'yorigin': 0.0,
        'yreference': 0.0,
    }


def convert_to_volts(raw_signal, scale_info):
    return (raw_signal.astype(np.float64) - scale_info['yreference'] - scale_info['yorigin']) * scale_info['yinc']


def get_threshold(waveform_volts):
    if THRESHOLD_V is not None:
        return THRESHOLD_V
    return (float(np.min(waveform_volts)) + float(np.max(waveform_volts))) / 2.0


def find_high_intervals(waveform_volts, scale_info, waveform_id):
    threshold = get_threshold(waveform_volts)
    digital = (waveform_volts > threshold).astype(int)

    # Hrany mezi LOW a HIGH převedeme na intervaly HIGH.
    padded = np.concatenate(([0], digital, [0]))
    diff = np.diff(padded)
    rising = np.where(diff == 1)[0]
    falling = np.where(diff == -1)[0]

    high_intervals = []
    for start_idx, end_idx in zip(rising, falling):
        high_intervals.append({
            'start_idx': int(start_idx),
            'end_idx': int(end_idx),
            'start_us': (scale_info['xorigin'] + start_idx * scale_info['xinc']) * 1e6,
            'duration_us': (end_idx - start_idx) * scale_info['xinc'] * 1e6,
        })

    for index in range(len(high_intervals) - 1):
        high_1 = high_intervals[index]
        high_2 = high_intervals[index + 1]
        gap_us = (high_2['start_idx'] - high_1['end_idx']) * scale_info['xinc'] * 1e6

        if MIN_GAP_US <= gap_us <= MAX_GAP_US:
            return {
                'waveform_id': str(waveform_id),
                'threshold': threshold,
                'gap_us': gap_us,
                'high_1_start_us': high_1['start_us'],
                'high_1_duration_us': high_1['duration_us'],
                'high_2_start_us': high_2['start_us'],
                'high_2_duration_us': high_2['duration_us'],
                'high_count': len(high_intervals),
                'pair_index': index,
            }

    return None


# %% Načtení dat
results = []
waveforms_found = 0
waveforms_processed = 0
waveforms_with_valid_pair = 0
with h5py.File(INPUT_FILE, 'r') as f:
    print("Klíče v souboru:", list(f.keys()))
    channel = f['CHAN1']
    waveform_keys = sorted(channel.keys(), key=int)
    waveforms_found = len(waveform_keys)

    for waveform_key in waveform_keys:
        current_dataset = channel[waveform_key]
        current_scale_info = parse_preamble(current_dataset)
        current_signal = current_dataset[:]
        current_volts = convert_to_volts(current_signal, current_scale_info)
        current_results = find_high_intervals(current_volts, current_scale_info, waveform_key)
        waveforms_processed += 1
        if current_results:
            waveforms_with_valid_pair += 1
            results.append(current_results)

# %% Výstup
print(f"Soubor: {Path(INPUT_FILE).name}")
print(f"Waveforms nalezeno: {waveforms_found}")
print(f"Waveforms zpracováno: {waveforms_processed}")
print(f"Waveforms s mezerou {MIN_GAP_US:.0f}-{MAX_GAP_US:.0f} us: {waveforms_with_valid_pair}")
print(f"Vypsáno {len(results)} waveformů\n")

high_1_duration_counts = Counter(round(r['high_1_duration_us'], 3) for r in results)
high_2_duration_counts = Counter(round(r['high_2_duration_us'], 3) for r in results)
total_results = len(results) if results else 1

print("Délky prvního pulzu:")
print("délka_us\tpočet\tprocent")
for duration_us, count in sorted(high_1_duration_counts.items()):
    percentage = count / total_results * 100.0
    print(f"{duration_us:.3f}\t{count}\t{percentage:.2f} %")

print("\nDélky druhého pulzu:")
print("délka_us\tpočet\tprocent")
for duration_us, count in sorted(high_2_duration_counts.items()):
    percentage = count / total_results * 100.0
    print(f"{duration_us:.3f}\t{count}\t{percentage:.2f} %")

print()
print(f"{'Waveform':<10} {'HIGH1 start (us)':<18} {'HIGH1 len (us)':<18} {'Gap (us)':<12} {'HIGH2 start (us)':<18} {'HIGH2 len (us)':<18}")
print("-" * 104)

filtered_results = [r for r in results if not (FILTER_MIN <= r['high_1_duration_us'] <= FILTER_MAX)]

for r in filtered_results:
    print(f"{r['waveform_id']:<10} {r['high_1_start_us']:<18.3f} {r['high_1_duration_us']:<18.3f} {r['gap_us']:<12.3f} {r['high_2_start_us']:<18.3f} {r['high_2_duration_us']:<18.3f}")

# %% Uložení do CSV
if results:
    output_file = INPUT_FILE.replace('.h5', '_io_markers.csv')
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['waveform_id', 'threshold', 'pair_index', 'gap_us', 'high_1_start_us', 'high_1_duration_us', 'high_2_start_us', 'high_2_duration_us', 'high_count'])
        writer.writeheader()
        for r in filtered_results:
            writer.writerow({
                key: r[key]
                for key in ['waveform_id', 'threshold', 'pair_index', 'gap_us', 'high_1_start_us', 'high_1_duration_us', 'high_2_start_us', 'high_2_duration_us', 'high_count']
            })
    print(f"\nVýstup uložen: {output_file}")

# %%
