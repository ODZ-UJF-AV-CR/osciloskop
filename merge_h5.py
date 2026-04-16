# %%%%%%

import h5py
import numpy as np
import glob
import os
import datetime
import sys
from scipy.interpolate import interp1d





# %%%%%%

def list_h5_objects(directory):

    good = []
    clipping = []
    noise = [] 

    h5_files = glob.glob(directory)
    for h5_file in h5_files:
        print(f"File: {h5_file}")
        try:
            with h5py.File(h5_file, 'r') as f:
                YINC = float(np.array(f.get("YINC", 0.0)))
                #YORIGIN = float(np.array(f.get("YORIGIN", 0.0)))
                YORIGIN =0
                TRIG = float(np.array(f.get("TRIG", 0.0)))

                print(f"TRIG: {TRIG*1000.0} mv")
                print(f"  YINC: {YINC}, YORIGIN: {YORIGIN}")

                numeric_keys = [int(key) for key in f.keys() if key.isdigit()]
                print(f"  Numeric keys: {numeric_keys}")


                for key in numeric_keys:
                    data = np.array(f.get(str(key)), dtype=np.float32)

                    #print(max(data))

                    if max(data) > 250:
                        data *= YINC
                        data += YORIGIN
                        clipping.append(data)
                        continue

                    data *= YINC
                    data += YORIGIN

                    if max(data[1200:]) > 0.02:
                        clipping.append(data)
                        continue

                    if min(data[200:600]) < 0:
                        noise.append(data)
                        continue

                    if max(data[0:50]) > 0.05:
                        noise.append(data)
                        continue

                    #print(data)
                    good.append(data)

        except Exception as e:
            print(f"  Error reading file: {e}")

    
    return good, clipping, noise


bor_good, bor_clipping, bor_noise = list_h5_objects("/home/roman/mnt/kapybara/storage/experiments/2025/05_CERF/NeutronExperiment/CERF_2025_05_27_RUN4*/data_oscB*.h5")
li_good, li_clipping, li_noise = list_h5_objects("/home/roman/mnt/kapybara/storage/experiments/2025/05_CERF/NeutronExperiment/CERF_2025_05_27_RUN4*/data_oscL*.h5")
si_good, si_clipping, si_noise = list_h5_objects("/home/roman/mnt/kapybara/storage/experiments/2025/05_CERF/NeutronExperiment/CERF_2025_05_27_RUN4*/data_oscS*.h5")


# %%%%%%


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))


for i, data in enumerate(li_good):
    plt.plot(data, label=f"Data {i+1}", linewidth=0.05, alpha=0.1, color='green')

for i, data in enumerate(bor_good):
    plt.plot(data, label=f"Data {i+1}", linewidth=0.05, alpha=0.1, color='blue')

for i, data in enumerate(si_good):
    plt.plot(data, label=f"Data {i+1}", linewidth=0.05, alpha=0.1, color='red')

# %%

# good_maximums = [max(data) for data in good]
# good_max_indexes = [np.argmax(data) for data in good]
# good_area = [np.trapezoid(data) for data in good]
def calculate_fwhm(data):
    half_max = max(data) / 2.0
    indices = np.where(data >= half_max)[0]
    if len(indices) < 2:
        return 0  # FWHM cannot be calculated
    left_idx, right_idx = indices[0], indices[-1]
    # Interpolate for more precise FWHM calculation
    if left_idx > 0:
        interp = interp1d([data[left_idx-1], data[left_idx]], [left_idx-1, left_idx])
        left = interp(half_max)
    else:
        left = left_idx

    if right_idx < len(data) - 1:
        interp = interp1d([data[right_idx], data[right_idx+1]], [right_idx, right_idx+1])
        right = interp(half_max)
    else:
        right = right_idx
    return right - left

si_max = [max(data) for data in si_good]
si_max_indexes = [np.argmax(data) for data in si_good]
si_area = [np.trapezoid(data) for data in si_good]
si_fwhm = [calculate_fwhm(data) for data in si_good]

bor_max = [max(data) for data in bor_good]
bor_max_indexes = [np.argmax(data) for data in bor_good]
bor_area = [np.trapezoid(data) for data in bor_good]
bor_fwhm = [calculate_fwhm(data) for data in bor_good]

li_max = [max(data) for data in li_good]
li_max_indexes = [np.argmax(data) for data in li_good]
li_area = [np.trapezoid(data) for data in li_good]
li_fwhm = [calculate_fwhm(data) for data in li_good]

# %%%%


plt.figure(figsize=(10, 6))
#plt.plot(good_maximums, good_area, '.', label="Maximums", color='blue', alpha=0.1)
#plt.plot(good_maximums, good_max_indexes, '.', label="Maximums", color='blue', alpha=0.1)

plt.plot(li_area, li_max, '.', label="Lithium", color='green', alpha=0.5, markersize=1)
plt.plot(bor_area, bor_max, '.', label="Boron", color='blue', alpha=0.5, markersize=1)
plt.plot(si_area, si_max, '.', label="Silicon", color='red', alpha=0.5, markersize=1)

# %%%%
plt.figure(figsize=(10, 6))
plt.plot(si_max, si_max_indexes, '.', label="Silicon max indexes", color='red', alpha=0.5, markersize=2)
plt.plot(bor_max, bor_max_indexes, '.', label="Boron max indexes", color='blue', alpha=0.5, markersize=2)
plt.plot(li_max, li_max_indexes, '.', label="Lithium max indexes", color='green', alpha=0.5, markersize=2)
plt.ylim(200, 500)
#plt.plot()


# %%%%


plt.figure(figsize=(10, 6))

# Plot FWHM vs Area for each material
plt.plot(si_max, si_fwhm, '.', label="Silicon", color='red', alpha=0.5, markersize=2)
plt.plot(bor_max, bor_fwhm, '.', label="Boron", color='blue', alpha=0.5, markersize=2)
plt.plot(li_max, li_fwhm, '.', label="Lithium", color='green', alpha=0.5, markersize=2)

plt.xlabel("Amplitude (Max)")
plt.ylabel("FWHM")
plt.legend()
plt.title("FWHM vs Area")
plt.show()


# %%%%

plt.figure(figsize=(10, 6))
threshold_min = 0.035
threshold_max = 0.045

# Filter values for Silicon
si_filtered_max = [m for m in si_max if threshold_min <= m <= threshold_max]
si_filtered_indexes = [si_max_indexes[i] for i, m in enumerate(si_max) if threshold_min <= m <= threshold_max]
si_filtered_area = [si_area[i] for i, m in enumerate(si_max) if threshold_min <= m <= threshold_max]

# Filter values for Boron
bor_filtered_max = [m for m in bor_max if threshold_min <= m <= threshold_max]
bor_filtered_indexes = [bor_max_indexes[i] for i, m in enumerate(bor_max) if threshold_min <= m <= threshold_max]
bor_filtered_area = [bor_area[i] for i, m in enumerate(bor_max) if threshold_min <= m <= threshold_max]

# Filter values for Lithium
li_filtered_max = [m for m in li_max if threshold_min <= m <= threshold_max]
li_filtered_indexes = [li_max_indexes[i] for i, m in enumerate(li_max) if threshold_min <= m <= threshold_max]
li_filtered_area = [li_area[i] for i, m in enumerate(li_max) if threshold_min <= m <= threshold_max]
# Plot the filtered data with swapped axes
plt.plot(si_filtered_indexes, si_filtered_area, '.', label="Silicon", color='red', alpha=0.5, markersize=2)
plt.plot(bor_filtered_indexes, bor_filtered_area, '.', label="Boron", color='blue', alpha=0.5, markersize=2)
plt.plot(li_filtered_indexes, li_filtered_area, '.', label="Lithium", color='green', alpha=0.5, markersize=2)

plt.ylabel("Area")
plt.xlabel("Max Indexes")
plt.legend()
plt.title("Filtered Data with Max around 0.04")
plt.show()


# %%
