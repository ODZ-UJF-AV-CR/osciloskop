#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from datetime import datetime
import h5py
import numpy as np
import vxi11

OSCILLOSCOPES = {
    "scopeA": "10.9.0.101",
    "scopeB": "10.9.0.102",
    "scopeC": "10.9.0.103",
}

CHANNEL = "CHAN1" 

def now_utc_str():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

class RigolScope:
    def __init__(self, name, ip):
        self.name = name
        self.ip = ip
        self.drv = vxi11.Instrument(ip, timeout=10)
    
    def write(self, cmd):
        self.drv.write(cmd)
    def ask(self, cmd):
        return self.drv.ask(cmd)
    def read(self, n=1024):
        return self.drv.read(n)
    def idn(self):
        return self.ask("*IDN?")
    
    def get_trigger_status(self):
        # Vrací např. 'RUN', 'STOP', 'WAIT', 'TD'
        return self.ask(":TRIG:STAT?").strip()
    def get_frame_count(self):
        return int(self.ask(":REC:COUN?").strip())
    def select_frame(self, idx):
        self.write(f":REC:FRAM {idx}")
    def stop(self):
        self.write(":STOP")
    def run(self):
        self.write(":RUN")
    def single(self):
        self.write(":SING")
    def set_channel(self, ch):
        self.write(f":WAV:SOUR {ch}")
    def set_waveform_format(self):
        self.write(":WAV:FORM BYTE")
    def set_waveform_mode(self):
        self.write(":WAV:MODE NORM")
    def set_waveform_points(self, points=14000):
        self.write(f":WAV:POIN {points}")
    def get_preamble(self):
        self.write(":WAV:PREAM?")
        return self.read(512).decode(errors="ignore")
    def get_waveform(self):
        self.write(":WAV:DATA?")
        block = self.read(14010)
        # Ořezat SCPI/TMC blok (#nXXXX...) -> čistá data
        if block.startswith(b'#'):
            numlen = int(block[1:2])
            dlen = int(block[2:2+numlen])
            data = block[2+numlen:2+numlen+dlen]
            return data
        return block

    def save_frame(self, frame_idx, preamble, data, start_time, end_time, tag="main"):
        fname = f"{self.name}_{start_time}_start_{end_time}_end_frame{frame_idx:03d}_{tag}.h5"
        with h5py.File(fname, "w") as f:
            f.create_dataset("waveform", data=np.frombuffer(data, dtype=np.uint8))
            f.attrs["scope_name"] = self.name
            f.attrs["ip"] = self.ip
            f.attrs["frame_index"] = frame_idx
            f.attrs["preamble"] = preamble
            f.attrs["start_time_utc"] = start_time
            f.attrs["end_time_utc"] = end_time

def poll_trigger_and_frames(scopes):
    print("Polling stav triggeru a počtu frames...")
    while True:
        stop_detected = False
        statuses = []
        frame_counts = []
        for name, sc in scopes.items():
            trig = sc.get_trigger_status()
            frames = sc.get_frame_count()
            statuses.append(f"{name}: {trig}")
            frame_counts.append(f"{name}: {frames} frames")
            if trig.upper() != "RUN" and trig.upper() != "WAIT":
                stop_detected = True
        print(" | ".join(statuses), "||", " | ".join(frame_counts))
        if stop_detected:
            print("Akvizice je zastavena na některém osciloskopu.")
            break
        time.sleep(1)

def download_all_frames(sc, start_time, end_time, tag="main"):
    frame_count = sc.get_frame_count()
    print(f"{sc.name}: Stahuji {frame_count} frame(s)...")
    results = []
    for i in range(1, frame_count+1):
        sc.select_frame(i)
        sc.set_channel(CHANNEL)
        sc.set_waveform_format()
        sc.set_waveform_mode()
        sc.set_waveform_points()
        data = sc.get_waveform()
        preamble = sc.get_preamble()
        sc.save_frame(i, preamble, data, start_time, end_time, tag=tag)
        results.append(i)
        print(f"{sc.name}: Frame {i} stažen a uložen.")
    return results

if __name__ == "__main__":
    print("Inicializuji připojení...")
    scopes = {name: RigolScope(name, ip) for name, ip in OSCILLOSCOPES.items()}

    # Před spuštěním akvizice stáhni případné staré snímky
    print("Kontroluji existující frames před měřením...")
    pre_time = now_utc_str()
    for sc in scopes.values():
        old_cnt = sc.get_frame_count()
        if old_cnt > 0:
            print(f"{sc.name}: Nalzeno {old_cnt} starých frame(s). Stahuji...")
            download_all_frames(sc, pre_time, pre_time, tag="preexisting")
        else:
            print(f"{sc.name}: Žádné staré frames.")

    print("Spouštím měření na všech osciloskopech...")
    for sc in scopes.values():
        sc.run()
    time.sleep(0.5)  # malá prodleva, aby se stihlo rozjet

    start_time = now_utc_str()
    print("Polling průběhu měření (CTRL+C = přerušení)...")
    try:
        poll_trigger_and_frames(scopes)
    except KeyboardInterrupt:
        print("Přerušeno uživatelem, ukončuji...")

    end_time = now_utc_str()

    print("Zastavuji měření na všech osciloskopech...")
    for sc in scopes.values():
        sc.stop()
    time.sleep(0.5)

    print("Stahuji výsledné frames...")
    for sc in scopes.values():
        download_all_frames(sc, start_time, end_time, tag="main")
    print("Hotovo.")

