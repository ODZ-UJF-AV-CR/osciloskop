#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from datetime import datetime, UTC
import h5py
import numpy as np
import vxi11
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


OSCILLOSCOPES = {
    "oscLi6": "10.9.9.101",
    "oscB10": "10.9.9.100",
    "oscSi":  "10.9.9.102",
}

# OSCILLOSCOPES = {
#     "oscLi6": "10.9.9.100",
#     "oscB10": "10.9.9.102",
#     "oscSi":  "10.9.9.101",
# }

# Filename prefix
PREFIX = ""
OUTDIR = "./data/CERF_2025_05_27_RUN14a"


CHANNEL = "CHAN1" 

def now_utc_str():
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

class RigolScope:
    def __init__(self, name, ip):
        self.name = name
        self.ip = ip
        drv_string = "TCPIP::{}::INSTR".format(ip)
        print("Connecting to OSC, ", drv_string)
        self.drv = vxi11.Instrument(drv_string)
        self.getName()
    
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
        # NEFUNGUJE, 
        return int(self.ask(":FUNCtion:WREPlay:FMAX?").strip())
    def select_frame(self, idx):
        self.write(f":REC:FRAM {idx}")
    def set_rec_mode(self, mode="RECORD"):
        self.write(f":FUNC:WRM {mode.upper()}")
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
    def getName(self):
        resp = self.drv.ask("*IDN?")
        print(resp)
        return resp
        #self.write("*IDN?")
        #return self.read(300)

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
            #frames = sc.get_frame_count()
            statuses.append(f"{name}: {trig}")
            #frame_counts.append(f"{name}: {frames} frames")
            #print( frame_counts )
            #if trig.upper() != "RUN" and trig.upper() != "WAIT" and trig.upper() != "TB":
            if trig.upper() == "STOP":
                stop_detected = True
        print(" | ".join(statuses)) # "||", " | ".join(frame_counts))
        if stop_detected:
            print("Akvizice je zastavena na některém osciloskopu.")
            break
        time.sleep(0.5)

def download_all_frames(sc, start_time, end_time, tag="main", pbar=None):
    import sys
    import os

    channels = ["CHAN1", "CHAN2"]
    run_time = end_time
    filename = start_time
    start_wfd = 0.01
    wfd = start_wfd

    for channel in channels:
        disp = sc.ask(f":{channel}:DISP?").strip()
        if disp == "0":
            print(f"{sc.name}: {channel} is not enabled")
            continue
        print(f"{sc.name}: Reading out {channel}")

        sc.write(f":WAV:SOUR {channel}")
        sc.write(":WAV:MODE NORM")
        sc.write(":WAV:FORM BYTE")
        sc.write(":WAV:POIN 1400")

        sc.write(":WAV:XINC?")
        xinc = float(sc.read(100))
        sc.write(":WAV:YINC?")
        yinc = float(sc.read(100))
        sc.write(":TRIGger:EDGe:LEVel?")
        trig = float(sc.read(100))
        sc.write(":WAVeform:YORigin?")
        yorig = float(sc.read(100))
        sc.write(":WAVeform:XORigin?")
        xorig = float(sc.read(100))
        sc.write(":FUNC:WREP:FEND?")
        frames = int(sc.read(100))

        lastwave = bytearray()
        os.makedirs(OUTDIR, exist_ok=True)
        h5name = f"{OUTDIR}/{PREFIX}data_{sc.name}_{filename}.h5"
        with h5py.File(h5name, "w") as hf:
            hf.create_dataset("FRAMES", data=frames)
            hf.create_dataset("XINC", data=xinc)
            hf.create_dataset("YINC", data=yinc)
            hf.create_dataset("TRIG", data=trig)
            hf.create_dataset("YORIGIN", data=yorig)
            hf.create_dataset("XORIGIN", data=xorig)
            hf.create_dataset("CAPTURING", data=run_time)
            hf.create_dataset("START_TIME", data=start_time)
            hf.create_dataset("END_TIME", data=end_time)
            hf.create_dataset("SCOPE_NAME", data=sc.name)
            hf.create_dataset("IP", data=sc.ip)
            hf.create_dataset("CHANNEL", data=channel)
            sc.write(":FUNC:WREP:FCUR 1")
            time.sleep(0.5)
            for n in tqdm(range(1, frames + 1), desc=f"{sc.name}-{channel}", leave=False, disable=(pbar is not None)):
                sc.write(f":FUNC:WREP:FCUR {n}")
                while True:
                    time.sleep(0.05)
                    fcur = sc.ask(":FUNC:WREP:FCUR?").strip()
                    if str(n) == fcur:
                        break

                reread_count = 0
                while True:
                    time.sleep(wfd)
                    sc.write(":WAV:DATA?")
                    time.sleep(wfd)
                    wave1 = bytearray(sc.drv.read_raw(500))
                    wave2 = bytearray(sc.drv.read_raw(500))
                    wave3 = bytearray(sc.drv.read_raw(500))
                    wave = np.concatenate((wave1[11:], wave2, wave3[:-1]))
                    if np.array_equal(wave, lastwave):
                        wfd += 0.005
                        reread_count += 1
                        if reread_count > 5:
                            print("------------ Wrong trigger level?")
                    else:
                        hf.create_dataset(str(n), data=wave)
                        lastwave = wave
                        if pbar:
                            pbar.update(1)
                        wfd = start_wfd
                        break
        print(f"Saved {frames} frames to {h5name}")

if __name__ == "__main__":
    print("Inicializuji připojení...")
    scopes = {name: RigolScope(name, ip) for name, ip in OSCILLOSCOPES.items()}

    print(scopes)

    while True:
        try:
            print("Spouštím měření na všech osciloskopech...")
            for sc in scopes.values():
                sc.stop()
                sc.set_rec_mode("RECORD")
            time.sleep(0.5)
            for sc in scopes.values():
                sc.run()
            time.sleep(0.5)

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

            print("Stahuji výsledné frames paralelně...")

            # Zjisti celkový počet snímků pro progress bar
            total_frames = 0
            for sc in scopes.values():
                for ch in ["CHAN1", "CHAN2"]:
                    disp = sc.ask(f":{ch}:DISP?").strip()
                    if disp == "0":
                        continue
                    sc.write(":FUNC:WREP:FEND?")
                    frames = int(sc.read(100))
                    total_frames += frames

            with tqdm(total=total_frames, desc="Celkem snímků") as pbar:
                with ThreadPoolExecutor(max_workers=len(scopes)) as executor:
                    futures = [
                        executor.submit(download_all_frames, sc, start_time, end_time, sc.name, pbar)
                        for sc in scopes.values()
                    ]
                    for future in as_completed(futures):
                        future.result()

            print("Hotovo.")

        except KeyboardInterrupt:
            print("Uživatel přerušil přípravu, ukončuji...")
            exit(0)
        except Exception as e:
            print(f"Chyba: {e}")
            time.sleep(1)
            continue