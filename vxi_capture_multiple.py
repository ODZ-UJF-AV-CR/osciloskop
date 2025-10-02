#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from datetime import datetime, UTC, timezone, timedelta
import h5py
import numpy as np
import vxi11
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


OSCILLOSCOPES = {
    "osc1": "10.42.0.19",
#    "osc2": "192.168.1.182",
    #"oscSi":  "10.9.9.102",
}

# OSCILLOSCOPES = {
#     "oscLi6": "10.9.9.100",
#     "oscB10": "10.9.9.102",
#     "oscSi":  "10.9.9.101",
# }

# Filename prefix
PREFIX = ""
OUTDIR = "./data/TEST"

# Maximální doba měření v sekundách (None = neomezeno, 120 = 2 minuty)
MAX_MEASUREMENT_TIME = 120  # Nastav na None pro neomezenou dobu

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
            f.attrs["start_time_utc"] = start_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            f.attrs["end_time_utc"] = end_time

def poll_trigger_and_frames(scopes, max_integration_time=None):
    print("Polling stav triggeru a počtu frames...")
    start_timestamp = time.time()
    
    while True:
        stop_detected = False
        statuses = []
        
        # Zkontroluj, jestli už neuplynul maximální čas (pouze pokud je nastaven)
        if max_integration_time is not None:
            elapsed_time = time.time() - start_timestamp
            if elapsed_time >= max_integration_time:
                print(f"Dosažen maximální čas měření ({max_integration_time} sekund), ukončuji...")
                return "timeout"
            
        for name, sc in scopes.items():
            trig = sc.get_trigger_status()
            statuses.append(f"{name}: {trig}")
            if trig.upper() == "STOP":
                stop_detected = True
                
        # Zobraz zbývající čas (pouze pokud je limit nastaven)
        if max_integration_time is not None:
            elapsed_time = time.time() - start_timestamp
            remaining_time = max_integration_time - elapsed_time
            print(" | ".join(statuses) + f" | Zbývá: {remaining_time:.1f}s")
        else:
            print(" | ".join(statuses))
        
        if stop_detected:
            print("Akvizice je zastavena na některém osciloskopu.")
            return "stopped"
            
        time.sleep(0.5)

def download_all_frames(sc, start_time, end_time, tag="main", pbar=None, channels=["CHAN1", "CHAN2"]):
    import sys
    import os

    run_time = (end_time - start_time).total_seconds()
    filename = start_time.strftime("%Y%m%d_%H%M%S")
    start_wfd = 0.01
    wfd = start_wfd

    for channel in channels:
        disp = sc.ask(f":{channel}:DISP?").strip()
        if disp == "0":
            print(f"{sc.name}: {channel} is not enabled")
            continue
        print(f"{sc.name}: Reading out {channel}")

        # copling = sc.ask(f":{channel}:COUP?").strip()

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
        
        trig_channel = sc.ask(":TRIGger:EDGe:SOUR?").strip()

        sc.write(":WAVeform:YORigin?")
        yorig = float(sc.read(100))
        sc.write(":WAVeform:XORigin?")
        xorig = float(sc.read(100))
        sc.write(":FUNC:WREP:FEND?")
        frames = int(sc.read(100))


        lastwave = bytearray()
        os.makedirs(OUTDIR, exist_ok=True)
        h5name = f"{OUTDIR}/{filename}_{sc.name}_{channel}.h5"
        with h5py.File(h5name, "w") as hf:
            hf.attrs["FRAMES"] = frames - 1  # remove 1 frame, because first is trigger time mark frame
            hf.attrs["XINC"] = xinc
            hf.attrs["YINC"] = yinc
            hf.attrs["TRIG"] = trig
            hf.attrs["TRIG_CHANNEL"] = trig_channel
            hf.attrs["YORIGIN"] = yorig
            hf.attrs["XORIGIN"] = xorig
            hf.attrs["CAPTURING"] = run_time
            hf.attrs["START_TIME"] = start_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            hf.attrs["START_TIMESTAMP"] = start_time.timestamp()
            hf.attrs["END_TIME"] = end_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            hf.attrs["END_TIMESTAMP"] = end_time.timestamp()
            hf.attrs["SCOPE_NAME"] = sc.name
            hf.attrs["IP"] = sc.ip
            hf.attrs["CHANNEL"] = channel
            
            sc.write(":FUNC:WREP:FCUR 2")
            time.sleep(0.5)

            # Zaciname od snimku 2, protoze 1 je jiz nacteny triggerem pro zaznamenani casu nula. 
            for n in tqdm(range(2, frames), desc=f"{sc.name}-{channel}", leave=False, disable=(pbar is not None)):
                sc.write(f":FUNC:WREP:FCUR {n}")
                while True:
                    time.sleep(0.05)
                    fcur = sc.ask(":FUNC:WREP:FCUR?").strip()
                    if str(n) == fcur:
                        break

                reread_count = 0
                ctag = float(eval(sc.ask(":FUNCtion:WREPlay:CTAG?")))
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
                        print(len(wave), "bytes read")
                        dset = hf.create_dataset(str(n), data=wave)
                        dset.attrs["frame_index"] = n
                        dset.attrs["channel"] = channel
                        dset.attrs["scope_name"] = sc.name
                        dset.attrs["CTAG"] = ctag
                        dset.attrs["TRG_TIME"] = (start_time + timedelta(seconds=ctag)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                        dset.attrs["TRG_TIMESTAMP"] = (start_time + timedelta(seconds=ctag)).timestamp()
                        dset.attrs["preamble"] = sc.ask(":WAV:PRE?").strip()
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

            # sc.write(":FUNCtion:WREPlay:TTAG 1")
            print("Spouštím měření na všech osciloskopech...")
            for sc in scopes.values():
                sc.stop()
                sc.set_rec_mode("RECORD")
            time.sleep(0.5)
            for sc in scopes.values():
                sc.run()
                sc.write(":TFORce")


            start_time = datetime.now(timezone.utc)
            time.sleep(0.5)

            print("Polling průběhu měření (CTRL+C = přerušení)...")
            try:
                result = poll_trigger_and_frames(scopes, MAX_MEASUREMENT_TIME)
            except KeyboardInterrupt:
                print("Přerušeno uživatelem, dokončuji současné měření a pokračuji...")
                result = "interrupted"

            
            end_time = datetime.now(timezone.utc)

            print("Zastavuji měření na všech osciloskopech...")
            for sc in scopes.values():
                sc.stop()
            time.sleep(0.5)
            
            print(f"Měření dokončeno ({result}), pokračuji stažením dat...")

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

            print("Hotovo. Spouštím nové měření..")

        except KeyboardInterrupt:
            print("Uživatel přerušil aplikaci, ukončuji...")
            break
        except Exception as e:
            print(f"Chyba: {e}")
            time.sleep(1)
            continue
