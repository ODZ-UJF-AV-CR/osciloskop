#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from datetime import datetime, UTC
import h5py
import numpy as np
import vxi11

OSCILLOSCOPES = {
    "oscSi": "10.9.9.101",
    "oscB": "10.9.9.100",
#    "scopeC": "10.9.0.103",
}

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
        return int(self.ask(":FUNC:WREP:FEND?").strip())
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
            #if trig.upper() != "RUN" and trig.upper() != "WAIT" and trig.upper() != "TB":
            if trig.upper() == "STOP":
                stop_detected = True
        print(" | ".join(statuses)) # "||", " | ".join(frame_counts))
        if stop_detected:
            print("Akvizice je zastavena na některém osciloskopu.")
            break
        time.sleep(1)
        print(statuses)
        print(frame_counts)

def download_all_frames(sc, start_time, end_time, tag="main"):
    import sys
    import os

    channels = ["CHAN1", "CHAN2"]
    run_time = end_time  # nebo spočítejte rozdíl, pokud chcete délku
    filename = start_time  # nebo použijte jiný identifikátor
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
        print("XINC:", xinc)
        sc.write(":WAV:YINC?")
        yinc = float(sc.read(100))
        print("YINC:", yinc)
        sc.write(":TRIGger:EDGe:LEVel?")
        trig = float(sc.read(100))
        print("TRIG:", trig)
        sc.write(":WAVeform:YORigin?")
        yorig = float(sc.read(100))
        print("YORIGIN:", yorig)
        sc.write(":WAVeform:XORigin?")
        xorig = float(sc.read(100))
        print("XORIGIN:", xorig)
        sc.write(":FUNC:WREP:FEND?")
        frames = int(sc.read(100))
        print("FRAMES:", frames, "SUBRUN", filename)

        lastwave = bytearray()
        outdir = "./data"
        os.makedirs(outdir, exist_ok=True)
        h5name = f"{outdir}/data_{sc.name}_{filename}_{channel}.h5"
        with h5py.File(h5name, "w") as hf:
            hf.create_dataset("FRAMES", data=frames)
            hf.create_dataset("XINC", data=xinc)
            hf.create_dataset("YINC", data=yinc)
            hf.create_dataset("TRIG", data=trig)
            hf.create_dataset("YORIGIN", data=yorig)
            hf.create_dataset("XORIGIN", data=xorig)
            hf.create_dataset("CAPTURING", data=run_time)
            sc.write(":FUNC:WREP:FCUR 1")
            time.sleep(0.5)
            for n in range(1, frames + 1):
                sc.write(f":FUNC:WREP:FCUR {n}")
                while True:
                    time.sleep(0.05)
                    fcur = sc.ask(":FUNC:WREP:FCUR?").strip()
                    if str(n) == fcur:
                        sys.stdout.write(str(n))
                        sys.stdout.flush()
                        break
                    else:
                        print(f"Needwait: {n} vs {fcur}")

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
                        print(f" Same waveform, wait {wfd} and reread")
                        reread_count += 1
                        if reread_count > 5:
                            print("------------ Wrong trigger level?")
                    else:
                        hf.create_dataset(str(n), data=wave)
                        lastwave = wave
                        sys.stdout.write(".")
                        sys.stdout.flush()
                        wfd = start_wfd
                        break
        print("DOWNLOADING DONE")
        print(f"Saved {frames} frames to {h5name}")

if __name__ == "__main__":
    print("Inicializuji připojení...")
    scopes = {name: RigolScope(name, ip) for name, ip in OSCILLOSCOPES.items()}

    print(scopes)

    
    while True:
        try:
       
            # Před spuštěním akvizice stáhni případné staré snímky
            #print("Kontroluji existující frames před měřením...")
            #pre_time = now_utc_str()
            #for sc in scopes.values():
            #     old_cnt = sc.get_frame_count()
            #     if old_cnt > 0:
            #         print(f"{sc.name}: Nalzeno {old_cnt} starých frame(s). Stahuji...")
            #         download_all_frames(sc, pre_time, pre_time, tag="preexisting")
            #     else:
            #         print(f"{sc.name}: Žádné staré frames.")

            print("Spouštím měření na všech osciloskopech...")
            for sc in scopes.values():
                sc.stop()
                sc.set_rec_mode("RECORD")
            time.sleep(0.5)
            for sc in scopes.values():
                sc.run()
            time.sleep(2)

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
                print(f"Stahuji frames z {sc.name}...")
                download_all_frames(sc, start_time, end_time, tag=sc.name)
            print("Hotovo.")


        except KeyboardInterrupt:
            print("Uživatel přerušil přípravu, ukončuji...")
            exit(0)
        except Exception as e:
            print(f"Chyba: {e}")
            time.sleep(1)
            continue