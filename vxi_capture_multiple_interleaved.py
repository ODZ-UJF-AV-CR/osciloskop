#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capture waveform frames from one or more Rigol oscilloscopes.
Channels from one scope are stored as groups inside one HDF5 file.

Varianta s prokládaným stahováním: pro každý frame se stáhnou všechny kanály,
než se přepne na další frame. To by mělo být rychlejší, protože osciloskop
nemusí znovu načítat stejný frame z paměti.

Examples:
    python3 vxi_capture_multiple_interleaved.py --scope osc1=192.168.1.224 --measurement-name test_01 --run-once
    python3 vxi_capture_multiple_interleaved.py --scope osc1=192.168.1.224 --scope osc2=192.168.1.182 \
        --outdir ~/captures/test --samples 14000 --max-measurement-time 300 \
        --measurement-name americium_run --channel CHAN1 --channel CHAN2
"""

import argparse
import os
import os
import time
from datetime import datetime, UTC, timezone, timedelta
import h5py
import numpy as np
import vxi11
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm


DEFAULT_OSCILLOSCOPES = {
    "osc1": "192.168.1.224",
#    "osc2": "192.168.1.182",
    #"oscSi":  "10.9.9.102",
}

# OSCILLOSCOPES = {
#     "oscLi6": "10.9.9.100",
#     "oscB10": "10.9.9.102",
#     "oscSi":  "10.9.9.101",
# }

DEFAULT_OUTDIR = "~/log_spacedos01B/PIND02_Si_Americium_2"
DEFAULT_HIGH_RES = True
DEFAULT_SAMPLES = 14000  # musi byt nastaveno stejne cislo v osciloskopech
DEFAULT_MAX_MEASUREMENT_TIME = 5 * 60
DEFAULT_CHANNELS = ["CHAN1", "CHAN2"]

def now_utc_str():
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

def sanitize_measurement_name(name):
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return safe.strip("_")

def parse_scope(scope_arg):
    if "=" not in scope_arg:
        raise argparse.ArgumentTypeError(
            f"Neplatny format osciloskopu '{scope_arg}', ocekavam NAME=IP."
        )
    name, ip = scope_arg.split("=", 1)
    name = name.strip()
    ip = ip.strip()
    if not name or not ip:
        raise argparse.ArgumentTypeError(
            f"Neplatny format osciloskopu '{scope_arg}', ocekavam NAME=IP."
        )
    return name, ip

def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture waveform frames from one or more Rigol oscilloscopes."
    )
    parser.add_argument(
        "--scope",
        action="append",
        type=parse_scope,
        help="Osciloskop ve formatu NAME=IP. Lze zadat vicekrat.",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help=f"Vystupni adresar pro HDF5 soubory. Default: {DEFAULT_OUTDIR}",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Pocet vzorku. Default: {DEFAULT_SAMPLES}",
    )
    parser.add_argument(
        "--max-measurement-time",
        type=float,
        default=DEFAULT_MAX_MEASUREMENT_TIME,
        help=f"Maximalni delka mereni v sekundach. Default: {DEFAULT_MAX_MEASUREMENT_TIME}",
    )
    parser.add_argument(
        "--channel",
        dest="channels",
        action="append",
        help="Kanal ke stazeni, napr. CHAN1. Lze zadat vicekrat. Default: CHAN1, CHAN2",
    )
    parser.add_argument(
        "--measurement-name",
        default="",
        help="Jmeno mereni pouzite v nazvech vystupnich souboru.",
    )
    parser.add_argument(
        "--high-res",
        dest="high_res",
        action="store_true",
        default=DEFAULT_HIGH_RES,
        help="Pouzit vysokorozlisene cteni waveformu.",
    )
    parser.add_argument(
        "--normal-res",
        dest="high_res",
        action="store_false",
        help="Pouzit normalni rozliseni waveformu.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Provest jen jedno mereni a pak skript ukoncit.",
    )
    parser.add_argument(
        "--wait-for-start",
        action="store_true",
        help="Nespoustet :RUN samo, ale cekat az osciloskop zacne bezet externim trigerem.",
    )
    args = parser.parse_args()
    args.scopes = dict(args.scope) if args.scope else DEFAULT_OSCILLOSCOPES.copy()
    args.channels = args.channels or DEFAULT_CHANNELS.copy()
    args.outdir = os.path.expanduser(args.outdir)
    args.measurement_name = sanitize_measurement_name(args.measurement_name)
    return args

class RigolScope:
    def __init__(self, name, ip):
        self.name = name
        self.ip = ip
        drv_string = "TCPIP::{}::INSTR".format(ip)
        print("Connecting to OSC, ", drv_string)
        self.drv = vxi11.Instrument(drv_string)
        
        # Optimalizace timeoutů pro rychlejší komunikaci
        self.drv.timeout = 5000  # 5s timeout (místo defaultních 25s)
        
        # Robustní připojení s retry mechanismem
        connected = False
        last_err = None
        for i in range(3):
            try:
                self.getName()
                connected = True
                break
            except Exception as e:
                last_err = e
                print(f"Pokus {i+1} o připojení k {ip} selhal: {e}. Zkouším znovu...")
                time.sleep(1)
        
        if not connected:
            print(f"Kritická chyba: Nepodařilo se připojit k osciloskopu {ip} ani po opakovaných pokusech.")
            if last_err:
                raise last_err

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

def wait_for_running(scopes):
    """Blokuje dokud alespon jeden osciloskop neni ve stavu RUN/WAIT/TD (ne STOP)."""
    print("Cekam na spusteni osciloskopů (stav != STOP)...")
    while True:
        statuses = {name: sc.get_trigger_status() for name, sc in scopes.items()}
        status_str = " | ".join(f"{n}: {s}" for n, s in statuses.items())
        print(f"  {status_str}", end="\r", flush=True)
        if any(s.upper() != "STOP" for s in statuses.values()):
            print()
            print("Osciloskop spusten, zacinam mereni...")
            return
        time.sleep(0.5)

def poll_trigger_and_frames(scopes, max_integration_time=None):
    print("Polling stav triggeru a počtu frames...")
    start_timestamp = time.time()

    with tqdm(total=max_integration_time, desc="Mereni", unit="s", leave=False) as status_bar:
        while True:
            stop_detected = False
            statuses = []

            elapsed_time = time.time() - start_timestamp
            if max_integration_time is not None and elapsed_time >= max_integration_time:
                status_bar.update(max(0, max_integration_time - status_bar.n))
                status_bar.set_postfix_str("timeout")
                tqdm.write(f"Dosažen maximální čas měření ({max_integration_time} sekund), ukončuji...")
                return "timeout"

            for name, sc in scopes.items():
                trig = sc.get_trigger_status()
                statuses.append(f"{name}: {trig}")
                if trig.upper() == "STOP":
                    stop_detected = True

            elapsed_time = time.time() - start_timestamp
            status_bar.update(max(0, elapsed_time - status_bar.n))
            status_text = " | ".join(statuses)
            if max_integration_time is not None:
                remaining_time = max(0.0, max_integration_time - elapsed_time)
                status_bar.set_postfix_str(f"{status_text} | zbyva: {remaining_time:.1f}s")
            else:
                status_bar.set_postfix_str(status_text)

            if stop_detected:
                tqdm.write("Akvizice je zastavena na některém osciloskopu.")
                return "stopped"

            time.sleep(0.5)

def download_all_frames(
    sc,
    start_time,
    end_time,
    outdir,
    samples,
    high_res,
    measurement_name,
    tag="main",
    pbar=None,
    channels=None,
    stop_event=None,
):
    channels = channels or DEFAULT_CHANNELS
    run_time = (end_time - start_time).total_seconds()
    filename = start_time.strftime("%Y%m%d_%H%M%S")
    if measurement_name:
        filename = f"{filename}_{measurement_name}"
    start_wfd = 0.01
    waveform_mode = "MAX" if high_res else "NORM"
    waveform_points = samples if high_res else 7000
    os.makedirs(outdir, exist_ok=True)
    h5name = f"{outdir}/{filename}_{sc.name}.h5"
    saved_channels = []

    with h5py.File(h5name, "w") as hf:
        hf.attrs["CAPTURING"] = run_time
        hf.attrs["START_TIME"] = start_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        hf.attrs["START_TIMESTAMP"] = start_time.timestamp()
        hf.attrs["END_TIME"] = end_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        hf.attrs["END_TIMESTAMP"] = end_time.timestamp()
        hf.attrs["SCOPE_NAME"] = sc.name
        hf.attrs["IP"] = sc.ip
        hf.attrs["MEASUREMENT_NAME"] = measurement_name

        # --- Fáze 1: Zjistit aktivní kanály, přečíst parametry, vytvořit HDF5 skupiny ---
        active_channels = []
        channel_info = {}  # channel -> {group, xinc, yinc, ...}

        for channel in channels:
            disp = sc.ask(f":{channel}:DISP?").strip()
            if disp == "0":
                tqdm.write(f"{sc.name}: {channel} is not enabled")
                continue

            tqdm.write(f"{sc.name}: Configuring {channel}")
            active_channels.append(channel)
            saved_channels.append(channel)

            # Configure waveform source and format
            sc.write(f":WAV:SOUR {channel}")
            sc.write(":WAV:FORM BYTE")
            sc.write(f":WAV:MODE {waveform_mode}")
            sc.write(f":WAV:POIN {waveform_points}")

            # Read measurement parameters
            sc.write(":WAV:XINC?")
            xinc = float(sc.read(100))
            sc.write(":WAV:YINC?")
            yinc = float(sc.read(100))
            sc.write(":TRIGger:EDGe:LEVel?")
            trig_level = float(sc.read(100))
            trig_channel = sc.ask(":TRIGger:EDGe:SOUR?").strip()
            sc.write(":WAVeform:YORigin?")
            yorig = float(sc.read(100))
            sc.write(":WAVeform:XORigin?")
            xorig = float(sc.read(100))
            sc.write(":FUNC:WREP:FEND?")
            frames = int(sc.read(100))

            channel_group = hf.create_group(channel)
            channel_group.attrs["FRAMES"] = frames
            channel_group.attrs["XINC"] = xinc
            channel_group.attrs["YINC"] = yinc
            channel_group.attrs["YORIGIN"] = yorig
            channel_group.attrs["XORIGIN"] = xorig
            channel_group.attrs["TRIG_LEVEL"] = trig_level
            channel_group.attrs["TRIG_CHANNEL"] = trig_channel
            channel_group.attrs["CHANNEL"] = channel

            preamble = sc.ask(":WAV:PRE?").strip()

            channel_info[channel] = {
                "group": channel_group,
                "xinc": xinc,
                "yinc": yinc,
                "yorig": yorig,
                "xorig": xorig,
                "trig_level": trig_level,
                "trig_channel": trig_channel,
                "frames": frames,
                "preamble": preamble,
                "lastwave": bytearray(),
                "wfd": start_wfd,
            }

        if not active_channels:
            tqdm.write(f"{sc.name}: Žádný aktivní kanál, přeskakuji.")
            return

        # Počet framů — vezmeme minimum přes kanály (měly by být stejné)
        frames = min(ci["frames"] for ci in channel_info.values())

        sc.write(":FUNC:WREP:FCUR 1")
        time.sleep(0.2)

        # --- Fáze 2: Prokládané stahování: frame 1 ch1+ch2, frame 2 ch1+ch2, ... ---
        for n in tqdm(range(1, frames+1), desc=f"{sc.name}-frames", leave=False, disable=(pbar is not None)):
            if stop_event is not None and stop_event.is_set():
                tqdm.write(f"{sc.name}: Přerušeno, dokončuji zápis souboru...")
                break

            sc.write(f":FUNC:WREP:FCUR {n}")

            # Čekání na přepnutí framu
            frame_switch_timeout = 50
            frame_switch_count = 0
            while True:
                time.sleep(0.02)
                fcur = sc.ask(":FUNC:WREP:FCUR?").strip()
                if str(n) == fcur:
                    break
                frame_switch_count += 1
                if frame_switch_count > frame_switch_timeout:
                    tqdm.write(f"Timeout při přepínání na frame {n}, přeskakuji")
                    break

            # Přečíst CTAG jednou pro tento frame
            try:
                ctag = float(eval(sc.ask(":FUNCtion:WREPlay:CTAG?")))
            except:
                ctag = 0.0

            # Stáhnout data pro každý kanál z tohoto framu
            for channel in active_channels:
                ci = channel_info[channel]
                wfd = ci["wfd"]

                # Přepnout zdroj na aktuální kanál
                sc.write(f":WAV:SOUR {channel}")
                sc.write(":WAV:FORM BYTE")
                sc.write(f":WAV:MODE {waveform_mode}")
                sc.write(f":WAV:POIN {waveform_points}")

                reread_count = 0

                while True:
                    time.sleep(wfd)
                    sc.write(":WAV:DATA?")

                    full_data = bytearray(sc.drv.read_raw(waveform_points))

                    if full_data.startswith(b'#'):
                        header_len = 2 + int(full_data[1:2])
                        wave = full_data[header_len:-1]
                    else:
                        wave = full_data[11:-1]

                    if np.array_equal(wave, ci["lastwave"]):
                        wfd += 0.003
                        reread_count += 1
                        if reread_count > 10:
                            tqdm.write(
                                f"Frame {n} {channel}: Opakované čtení identických dat, přeskakuji"
                            )
                            dset = ci["group"].create_dataset(str(n), data=wave)
                            dset.attrs["frame_index"] = n
                            dset.attrs["channel"] = channel
                            dset.attrs["scope_name"] = sc.name
                            dset.attrs["CTAG"] = ctag
                            dset.attrs["TRG_TIME"] = (start_time + timedelta(seconds=ctag)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                            dset.attrs["TRG_TIMESTAMP"] = (start_time + timedelta(seconds=ctag)).timestamp()
                            dset.attrs["preamble"] = ci["preamble"]
                            dset.attrs["error"] = "repeated_data_timeout"
                            if pbar:
                                pbar.update(1)
                            break
                    else:
                        dset = ci["group"].create_dataset(str(n), data=wave)
                        dset.attrs["frame_index"] = n
                        dset.attrs["channel"] = channel
                        dset.attrs["scope_name"] = sc.name
                        dset.attrs["CTAG"] = ctag
                        dset.attrs["TRG_TIME"] = (start_time + timedelta(seconds=ctag)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                        dset.attrs["TRG_TIMESTAMP"] = (start_time + timedelta(seconds=ctag)).timestamp()
                        dset.attrs["preamble"] = ci["preamble"]
                        ci["lastwave"] = wave
                        if pbar:
                            pbar.update(1)
                        ci["wfd"] = start_wfd
                        reread_count = 0
                        break

    tqdm.write(f"Saved channels {', '.join(saved_channels)} to {h5name}")

if __name__ == "__main__":
    args = parse_args()

    print("Inicializuji připojení...")
    scopes = {name: RigolScope(name, ip) for name, ip in args.scopes.items()}
    stop_event = threading.Event()

    print(scopes)
    print("Konfiguruji osciloskopy...")
    for sc in scopes.values():
        sc.write(f":ACQuire:MDEPth {args.samples}")
        sc.write(":TRIGger:SWEep NORMal")


        if args.high_res:
            sc.write(":WAV:MODE MAX") # NORMal, MAXimum, RAW
            sc.write(f":WAV:POIN {args.samples}")
        else:
            sc.write(":WAV:MODE NORM") # NORMal, MAXimum, RAW
            sc.write(":WAV:POIN 7000")



    try:
      while True:
        try:

            # sc.write(":FUNCtion:WREPlay:TTAG 1")
            print("Spouštím měření na všech osciloskopech...")
            if args.measurement_name:
                print(f"Jmeno mereni: {args.measurement_name}")
            if args.wait_for_start:
                print("Rezim --wait-for-start: nastavuji record mode, ale nespoustim :RUN.")
                for sc in scopes.values():
                    sc.stop()
                    sc.set_rec_mode("RECORD")
                time.sleep(0.5)
                wait_for_running(scopes)
            else:
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
                result = poll_trigger_and_frames(scopes, args.max_measurement_time)
            except KeyboardInterrupt:
                print("Přerušeno uživatelem, dokončuji současné měření a pokračuji...")
                result = "interrupted"

            
            end_time = datetime.now(timezone.utc)

            print("Zastavuji měření na všech osciloskopech...")
            for sc in scopes.values():
                sc.stop()
            time.sleep(0.5)
            
            print(f"Měření dokončeno ({result}), pokračuji stažením dat...")

            print("Stahuji výsledné frames (prokládaně po framech)...")

            # Zjisti celkový počet snímků pro progress bar
            total_frames = 0
            for sc in scopes.values():
                for ch in args.channels:
                    disp = sc.ask(f":{ch}:DISP?").strip()
                    if disp == "0":
                        continue
                    sc.write(":FUNC:WREP:FEND?")
                    frames = int(sc.read(100))
                    total_frames += frames

            stop_event.clear()
            with tqdm(total=total_frames, desc="Celkem snímků") as pbar:
                with ThreadPoolExecutor(max_workers=len(scopes)) as executor:
                    futures = [
                        executor.submit(
                            download_all_frames,
                            sc,
                            start_time,
                            end_time,
                            args.outdir,
                            args.samples,
                            args.high_res,
                            args.measurement_name,
                            sc.name,
                            pbar,
                            args.channels,
                            stop_event,
                        )
                        for sc in scopes.values()
                    ]
                    try:
                        for future in as_completed(futures):
                            future.result()
                    except KeyboardInterrupt:
                        tqdm.write("\nPřerušeno uživatelem, dokončuji zápis souborů...")
                        stop_event.set()
                        for future in futures:
                            future.result()

            if stop_event.is_set():
                break

            if args.run_once:
                print("Hotovo. Parametr --run-once je aktivni, ukoncuji skript.")
                break

            print("Hotovo. Spouštím nové měření..")

        except KeyboardInterrupt:
            print("Uživatel přerušil aplikaci, ukončuji...")
            break
        except Exception as e:
            print(f"Chyba: {e}")
            time.sleep(1)
            continue
    finally:
        print("Ukončuji record režim na osciloskopech...")
        for sc in scopes.values():
            try:
                sc.stop()
            except Exception as e:
                print(f"Chyba při zastavování {sc.name}: {e}")
