#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pošle :RUN a po zadaném čase :STOP na jeden osciloskop.

Příklad:
    python3 vxi_run_stop.py 192.168.1.224
    python3 vxi_run_stop.py 192.168.1.224 --wait 30
"""

import argparse
import time
import vxi11


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spustí a po zadaném čase zastaví osciloskop."
    )
    parser.add_argument("ip", help="IP adresa osciloskopu")
    parser.add_argument(
        "--wait",
        type=float,
        default=10.0,
        help="Čekání mezi :RUN a :STOP v sekundách (default: 10)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Připojuji se k {args.ip}...")
    instr = vxi11.Instrument(f"TCPIP::{args.ip}::INSTR")
    print(instr.ask("*IDN?"))

    print("Spouštím (:RUN)...")
    instr.write(":RUN")

    print(f"Čekám {args.wait} s...")
    time.sleep(args.wait)

    print("Zastavuji (:STOP)...")
    instr.write(":STOP")

    print("Hotovo.")
