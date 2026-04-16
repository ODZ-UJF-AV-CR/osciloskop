# osciloskop
Python utility for USBTMC oscilloscope

Script is writen for JUPYTER notebook in Python 2.
It was writen and tested with RIGOL DS2102A.

## vxi_capture_multiple.py
Script `vxi_capture_multiple.py` umi spoustet opakovana mereni nebo jedno mereni pres `--run-once`.
Defaultni `--max-measurement-time` je 300 sekund, tedy 5 minut.
Kanaly jednoho osciloskopu se ukladaji do jednoho HDF5 souboru, kazdy jako vlastni skupina.

## oscilloscope_app.py
Nova desktop aplikace v `PySide6` pro obsluhu vice osciloskopu.

Umi:
- spravovat vice osciloskopu najednou (`name + IP`)
- spustit `single trigger` i `multiple triggers`
- pouzit trigger osciloskopu nebo vynutit trigger z aplikace (`:TFORce`)
- ulozit data do HDF5
- po dokonceni vygenerovat PNG nahledy pres renderer z `waveform_viewer.py`
- rovnou zobrazit posledni ziskany waveform v GUI

Zakladni spusteni:
```bash
python3 oscilloscope_app.py --scope osc1=10.11.111.36
```

Zakladni pouziti:
```bash
python3 vxi_capture_multiple.py --scope osc1=192.168.1.224 --measurement-name test_01 --run-once
```

Priklad s vice parametry:
```bash
python3 vxi_capture_multiple.py \
  --scope osc1=192.168.1.224 \
  --scope osc2=192.168.1.182 \
  --outdir ~/captures/test \
  --samples 14000 \
  --max-measurement-time 300 \
  --measurement-name americium_run \
  --channel CHAN1 \
  --channel CHAN2 \
  --run-once
```

Podporovane argumenty:
- `--scope NAME=IP` pro definici osciloskopu, lze zadat vicekrat
- `--outdir PATH` pro vystupni adresar
- `--samples N` pro pocet vzorku
- `--max-measurement-time SECONDS` pro maximalni dobu mereni
- `--measurement-name NAME` pro pojmenovani mereni a vystupnich souboru
- `--channel CHANx` pro vyber kanalu, lze zadat vicekrat
- `--high-res` nebo `--normal-res` pro rezim cteni waveformu
- `--run-once` pro ukonceni skriptu po jednom cyklu

## Instalace

### Vytvoření virtuálního prostředí
```bash
python3 -m venv venv
source venv/bin/activate
```

### Instalace závislostí
```bash
pip install -r requirements.txt
```

## Steps
### Prepare oscilloscope
Connect bias voltage source, setup oscilloscope.
Run first recording round (via the "Record" button").
508 frames shall be captured.

### Initialize and start capture 
See notes in CERN.ipynb.

### Data plotting
One mean of data processing is in the oscilloskop-hystogram.ipynb.

# Blesk

Micsig & gps tagger unit.

# Blesk-standalone
Script for stand-alone use.
