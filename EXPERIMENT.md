# Logování s osciloskopem
```
  kakl@kakl-Latitude-5450:~/git/osciloskop$ python3 vxi_capture_multiple.py --scope osc=192.168.1.2 --outdir /home/kakl/DATA --samples 7000 --max-measurement-time 60 --measurement-name TEST2
```



```
usage: vxi_capture_multiple.py [-h] [--scope SCOPE] [--outdir OUTDIR] [--samples SAMPLES] [--max-measurement-time MAX_MEASUREMENT_TIME] [--channel CHANNELS]
                               [--measurement-name MEASUREMENT_NAME] [--high-res] [--normal-res] [--run-once]

Capture waveform frames from one or more Rigol oscilloscopes.

options:
  -h, --help            show this help message and exit
  --scope SCOPE         Osciloskop ve formatu NAME=IP. Lze zadat vicekrat.
  --outdir OUTDIR       Vystupni adresar pro HDF5 soubory. Default: ~/log_spacedos01B/PIND02_Si_Americium_2
  --samples SAMPLES     Pocet vzorku. Default: 14000
  --max-measurement-time MAX_MEASUREMENT_TIME
                        Maximalni delka mereni v sekundach. Default: 300
  --channel CHANNELS    Kanal ke stazeni, napr. CHAN1. Lze zadat vicekrat. Default: CHAN1, CHAN2
  --measurement-name MEASUREMENT_NAME
                        Jmeno mereni pouzite v nazvech vystupnich souboru.
  --high-res            Pouzit vysokorozlisene cteni waveformu.
  --normal-res          Pouzit normalni rozliseni waveformu.
  --run-once            Provest jen jedno mereni a pak skript ukoncit.
```



```
cd ~/git/osciloskop
python3 vxi_capture_multiple.py --scope osc=192.168.1.2 --outdir /home/kakl/DATA --samples 7000 --max-measurement-time 60 --measurement-name TEST2
```

### Zobrazení dat

```
python3 waveform_viewer.py <filename>
```



# Logovani spacedos


```
kakl@kakl-Latitude-5491:~/git/SPACEDOS04/sw$ ./log_spacedos.sh 
SPACEDOS data logger
To exit, press ctrl-c
usage example: ./log_spacedos.sh /dev/ttyUSB0 [description]
```






# Upload FW

ve slozce: `~/git/SPACEDOS04/fw/SPACEDOS04_EM`

  /home/kakl/.platformio/penv/bin/platformio run -t upload -e TFUNIPAYLOAD01_uart


