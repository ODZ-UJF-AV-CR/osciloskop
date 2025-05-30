#!/usr/bin/env python


import matplotlib.pyplot as plt
import sys
import os
import time
import h5py
import numpy as np

import ublox # pyUblox librarie
import util
import datetime

# TRYING TO ADAPT THE SCRIPT FOR THE MICSIG oscilloscope

# Create ownership rule as /etc/udev/rules.d/99-micsig.rules
# SUBSYSTEMS=="usb", ATTRS{idVendor}=="18d1", ATTRS{idProduct}=="0303", GROUP="medved", MODE="0666"

class UsbTmcDriver:

    def __init__(self, device):
        self.device = device
        self.FILE = os.open(device, os.O_RDWR)
 
    def write(self, command):
        os.write(self.FILE, command);
 
    def read(self, length = 2048):
        return os.read(self.FILE, length)
 
    def getName(self):
        self.write("*IDN?")
        return self.read(300)
 
    def sendReset(self):
        self.write("*RST")  # Be carefull, this real resets an oscilloscope

# Looking for USBTMC device
def getDeviceList(): 
    dirList=os.listdir("/dev")
    result=list()

    for fname in dirList:
        if(fname.startswith("usbtmc")):
            result.append("/dev/" + fname)

    return result


# looking for oscilloscope
devices =  getDeviceList()
# initiate oscilloscope
osc = UsbTmcDriver(devices[0])

print "$OSC,", osc.getName(),
osc.write("MENU:RUN")


gpsport = '/dev/ttyACM0'
gpsbaudrate = 921600

# Open GPS port
dev = ublox.UBlox(gpsport, baudrate=gpsbaudrate, timeout=0)

while True:
	'''
	while True:
		try:
			msg = dev.receive_message()
		except:
			continue
		if (msg is None):
			break
	'''
	osc.write(":SINGLE")

	# Read GPS messages, if any
	while True:
		try:
			msg = dev.receive_message()
		except:
			continue
		if (msg is None):
			pass#break
		else:
			if msg.name() == 'TIM_TM2':
				#print('Got TM2 message')
				try:
					msg.unpack()
					timestring = '$HIT,'
					timestring += str(msg.count)
					timestring += ','
					timestring += str(datetime.datetime.utcnow())
					timestring += ','
					filename = util.gpsTimeToTime(msg.wnR, 1.0e-3*msg.towMsR + 1.0e-9*msg.towSubMsR)
					timestring += str(filename)
					timestring += ','
					timestring += str(datetime.datetime.utcfromtimestamp(filename))
					print(timestring)
					sys.stdout.flush()
			
					''' Micsig
					osc.write("MENU:STOP")
					time.sleep(1)
					osc.write(':STORage:CAPTure')
					time.sleep(2)
					#osc.write(':STORage:SAVECH1,UDISK')
					#time.sleep(5)
					osc.write("MENU:RUN")
					#time.sleep(.1s)
					'''
					#''' RIgol
					time.sleep(5)			//!!! Wait for the end of capturing
					#osc.write(":STOP")
					#time.sleep(1)
					osc.write(':SAVE:WAVeform D:\\blesky\\' + str(filename) + '.wfm')
					time.sleep(1)
					osc.write(":RUN")
					raw_input()
					break
					#'''
		

				except ublox.UBloxError as e:
					print(e)
					break

