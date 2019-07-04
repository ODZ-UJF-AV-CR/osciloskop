#!/usr/bin/env python


import matplotlib.pyplot as plt
import sys
import os
import time
import h5py
import numpy as np

import ublox # pyUblox library
import util
import datetime


gpsport = '/dev/ttyACM0'
gpsbaudrate = 9600
#gpsbaudrate = 921600

# Open GPS port
dev = ublox.UBlox(gpsport, baudrate=gpsbaudrate, timeout=0)

print "# Listening at " + gpsport + "\n";

# Read GPS messages, if any
while True:
	try:
		msg = dev.receive_message()
	except:
		continue

	if (msg is None):
		pass#break
	else:
		print msg.name()
		if msg.name() == 'TIM_TM2':
			#print('Got TM2 message')
			try:
				msg.unpack()
				timestring = '$HIT,'
				timestring += str(msg.count)
				timestring += ','
				timestring += str(datetime.datetime.utcnow())
				timestring += ','
				timestring += str(datetime.datetime.utcfromtimestamp(util.gpsTimeToTime(msg.wnR, 1.0e-3*msg.towMsR)))
				print(timestring)
				sys.stdout.flush()

			except ublox.UBloxError as e:
				print(e)
			#break;

