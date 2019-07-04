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
#gpsbaudrate = 9600
gpsbaudrate = 921600

# Open GPS port
dev = ublox.UBlox(gpsport, baudrate=gpsbaudrate, timeout=0)

print "# Listening at " + gpsport + "\n"
print "# >>> Do one FORCE trigger to get me ready."

# Read GPS messages, if any
last_count = -1
first_count = 0
while True:
	try:
		msg = dev.receive_message()
	except:
		continue

	if (msg is None):
		pass#break
	else:
		#print msg.name()
		if msg.name() == 'TIM_TM2':
			#print('Got TM2 message')
			#try:
			msg.unpack()
			if (last_count < 0):
				first_count = msg.count
				last_count = first_count
				print('# May the force be with you. Setting first count '+str(first_count))
				print('# >>> Now press RECORD on Rigol.')
			else:
				if (msg.count > last_count):
					timestring = '$HIT,'
					# Edges since first count
					timestring += str(msg.count-first_count)
					timestring += ','
					# Edges in this run
					timestring += str(msg.count-last_count)
					timestring += ','
					# Computer time
					timestring += str(datetime.datetime.utcnow())
					timestring += ','
					# This provides the rising edge time to microsecond precision
					timestring += str(util.gpsTimeToTime(msg.wnR, 1.0e-3*msg.towMsR + 1.0e-9*msg.towSubMsR))
					timestring += ','
					timestring += str(datetime.datetime.utcfromtimestamp(util.gpsTimeToTime(msg.wnR, 1.0e-3*msg.towMsR + 1.0e-9*msg.towSubMsR)))
					#timestring += str(datetime.datetime.utcfromtimestamp(util.gpsTimeToTime(msg.wnR, 1.0e6*(msg.towMsR % 1000 + msg.SubMsR)))
					timestring += ','
					# Tow sub millisecond fraction, in nanoseconds
					timestring += str(msg.towMsR)
					timestring += ','
					# Tow sub millisecond fraction, in nanoseconds
					timestring += str(msg.towSubMsR)
					print(timestring)
					sys.stdout.flush()
					last_count = msg.count
				else:
					print('# DUP')

			#except ublox.UBloxError as e:
			#		print(e)
			#break;

