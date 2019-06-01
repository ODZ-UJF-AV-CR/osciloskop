#!/usr/bin/env python

import ublox # pyUblox librarie
import util
import datetime
import sys

gpsport = '/dev/ttyACM0'
gpsbaudrate = 9600

# Reopen GPS port
dev = ublox.UBlox(gpsport, baudrate=gpsbaudrate, timeout=0)

try:
	# Read GPS messages, if any
	while True:
		msg = dev.receive_message()
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
					timestring += str(datetime.datetime.utcfromtimestamp(util.gpsTimeToTime(msg.wnR, 1.0e-3*msg.towMsR)))
					print(timestring)
					sys.stdout.flush()
				except ublox.UBloxError as e:
					print(e)
				#break;
except:
	dev.close()
