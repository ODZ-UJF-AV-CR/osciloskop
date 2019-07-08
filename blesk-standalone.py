#!/usr/bin/env python

# Standalone version of time-tagging script
# Based on Kakl's version for Micsig.
# MK, VS
# 
# Usage:
#  - Connect GPS antenna to time tagger-unit
#  - Configure RIGOL
#  - Connect Rigol Trigger Out BNC by cable to BNC input on time-tagger unit
#  - Connect USB

# Settings:
# --------
#
# Trigger out on Rigol works with cca 50% is possible, increasing duty cycle
# to above 80% as the time between triggers gets shorter.
#
# This was done with pulses generated every 1 ms from signal generator.
#
# Generally, the trigger period is roughly proportional to timespan window on Rigol,
# it also slightly depends on MemDepth.
#
# With MemDepth of 1.4 MPoints the HScale to trigger period translates as:
# H   5 ms =>  73 ms (63 frames, 20 MSa/s) => Leads to multiple triggers per 100 ms
# H  10 ms => 143 ms (63 frames, 10 MSa/s) => Leads to no more than 1 trigger per 100 ms
# H  20 ms => 284 ms (63 frames, 5 MSa/s)  => ...ditto, safe.
# 
# => In order to time-tag every frame with 100 ns holdoff and get 50 MSa/s, 
#    - hscale = 10 ms (=> 140 ms time span)
#    - MemDepth = 7 MPoints
#
# 50 ms hscale results in 0.7 s time span, two frames, 20 MSa/s (two channels).
# 100 ms hscale results in 1.4 s time span, two frames, 20 MSa/s (two channels).
#
# If frames with hscale shorter than 5 ms are requested, trigger holdoff can be set
# so that subsequent triggers are never closer in time than 100 ms. Some events will
# be skipped, but every recorded event will be time tagged.
#
# Note that for very short frames, this won't work either, because the GPS
# will not register each trigger event.

# Todo:
# - Connect this with readout of the frames from oscilloscope, so that the recorded files
#   are tagged with the GPS time
# - Verify the trigger horizontal position is saved 
# - Include precise position recording.
# - Include position recording into the data frames.

import matplotlib.pyplot as plt
import sys
import os
import time
import h5py
import numpy as np

import ublox # pyUblox library
import util
import datetime


# If the device was disconnected briefly, this can jump to ACM1 or whatever
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

