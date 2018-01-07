#!/usr/bin/env python

import ublox, sys, time, struct
import util
import datetime

from optparse import OptionParser

parser = OptionParser("ublox_capture_raw.py [options]")
parser.add_option("--port", help="serial port", default='/dev/ttyACM0')
parser.add_option("--baudrate", type='int',
                  help="serial baud rate", default=115200)
parser.add_option("--log", help="log file", default=None)
parser.add_option("--append", action='store_true', default=False, help='append to log file')
parser.add_option("--reopen", action='store_true', default=False, help='re-open on failure')
parser.add_option("--show", action='store_true', default=False, help='show messages while capturing')
parser.add_option("--dynModel", type='int', default=None, help='set dynamic navigation model')
parser.add_option("--usePPP", action='store_true', default=False, help='enable precise point positioning')
parser.add_option("--dots", action='store_true', default=False, help='print a dot on each message')


(opts, args) = parser.parse_args()

dev = ublox.UBlox(opts.port, baudrate=opts.baudrate, timeout=2)
dev.set_logfile(opts.log, append=opts.append)
dev.set_binary()
dev.configure_poll_port()
dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_USB)
dev.configure_poll(ublox.CLASS_MON, ublox.MSG_MON_HW)
dev.configure_port(port=ublox.PORT_SERIAL1, inMask=1, outMask=0)
dev.configure_port(port=ublox.PORT_USB, inMask=1, outMask=1)
dev.configure_port(port=ublox.PORT_SERIAL2, inMask=1, outMask=0)
dev.configure_poll_port()
dev.configure_poll_port(ublox.PORT_SERIAL1)
dev.configure_poll_port(ublox.PORT_SERIAL2)
dev.configure_poll_port(ublox.PORT_USB)
dev.configure_solution_rate(rate_ms=100)

dev.set_preferred_dynamic_model(opts.dynModel)
dev.set_preferred_usePPP(opts.usePPP)

dev.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_POSLLH, 0)
dev.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_POSECEF, 0)
dev.configure_message_rate(ublox.CLASS_RXM, ublox.MSG_RXM_RAW, 0)
dev.configure_message_rate(ublox.CLASS_RXM, ublox.MSG_RXM_SFRB, 0)

# Poll UTC time
dev.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_TIMEUTC, 0)
# Poll TM2 messages
dev.configure_message_rate(ublox.CLASS_TIM, ublox.MSG_TIM_TM2,1)

count_initialized = 0

while True:
    msg = dev.receive_message()
    if msg is None:
        if opts.reopen:
            dev.close()
            dev = ublox.UBlox(opts.port, baudrate=opts.baudrate, timeout=2)
            dev.set_logfile(opts.log, append=True)
            continue
        break
    elif opts.show:
        try:
            print(str(msg))
            sys.stdout.flush()
        except ublox.UBloxError as e:
            print(e)
    elif opts.dots:
        sys.stdout.write('.')
        sys.stdout.flush()
    if msg.name() == 'TIM_TM2':
		try:
			msg.unpack()
			if (count_initialized == 0):
				count_start = msg.count - 1
				last_count = msg.count
				first_towMsR = msg.towMsR
				count_initialized = 1
				print('First counter value: ' + str(count_start))
			if (msg.count < count_start):
				count_base += 65535;
			#timestring = datetime.datetime.strftime("%d %b %Y %H:%M:%S.%f", datetime.datetime.utcfromtimestamp(util.gpsTimeToTime(msg.wnR, 1.0e-3*msg.towMsR)))
			timestring = datetime.datetime.utcfromtimestamp(util.gpsTimeToTime(msg.wnR, 1.0e-3*msg.towMsR))
			txt = "Counter = %5d Zero = %5d Diff = %5d %s\n" % (msg.count-count_start, count_start, msg.count-last_count, timestring)
			print txt
			last_count = msg.count
		except ublox.UBloxError as e:
			print(e)
