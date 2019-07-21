#!/usr/bin/env python
#
# coding: utf-8
#
# Read raw memory from the Rigol oscilloscope
# VS and MK
#
# One big mess. Sorry.

import matplotlib.pyplot as plt
import sys
import os
import time
import h5py
import numpy as np
import glob
import vxi11

import ublox # pyUblox library
import util
import datetime

DEBUG = False
#DEBUG = True

def log(s):
	if DEBUG:
		print s

#------------------------------------------------------------------------------
# Delete data files if so requested
filename = 0
    
if (filename == 1):
    for f in glob.iglob("./data/*.h5"): # delete all .h5 files 
        print 'Deleting', f
        os.remove(f)
else:
    print 'Not removing old files, as filename {0} is not 1.'.format(filename)

# Set the initial delay for frame capture
#start_wfd = 0.005 #0.05 #0.012
start_wfd = 0.02
wfd=start_wfd
#------------------------------------------------------------------------------

# This will need a rewrite
class TmcDriver:

    def __init__(self, device):
        print("Initializing connection to: " + device)
        self.device = device
        self.instr = vxi11.Instrument(device)
 
    def write(self, command):
        #print command
        self.instr.write(command);

    def read(self, length = 500):
        return self.instr.read(length)

    def read_raw(self, length = 500):
        return self.instr.read_raw(length)
 
    def getName(self):
        self.write("*IDN?")
        return self.read(300)
    
    def ask(self, command):
        return self.instr.ask(command)
 
    def sendReset(self):
        self.write("*RST")  # Be carefull, this real resets an oscilloscope

def read_this_frame_raw():
        osc.write(':WAV:MODE RAW')
        osc.write(':WAV:FORM BYTE')
        osc.write(':ACQ:MDEP?')
        mdepth = float(osc.read(100))
        print 'MDEPTH:', int(mdepth),

        osc.write(':STOP')
        time.sleep(1)

	retry_count = 3
        while (retry_count > 0):
		retry_count = retry_count - 1

		osc.write(':WAV:STAR 1')
		time.sleep(0.5)

		osc.write(':WAV:STOP '+"{0:.0f}".format(mdepth))
		time.sleep(0.5)

		osc.write(':WAV:RES')
		time.sleep(1)

		osc.write(':WAV:BEG')
		time.sleep(1)

		wavepart = bytearray()
		lastwave = bytearray()

		bytesread = 0
		flag = 1
		nowwaiting = 0
		chrup = 0.25

		# We will try to read the waveform three times
		while (nowwaiting < 5.0):
			time.sleep(0.2)
			status = osc.ask(':WAV:STAT?')
			read_status = status[:4]
			bytes_to_read = int(status[5:])
			bytesread = bytesread + bytes_to_read
			log(' '+status)
			if (read_status == 'IDLE' and bytes_to_read == 0):
				break
			
			if (read_status == 'READ' or read_status == 'IDLE'):
				if (bytes_to_read > 0):            
					osc.write(':WAV:DATA?')
					time.sleep(0.2)
					wavepart = bytearray(osc.read_raw(bytes_to_read))
					lastwave = lastwave + wavepart
					sys.stdout.write(' '+str(len(wavepart))+' ')
					sys.stdout.flush()
					time.sleep(0)
					nowwaiting = 0
				else:
					# There are no data to read now
					time.sleep(chrup)
					nowwaiting = nowwaiting + chrup
					
			if (flag == 0):
				break
			if (read_status == 'IDLE'):
				flag = 0

		osc.write(':WAV:END')

		if (len(lastwave) == mdepth and mdepth == bytesread):
			log('  Expected and read bytes match: '+str(mdepth))
			break
		else:
			print '  Adjust delay after BEG - we are loosing data somewhere.'
        return(lastwave)
        
def get_one_frame_raw(thetime,ns):
    global filename
    global wfd
    while True:
        osc.write(':FUNC:WREC:OPER?') # finish recording?
        reply = osc.read()
        if reply == 'STOP':
            run_time = round(time.time() - run_start_time, 2)
            log('  Frame capture finished after %.2f seconds.' % run_time)
            break
        time.sleep(0.05)

    for channel in ['CHAN1','CHAN2']:
        if (osc.ask(':'+channel+':DISP?') == u'0'):
            log(channel+' is not enabled')
        else:
            log('Reading out '+channel)
            osc.write(':WAV:SOUR '+channel)
            osc.write(':WAV:MODE RAW')
            osc.write(':WAV:FORM BYTE')

            print 'TIME:', thetime,
            print 'NSPART:', ns

            # Read oscilloscope settings
            osc.write(':ACQ:MDEP?')
            mdepth = float(osc.read(100))
            print 'MDEPTH:', int(mdepth),

            osc.write(':WAV:XINC?')
            xinc = float(osc.read(100))
            print 'XINC:', xinc,
            osc.write(':WAV:YINC?')
            yinc = float(osc.read(100))
            print 'YINC:', yinc,
            osc.write(':TRIGger:EDGe:LEVel?')
            trig = float(osc.read(100))
            print 'TRIG:', trig,
            osc.write(':WAVeform:YORigin?')
            yorig = float(osc.read(100))
            print 'YORIGIN:', yorig,
            osc.write(':WAVeform:XORigin?')
            xorig = float(osc.read(100))
            print 'XORIGIN:', xorig,

            osc.write(':WAVeform:XREFerence?')
            xref = float(osc.read(100))
            print 'XREF:', xref

            osc.write(':FUNC:WREP:FEND?') # get number of last frame
            frames = int(osc.read(100))
            print '  FRAMES:', frames, 'SUBRUN', filename

            wave = bytearray()
            lastwave = bytearray()

	    with h5py.File('./data/data'+'{:02.0f}'.format(filename)+'_'+str(int(round(run_start_time,0)))+'-'+channel+'.h5', 'w') as hf:       
		hf.create_dataset('FRAMES', data=(frames)) # write number of frames
		hf.create_dataset('XINC', data=(xinc)) # write axis parameters
		hf.create_dataset('YINC', data=(yinc))
		hf.create_dataset('TRIG', data=(trig))
		hf.create_dataset('YORIGIN', data=(yorig))
		hf.create_dataset('XORIGIN', data=(xorig))
		hf.create_dataset('XREFERENCE', data=(xref))
		hf.create_dataset('THETIME', data=(thetime))
		hf.create_dataset('NSPART', data=(ns))
		hf.create_dataset('CAPTURING', data=(run_time))
		if (frames > 1):
			osc.write(':FUNC:WREP:FCUR 1') # go to first frame
		time.sleep(0.5)
		for n in range(1,frames+1):
		    if (frames > 1):
			osc.write(':FUNC:WREP:FCUR ' + str(n)) # skip to n-th frame
		    while True:
			time.sleep(0.05)
			fcur = osc.ask(':FUNC:WREP:FCUR?')
			if (str(n) == fcur):
			    # Rigol returns correct current frame
			    sys.stdout.write(str(n))
			    break
			else:
			    # Rigol has not yet made the seek, wait
			    # or consider extending the sleep above
			    print("Needwait: "+str(n)+' vs '+fcur)

		    reread_count = 0
		    while True:
                        # Read this frame
                        wave = read_this_frame_raw()
                            
			if (np.array_equal(wave, lastwave)):
			    wfd = wfd + 0.005
			    print(' Same waveform, wait ' + str(wfd) + ' and reread')
			    reread_count = reread_count + 1
			    if (reread_count > 5):
				print('------------ Wrong trigger level?')
			else:
			    hf.create_dataset(str(n), data=wave)
			    lastwave = wave
			    sys.stdout.write('O')
			    wfd = start_wfd
			    break
	    print 'K'
	    # End of channel recording


def get_one_frame(thetime,ns):
    global filename
    global wfd
    while True:
        osc.write(':FUNC:WREC:OPER?') # finish recording?
        reply = osc.read()
        if reply == 'STOP':
            run_time = round(time.time() - run_start_time, 2)
            log('  Frame capture finished after %.2f seconds.' % run_time)
            break
        time.sleep(0.05)

    for channel in ['CHAN1','CHAN2']:
        if (osc.ask(':'+channel+':DISP?') == u'0'):
            log(channel+' is not enabled')
        else:
            log('Reading out '+channel)
            osc.write(':WAV:SOUR '+channel)

            print '  TIME:', thetime,
            print '  NSPART:', ns
            
            #------------------------------
            # RAW
            print '  RAW    ',
            osc.write(':WAV:MODE RAW')
            osc.write(':WAV:FORM BYTE')
            #osc.write(':WAV:POIN 28000')
            
            osc.write(':WAV:XINC?')
            xincr = float(osc.read(100))
            print 'XINC:', xincr,
            osc.write(':WAV:YINC?')
            yincr = float(osc.read(100))
            print 'YINC:', yincr,
            osc.write(':TRIGger:EDGe:LEVel?')
            trigr = float(osc.read(100))
            print 'TRIG:', trigr,
            osc.write(':WAVeform:YORigin?')
            yorigr = float(osc.read(100))
            print 'YORIGIN:', yorigr,
            osc.write(':WAVeform:XORigin?')
            xorigr = float(osc.read(100))
            print 'XORIGIN:', xorigr,
            osc.write(':WAVeform:XREFerence?')
            xrefr = float(osc.read(100))
            print 'XREF:', xrefr
            #-------------------------------

            print '  NORMAL ',
            # NORM
            osc.write(':WAV:MODE NORM')
            osc.write(':WAV:FORM BYTE')
            osc.write(':WAV:POIN 1400')

            osc.write(':WAV:XINC?')
            xinc = float(osc.read(100))
            print 'XINC:', xinc,
            osc.write(':WAV:YINC?')
            yinc = float(osc.read(100))
            print 'YINC:', yinc,
            osc.write(':TRIGger:EDGe:LEVel?')
            trig = float(osc.read(100))
            print 'TRIG:', trig,
            osc.write(':WAVeform:YORigin?')
            yorig = float(osc.read(100))
            print 'YORIGIN:', yorig,
            osc.write(':WAVeform:XORigin?')
            xorig = float(osc.read(100))
            print 'XORIGIN:', xorig,

            osc.write(':WAVeform:XREFerence?')
            xref = float(osc.read(100))
            print 'XREF:', xref,
            #---------------------------

                       
            osc.write(':FUNC:WREP:FEND?') # get number of last frame
            frames = int(osc.read(100))
            print 'FRAMES:', frames, 'SUBRUN', filename

            lastwave = bytearray()
            
	    with h5py.File('./data/data'+'{:02.0f}'.format(filename)+'_'+str(int(round(run_start_time,0)))+'-'+channel+'.h5', 'w') as hf:       
		hf.create_dataset('FRAMES', data=(frames)) # write number of frames
		hf.create_dataset('XINC', data=(xinc)) # write axis parameters
		hf.create_dataset('YINC', data=(yinc))
		hf.create_dataset('TRIG', data=(trig))
		hf.create_dataset('YORIGIN', data=(yorig))
		hf.create_dataset('XORIGIN', data=(xorig))
		hf.create_dataset('XREFERENCE', data=(xref))

                # These are for the RAW writeout
		hf.create_dataset('XINCR', data=(xincr)) # write axis parameters
		hf.create_dataset('YINCR', data=(yincr))
		hf.create_dataset('TRIGR', data=(trigr))
		hf.create_dataset('YORIGINR', data=(yorigr))
		hf.create_dataset('XORIGINR', data=(xorigr))
		hf.create_dataset('XREFERENCER', data=(xrefr))

		hf.create_dataset('THETIME', data=(thetime))
		hf.create_dataset('NSPART', data=(ns))
		hf.create_dataset('CAPTURING', data=(run_time))
		if (frames > 1):
			osc.write(':FUNC:WREP:FCUR 1') # go to first frame
		time.sleep(0.5)
		for n in range(1,frames+1):
		    if (frames > 1):
			osc.write(':FUNC:WREP:FCUR ' + str(n)) # skip to n-th frame
		    while True:
			time.sleep(0.05)
			fcur = osc.ask(':FUNC:WREP:FCUR?')
			if (str(n) == fcur):
			    # Rigol returns correct current frame
			    print(str(n)+':')
			    break
			else:
			    # Rigol has not yet made the seek, wait
			    # or consider extending the sleep above
			    print("Needwait: "+str(n)+' vs '+fcur)

		    reread_count = 0
                    lastwave = bytearray()
                    # Now, get the frame in NORMAL mode
                    print '  NORMAL',
                    osc.write(':WAV:MODE NORM')
                    osc.write(':WAV:FORM BYTE')
                    osc.write(':WAV:POIN 1400')

		    while True:
			time.sleep(wfd)        
			osc.write(':WAV:DATA?') # read data
			time.sleep(wfd)
			wave1 = bytearray(osc.read_raw(500))
			wave2 = bytearray(osc.read_raw(500))
			wave3 = bytearray(osc.read_raw(500))
			#wave4 = bytearray(osc.read(500))
			#wave = np.concatenate((wave1[11:],wave2[:(500-489)],wave3[:(700-489)]))
			wave = np.concatenate((wave1[11:],wave2,wave3[:-1]))
			if (np.array_equal(wave, lastwave)):
			    wfd = wfd + 0.005
			    print(' Same waveform, wait ' + str(wfd) + ' and reread')
			    reread_count = reread_count + 1
			    if (reread_count > 5):
				print('------------ Wrong trigger level?')
			else:
			    hf.create_dataset(str(n), data=wave)
			    lastwave = wave
			    print(' OK ')
			    wfd = start_wfd
			    break

                    # Read this frame in RAW mode
                    reread_count = 0
                    lastwave = bytearray()

                    print '  RAW',
		    while True:
                        # Read this frame
                        wave = read_this_frame_raw()
                            
			if (np.array_equal(wave, lastwave)):
			    wfd = wfd + 0.005
			    print(' Same waveform, wait ' + str(wfd) + ' and reread')
			    reread_count = reread_count + 1
			    if (reread_count > 5):
				print('------------ Wrong trigger level?')
			else:
			    hf.create_dataset(str(n)+'R', data=wave)
			    lastwave = wave
			    print(' OK')
			    wfd = start_wfd
			    break
                print ' '
	    # End of channel recording


# Initialize ethernet link to oscilloscope

# For Ethernet
osc = TmcDriver("TCPIP::10.1.1.3::INSTR")
print(osc.ask("*IDN?"))


osc.write(':STOP') # go to STOP mode
time.sleep(0.5)

# Initialize GPS serial link
# If the device was disconnected briefly, this can jump to ACM1 or whatever
gpsport = '/dev/ttyACM0'
gpsbaudrate = 921600

# Open GPS port
dev = ublox.UBlox(gpsport, baudrate=gpsbaudrate, timeout=0)

print "# Configured to listen at " + gpsport, ' forcing trigger.'

# Prepare the oscilloscope 
# Switch to single trigger mode and do one trigger
time.sleep(1)
osc.write(':SINGle')
time.sleep(1)
osc.write(':TFORce')

# ----------------------------
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
				print('# May the force be with you. Setting first count to '+str(first_count))
				run_start_time = time.time()
				time.sleep(wfd)
				log('  Starting single capture at time: %.2f' % run_start_time)
				osc.write(':SINGle')
			else:
				# If this is not a duplicate message
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
					log(timestring)
					sys.stdout.flush()
					last_count = msg.count
					# Get the data frame from oscilloscope
					get_one_frame(util.gpsTimeToTime(msg.wnR, 1.0e-3*msg.towMsR + 1.0e-9*msg.towSubMsR), msg.towSubMsR)
					filename = filename + 1
					# and go again for single capture
					time.sleep(0.5)
					run_start_time = time.time()
					log('  Run start time: %.2f' % run_start_time)
					log('  Capturing...')
					osc.write(':SINGle')
#
#				else:
#					print('# DUP')

