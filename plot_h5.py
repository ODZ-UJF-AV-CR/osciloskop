#!/usr/bin/python

import matplotlib.pyplot as plt
import sys
import os
import time
import h5py
import glob
import numpy as np
import datetime as dt

plt.rcParams['figure.figsize'] = [18, 10]

for f in glob.iglob("./data_magnetic_loop/data*.h5"): # generator, search immediate subdirectories
    print '  ', f

    with h5py.File(f,'r') as hf:
        plt.figure()
        #print('List of arrays in this file: \n', hf.keys())
        print 'NORMAL'
        print '  XINC:', np.array(hf.get('XINC')), '   YINC:', np.array(hf.get('YINC')),
        print '   FRAMES:', np.array(hf.get('FRAMES')), '   TRIG:', np.array(hf.get('TRIG')),
        print '   XORIGIN:', np.array(hf.get('XORIGIN'))
        print '   YORIGIN:', np.array(hf.get('YORIGIN'))
        print '   TIME:   ', np.array(hf.get('TIME'))

        print 'RAW'
        print '  XINC:', np.array(hf.get('XINCR')), '   YINC:', np.array(hf.get('YINCR')),
        print '   FRAMES:', np.array(hf.get('FRAMES')), '   TRIG:', np.array(hf.get('TRIGR')),
        print '   XORIGIN:', np.array(hf.get('XORIGINR'))
        print '   YORIGIN:', np.array(hf.get('YORIGINR'))
        print '   TIME:   ', np.array(hf.get('THETIME'))
        print '   NSPART:   ', np.array(hf.get('NSPART'))


        frames = np.array(hf.get('FRAMES'))
        for n in range(1,frames+1):
            data = hf.get(str(n))
            np_data = 1.0 * np.array(data)
            #print 'np_data:', max(np_data)
            np_data = np_data - 128 - np.array(hf.get('YORIGIN'))
            np_data = np.array(hf.get('YINC')) * np_data    # to Volts
            np_time = range(0,1400)
            np_time *= np.array(hf.get('XINC'))
            np_time *= 1e6   # to microseconds
            #np_time -= 10    # trigger time offset
            np_time += 1.0e6*np.array(hf.get('XORIGIN'))
            #np_time *= 1.0e-3 # milliseconds are more fun

            # Process the raw data
            rawdata = hf.get(str(n)+'R')
            np_rawdata = 1.0 * np.array(rawdata)

            np_rawdata = np_rawdata - 128 - np.array(hf.get('YORIGINR'))
            np_rawdata = np.array(hf.get('YINCR')) * np_rawdata    # to Volts
            np_rawtime = range(0,len(np_rawdata))
            np_rawtime *= np.array(hf.get('XINCR'))
            np_rawtime *= 1e6   # to microseconds
            #np_rawtime -= 10    # trigger time offset
            np_rawtime += 1.0e6*np.array(hf.get('XORIGINR'))
            #np_rawtime *= 1.0e-3 # milliseconds are more fun
            plt.plot(np_rawtime,np_rawdata)
            #plt.plot(np_time, np_data)
        #plt.xlim(-100,100)
        plt.xlabel('Time [us]')
        plt.ylabel('Amplitude [V]')
        plt.title(dt.datetime.utcfromtimestamp(np.array(hf.get('THETIME'))))
        plt.savefig(f+'.png')
        plt.close()


