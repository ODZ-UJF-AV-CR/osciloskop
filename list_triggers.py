#!/usr/bin/python
import matplotlib.pyplot as plt
import sys
import os
import time
import h5py
import glob
import numpy as np
import datetime as dt

print 'epoch,nspart,time'
for f in glob.iglob("./190721-u-Zdaru/data_magnetic_loop_selected/*.h5"): # generator, search immediate subdirectories 

    #print f
    with h5py.File(f,'r') as hf:
        #print f, str(np.array(hf.get('THETIME'))), str(np.array(hf.get('NSPART')))
        #print str(np.array(hf.get('THETIME'))),str(np.array(hf.get('NSPART'))),str(np.array(hf.get('NSPART')))
        print str(np.array(hf.get('THETIME')))+','+str(np.array(hf.get('NSPART')))+','+str(dt.datetime.utcfromtimestamp(np.array(hf.get('THETIME'))))
        #print('List of arrays in this file: \n', hf.keys())
