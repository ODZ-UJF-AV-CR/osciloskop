#!/usr/bin/python
import matplotlib.pyplot as plt
import sys
import os
import time
import h5py
import glob
import numpy as np

for f in glob.iglob("**/*.h5"): # generator, search immediate subdirectories 

    #print f
    with h5py.File(f,'r') as hf:
        #print f, str(np.array(hf.get('THETIME'))), str(np.array(hf.get('NSPART')))
        print np.datetime64(np.array(hf.get('THETIME')), 's')
        print str(np.array(hf.get('THETIME'))),str(np.array(hf.get('NSPART'))),str(np.array(hf.get('NSPART')))
        #print('List of arrays in this file: \n', hf.keys())
