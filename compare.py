#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import sys
import os
import time
import h5py
import numpy as np
import glob
import datetime as dt
import pandas as pd

blitz = pd.read_csv("190721-u-Zdaru/blitz_2019_07_21_Zdar.txt", parse_dates=[0], header=0, usecols=[1,8])
blitz.set_index('time',drop=False,inplace=True)
blitzy =  np.array([1 for i in xrange(len(blitz))])
#blitzy = blitz['distance']

smycka = pd.read_csv("190721-u-Zdaru/trigger_times.csv",parse_dates=[0],header=0,usecols=[2])
smyckay =  np.array([1.2 for i in xrange(len(smycka))])
smycka.set_index('time',drop=False,inplace=True)

smyckaf = pd.read_csv("190721-u-Zdaru/trigger_times_filtered.csv",parse_dates=[0],header=0,usecols=[2])
smyckayf =  np.array([1.1 for i in xrange(len(smyckaf))])
smyckaf.set_index('time',drop=False,inplace=True)

fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
hfmt = dates.DateFormatter('%H:%M:%S')
ax.xaxis.set_major_locator(dates.MinuteLocator())
ax.xaxis.set_major_formatter(hfmt)

ax.xaxis.set_major_formatter(hfmt)
plt.plot(smycka['time'],smyckay,'g.',label='magnetic loop')
plt.plot(smyckaf['time'],smyckayf,'r.',label='magnetic loop - lightnings')
plt.plot(blitz['time'],blitzy,'b.', label='Blitzortung')

plt.ylim(0,3)
plt.tight_layout()
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=.2)
plt.legend()

plt.savefig('compare.png')
#plt.show()
