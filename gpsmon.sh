#!/bin/bash

while (sleep 2 ) ; do N=$( ls -1rt *.gps | tail -1 ) ; echo -n $N " " ; ../../pyUblox/ublox_show.py $N | grep TM2 | wc -l ; done
