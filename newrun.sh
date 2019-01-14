#!/bin/bash

BASE=CERF
starttime=$( date "+%Y%m%d-%H%M%S" )
echo Start tag: $starttime
echo -n Enter description: 
read DESCRIPTION
D="$starttime-$BASE-$DESCRIPTION"

mkdir $D
mv -v *.h5 *.gps $D

