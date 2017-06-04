#!/bin/bash
TD=$(date +%y%m%d-%H%M%S-$1)
echo Moving current data to $TD
mkdir $TD || exit 1
mv -v data* $TD
cp README $TD
scp -r $TD hroch.ujf.cas.cz:/volume1/public/cosmic/Experiments/2017/06_himac/
