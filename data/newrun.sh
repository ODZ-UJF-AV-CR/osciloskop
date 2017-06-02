#!/bin/bash
TD=$(date +%y%m%d-%H%M%S-$1)
echo Moving current data to $TD
mkdir $TD
mv -v data* $TD
