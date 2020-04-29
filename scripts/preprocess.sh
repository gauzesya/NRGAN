#!/bin/sh

# This script is for converting wav file to monaural 16kHz data

WAVE_DIR=$1 # Directory path to wav file
OUT_DIR=$2  # Output directory path

mkdir -p $OUT_DIR

for file in `find $WAVE_DIR -name '*.wav'`; do
  converted=`basename $file`
  sox $file -b 16 -r 16000 -c 1 $OUT_DIR/$converted
done
