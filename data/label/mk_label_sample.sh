#!/bin/sh

find '../wav/original' -name 'V*.wav' | sort > train_denoise
find '../wav/noised' -name 'V*.wav' | sort > train_noised

find '../wav/original' -name '[!V]*.wav' | sort > test_denoise
find '../wav/noised' -name '[!V]*.wav' | sort > test_noised
