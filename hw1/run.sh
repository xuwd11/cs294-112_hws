#!/bin/bash
set -eux
mkdir -p experiments
rm -rf ./experiments/*
for e in Hopper-v2
do
    python main.py $e --algorithm behavioral_cloning --save_name $e
    python main.py $e --algorithm dagger --save_name $e
done