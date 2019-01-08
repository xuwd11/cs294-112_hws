#!/bin/bash
set -eux
mkdir -p experiments
rm -rf ./experiments/*
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
    python main.py $e --algorithm behavioral_cloning --save_name $e
    python main.py $e --algorithm dagger --save_name $e
    python main.py $e --algorithm behavioral_cloning --save_name $e\_smooth-l1 --loss smooth_l1
done
python report.py