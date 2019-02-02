#!/bin/bash
rm -rf data/*_InvertedPendulum-v2 data/*_HalfCheetah-v2
python train_ac_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.95 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name 10_10 -ntu 10 -ngsptu 10 -nt --gpu 1
python train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --exp_name 10_10 -ntu 10 -ngsptu 10 -nt --gpu 1
python plot.py data/ac_10_10_InvertedPendulum-v2 data/ac_10_10_HalfCheetah-v2 --legend ac_10_10_InvertedPendulum-v2 ac_10_10_HalfCheetah-v2 --save_name p22