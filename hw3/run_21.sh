#!/bin/bash
rm -rf data/*_CartPole-v0
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_1 -ntu 1 -ngsptu 1 -nt --gpu 0
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 100_1 -ntu 100 -ngsptu 1 -nt --gpu 0
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_100 -ntu 1 -ngsptu 100 -nt --gpu 0
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 10_10 -ntu 10 -ngsptu 10 -nt --gpu 0
python plot.py data/ac_1_1_CartPole-v0 data/ac_100_1_CartPole-v0 data/ac_1_100_CartPole-v0 data/ac_10_10_CartPole-v0 --save_name p21