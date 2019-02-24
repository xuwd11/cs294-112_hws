#!/usr/bin/env bash

rm -rf data/pm_gru_h-1
#python train_policy.py 'pm' --exp_name gru_h-1 --history 1 --discount 0.90 -lr 5e-4 -n 60 -rec -e 1 
#python plot.py data/pm-obs_mlp_h-1 --save_name p2