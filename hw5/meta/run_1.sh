#!/usr/bin/env bash

rm -rf data/pm-obs_mlp_history-1
python train_policy.py 'pm-obs' --exp_name mlp_history-1 --history 1 -lr 5e-5 -n 200 --num_tasks 4 -e 3 
python plot.py data/pm-obs_mlp_history-1 --save_name p1