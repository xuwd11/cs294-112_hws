#!/usr/bin/env bash

rm -rf data/pm_mlp_h-45_gs-1 data/pm_mlp_h-60_gs-1
python train_policy.py 'pm' --exp_name mlp_h-45_gs-1 --history 45 --discount 0.90 -lr 5e-4 -n 60 -gs 1 -e 3 --gpu 0
python train_policy.py 'pm' --exp_name mlp_h-60_gs-1 --history 60 --discount 0.90 -lr 5e-4 -n 60 -gs 1 -e 3 --gpu 0