#!/usr/bin/env bash

rm -rf data/pm_gru_h-60_gs-10 data/pm_gru_h-60_gs-5
python train_policy.py 'pm' --exp_name gru_h-60_gs-10 --history 60 --discount 0.90 -lr 5e-4 -n 60 -gs 10 -rec -e 3 --gpu 1
python train_policy.py 'pm' --exp_name gru_h-60_gs-5 --history 60 --discount 0.90 -lr 5e-4 -n 60 -gs 5 -rec -e 3 --gpu 1