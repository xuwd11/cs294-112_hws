#!/usr/bin/env bash

rm -rf data/pm_gru_h-60_gs-10_train data/pm_gru_h-60_gs-5_train data/pm_gru_h-60_gs-1_train
python train_policy.py 'pm' --exp_name gru_h-60_gs-10_train --history 60 --discount 0.90 -lr 5e-4 -n 60 -gs 10 -rec -st -e 3 --gpu 0
python train_policy.py 'pm' --exp_name gru_h-60_gs-5_train --history 60 --discount 0.90 -lr 5e-4 -n 60 -gs 5 -rec -st -e 3 --gpu 0
python train_policy.py 'pm' --exp_name gru_h-60_gs-1_train --history 60 --discount 0.90 -lr 5e-4 -n 60 -gs 1 -rec -st -e 3 --gpu 0