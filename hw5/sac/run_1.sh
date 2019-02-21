#!/usr/bin/env bash

rm -rf data/sac_HalfCheetah-v2_reinf data/sac_HalfCheetah-v2_reparam
python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reinf -e 3 --gpu 0
python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reparam --reparam -e 3 --gpu 0
python plot.py data/sac_HalfCheetah-v2_reinf data/sac_HalfCheetah-v2_reparam --save_name p1