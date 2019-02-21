#!/usr/bin/env bash

rm -rf data/sac_Ant-v2_reparam data/sac_Ant-v2_reparam_2qf
python train_mujoco.py --env_name Ant-v2 --exp_name reparam --reparam -e 3 --gpu 1
python train_mujoco.py --env_name Ant-v2 --exp_name reparam_2qf --reparam --two_qf -e 3 --gpu 1
python plot.py data/sac_Ant-v2_reparam data/sac_Ant-v2_reparam_2qf --save_name p2