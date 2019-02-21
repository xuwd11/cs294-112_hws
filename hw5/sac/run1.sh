#!/usr/bin/env bash

rm -rf data/sac_HalfCheetah-v2_reinf
python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reinf -e 3